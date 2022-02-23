use byteorder::{BigEndian, ByteOrder};
use bytes::{Bytes,BytesMut, BufMut};
use std::io;
use std::error;
use std::fmt;
use std::cmp::Ordering;
use thiserror::Error;

// Static assertion: We expect the system architecture bus width to be >= 32 bits. If it is not,
// the following line will cause a compiler error. (Ignore the unrelated error message itself.)
const _: usize = 0 - !(std::mem::size_of::<usize>() >= 32 >> 3) as usize;

/// An in-memory log array
#[derive(Clone)]
pub struct LogArray {
    /// Number of accessible elements
    ///
    /// For an original log array, this is initialized to the value read from the control word. For
    /// a slice, it is the length of the slice.
    len: u32,

    /// Bit width of each element
    width: u8,

    /// Shared reference to the input buffer
    ///
    /// Index 0 points to the first byte of the first element. The last word is the control word.
    input_buf: Bytes,
}

/// An error that occurred during a log array operation.
#[derive(Debug, PartialEq)]
pub enum LogArrayError {
    InputBufferTooSmall(usize),
    WidthTooLarge(u8),
    UnexpectedInputBufferSize(u64, u64, u32, u8),
}

impl LogArrayError {
    /// Validate the input buffer size.
    ///
    /// It must have at least the control word.
    fn validate_input_buf_size(input_buf_size: usize) -> Result<(), Self> {
        if input_buf_size < 8 {
            return Err(LogArrayError::InputBufferTooSmall(input_buf_size));
        }
        Ok(())
    }

    /// Validate the number of elements and bit width against the input buffer size.
    ///
    /// The bit width should no greater than 64 since each word is 64 bits.
    ///
    /// The input buffer size should be the appropriate multiple of 8 to include the exact number
    /// of encoded elements plus the control word.
    fn validate_len_and_width(input_buf_size: usize, len: u32, width: u8) -> Result<(), Self> {
        if width > 64 {
            return Err(LogArrayError::WidthTooLarge(width));
        }

        // Calculate the expected input buffer size. This includes the control word.
        // To avoid overflow, convert `len: u32` to `u64` and do the addition in `u64`.
        let expected_buf_size = u64::from(len) * u64::from(width) + 127 >> 6 << 3;
        let input_buf_size = u64::try_from(input_buf_size).unwrap();

        if input_buf_size != expected_buf_size {
            return Err(LogArrayError::UnexpectedInputBufferSize(
                input_buf_size,
                expected_buf_size,
                len,
                width,
            ));
        }

        Ok(())
    }
}

impl fmt::Display for LogArrayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LogArrayError::*;
        match self {
            InputBufferTooSmall(input_buf_size) => {
                write!(f, "expected input buffer size ({}) >= 8", input_buf_size)
            }
            WidthTooLarge(width) => write!(f, "expected width ({}) <= 64", width),
            UnexpectedInputBufferSize(input_buf_size, expected_buf_size, len, width) => write!(
                f,
                "expected input buffer size ({}) to be {} for {} elements and width {}",
                input_buf_size, expected_buf_size, len, width
            ),
        }
    }
}

impl error::Error for LogArrayError {}

impl From<LogArrayError> for io::Error {
    fn from(err: LogArrayError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, err)
    }
}

#[derive(Clone)]
pub struct LogArrayIterator {
    logarray: LogArray,
    pos: usize,
    end: usize,
}

impl Iterator for LogArrayIterator {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.pos == self.end {
            None
        } else {
            let result = self.logarray.entry(self.pos);
            self.pos += 1;

            Some(result)
        }
    }
}

/// Read the length and bit width from the control word buffer. `buf` must start at the first word
/// after the data buffer. `input_buf_size` is used for validation.
fn read_and_validate_control_word(buf: &[u8], input_buf_size: usize) -> Result<(u32, u8), LogArrayError> {
    let (len, width) = read_control_word(buf)?;
    LogArrayError::validate_len_and_width(input_buf_size, len, width)?;
    Ok((len, width))
}

/// Read the length and bit width from the control word buffer.
pub fn read_control_word(buf: &[u8]) -> Result<(u32, u8), LogArrayError> {
    assert_eq!(8, buf.len());
    let len = BigEndian::read_u32(buf);
    let width = buf[4];
    Ok((len, width))
}

impl LogArray {
    /// Construct a `LogArray` by parsing a `Bytes` buffer.
    pub fn parse(input_buf: Bytes) -> Result<LogArray, LogArrayError> {
        let input_buf_size = input_buf.len();
        LogArrayError::validate_input_buf_size(input_buf_size)?;
        let (len, width) = read_and_validate_control_word(&input_buf[input_buf_size - 8..], input_buf_size)?;
        Ok(LogArray {
            len,
            width,
            input_buf,
        })
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        // `usize::try_from` succeeds if `std::mem::size_of::<usize>()` >= 4.
        usize::try_from(self.len).unwrap()
    }

    /// Returns `true` if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the bit width.
    pub fn width(&self) -> u8 {
        self.width
    }

    /// Reads the data buffer and returns the element at the `index`.
    ///
    /// Panics if `index` is >= the length of the log array.
    pub fn entry(&self, index: usize) -> u64 {
        assert!(
            index < self.len(),
            "expected index ({}) < length ({})",
            index,
            self.len
        );

        // `usize::try_from` succeeds if `std::mem::size_of::<usize>()` >= 4.
        let bit_index = (self.width as usize) * index;

        // Read the words that contain the element.
        let (first_word, second_word) = {
            // Calculate the byte index from the bit index.
            let byte_index = bit_index >> 6 << 3;

            let buf = &self.input_buf;

            // Read the first word.
            let first_word = BigEndian::read_u64(&buf[byte_index..]);

            // Read the second word (optimistically).
            //
            // This relies on the buffer having the control word at the end. If that is not there,
            // this may panic.
            let second_word = BigEndian::read_u64(&buf[byte_index + 8..]);

            (first_word, second_word)
        };

        // This is the minimum number of leading zeros that a decoded value should have.
        let leading_zeros = 64 - self.width;

        // Get the bit offset in `first_word`.
        let offset = (bit_index & 0b11_1111) as u8;

        // If the element fits completely in `first_word`, we can return it immediately.
        if offset + self.width <= 64 {
            // Decode by introducing leading zeros and shifting all the way to the right.
            return first_word << offset >> leading_zeros;
        }

        // At this point, we have an element split over `first_word` and `second_word`. The bottom
        // bits of `first_word` become the upper bits of the decoded value, and the top bits of
        // `second_word` become the lower bits of the decoded value.

        // These are the bit widths of the important parts in `first_word` and `second_word`.
        let first_width = 64 - offset;
        let second_width = self.width - first_width;

        // These are the parts of the element with the unimportant parts removed.

        // Introduce leading zeros and trailing zeros where the `second_part` will go.
        let first_part = first_word << offset >> offset << second_width;

        // Introduce leading zeros where the `first_part` will go.
        let second_part = second_word >> 64 - second_width;

        // Decode by combining the first and second parts.
        first_part | second_part
    }

    pub fn iter(&self) -> LogArrayIterator {
        LogArrayIterator {
            logarray: self.clone(),
            pos: 0,
            end: self.len(),
        }
    }

    /// Returns a logical slice of the elements in a log array.
    ///
    /// Panics if `index` + `length` is >= the length of the log array.
    pub fn slice(&self, offset: usize, len: usize) -> LogArraySlice {
        let offset = u32::try_from(offset)
            .unwrap_or_else(|_| panic!("expected 32-bit slice offset ({})", offset));
        let len =
            u32::try_from(len).unwrap_or_else(|_| panic!("expected 32-bit slice length ({})", len));
        let slice_end = offset.checked_add(len).unwrap_or_else(|| {
            panic!("overflow from slice offset ({}) + length ({})", offset, len)
        });
        assert!(
            slice_end <= self.len,
            "expected slice offset ({}) + length ({}) <= source length ({})",
            offset,
            len,
            self.len
        );

        LogArraySlice {
            original: self.clone(),
            offset: offset,
            len: len
        }
    }

    pub fn bytes(&self) -> &Bytes {
        &self.input_buf
    }
}

pub struct LogArraySlice {
    original: LogArray,
    offset: u32,
    len: u32,
}

impl LogArraySlice {
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> u8 {
        self.original.width()
    }

    pub fn entry(&self, index: usize) -> u64 {
        assert!(
            index < self.len(),
            "expected index ({}) < length ({})",
            index,
            self.len
        );

        self.original.entry(self.offset as usize + index)
    }

    pub fn iter(&self) -> LogArrayIterator {
        LogArrayIterator {
            logarray: self.original.clone(),
            pos: self.offset as usize,
            end: (self.offset + self.len) as usize,
        }
    }

    pub fn slice(&self, offset: usize, len: usize) -> LogArraySlice {
        self.original.slice(self.offset as usize+offset, len)
    }

    pub fn original(&self) -> &LogArray {
        &self.original
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }
}

#[derive(Clone)]
pub struct MonotonicLogArray(LogArray);

impl MonotonicLogArray {
    pub fn from_logarray(logarray: LogArray) -> MonotonicLogArray {
        if cfg!(debug_assertions) {
            // Validate that the elements are monotonically increasing.
            let mut iter = logarray.iter();
            if let Some(mut pred) = iter.next() {
                for succ in iter {
                    assert!(
                        pred <= succ,
                        "not monotonic: expected predecessor ({}) <= successor ({})",
                        pred,
                        succ
                    );
                    pred = succ;
                }
            }
        }

        Self::from_logarray_unchecked(logarray)
    }

    pub fn from_logarray_unchecked(logarray: LogArray) -> MonotonicLogArray {
        MonotonicLogArray(logarray)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn entry(&self, index: usize) -> u64 {
        self.0.entry(index)
    }

    pub fn iter(&self) -> LogArrayIterator {
        self.0.iter()
    }

    pub fn index_of(&self, element: u64) -> Option<usize> {
        let index = self.nearest_index_of(element);
        if index >= self.len() || self.entry(index) != element {
            None
        } else {
            Some(index)
        }
    }

    pub fn nearest_index_of(&self, element: u64) -> usize {
        if self.is_empty() {
            return 0;
        }

        let mut min = 0;
        let mut max = self.len() - 1;
        while min <= max {
            let mid = (min + max) / 2;
            match element.cmp(&self.entry(mid)) {
                Ordering::Equal => return mid,
                Ordering::Greater => min = mid + 1,
                Ordering::Less => {
                    if mid == 0 {
                        return 0;
                    }
                    max = mid - 1
                }
            }
        }

        (min + max) / 2 + 1
    }

    pub fn bytes(&self) -> &Bytes {
        self.0.bytes()
    }
}

impl From<LogArray> for MonotonicLogArray {
    fn from(l: LogArray) -> Self {
        Self::from_logarray(l)
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum LogArrayBuilderError {
    #[error("pushed number {number} is too large for width {width}")]
    PushedNumberTooLargeForWidth {
        number: u64,
        width: u8
    }
}

pub struct LogArrayBuilder {
    bytes: BytesMut,
    width: u8,
    len: u32,
    current: u64,
}

impl LogArrayBuilder {
    pub fn new(width: u8) -> Self {
        Self {
            bytes: BytesMut::new(),
            width,
            current: 0,
            len: 0
        }
    }

    pub fn push(&mut self, number: u64) -> Result<(), LogArrayBuilderError> {
        let leading_zeros = 64-(self.width as u32);
        if number.leading_zeros() < leading_zeros {
            return Err(LogArrayBuilderError::PushedNumberTooLargeForWidth{number, width: self.width});
        }

        let index_in_u64 = find_position_of_element(self.width, self.len);

        self.current |= number << leading_zeros >> index_in_u64;
        if index_in_u64 as u32 > leading_zeros {
            self.bytes.put_u64(self.current);

            self.current = number << (leading_zeros+index_in_u64 as u32);
        }

        self.len += 1;

        Ok(())
    }

    pub fn push_slice(&mut self, slice: &[u64]) -> Result<(), LogArrayBuilderError> {
        let index_in_u64 = find_position_of_element(self.width, self.len);
        let expansion = extra_u64s_taken_up(self.width, index_in_u64, slice.len());

        self.bytes.reserve(expansion*8);

        for &element in slice.iter() {
            self.push(element)?;
        }

        Ok(())
    }

    pub fn push_iter<I: Iterator<Item=u64>>(&mut self, iter: I) -> Result<(), LogArrayBuilderError> {
        for element in iter {
            self.push(element)?;
        }

        Ok(())
    }

    pub fn finalize(mut self) -> Bytes {
        let index_in_u64 = find_position_of_element(self.width, self.len);
        if index_in_u64 != 0 {
            self.bytes.put_u64(self.current);
        }

        self.bytes.put_u32(self.len);
        self.bytes.put_u8(self.width);
        self.bytes.put([0,0,0].as_slice());

        self.bytes.into()
    }
}

fn find_position_of_element(width: u8, index: u32) -> u8 {
    let bit_index = (width as u32) * index;
    let index_in_u64 = (bit_index & 0x3f) as u8;

    index_in_u64
}

fn extra_u64s_taken_up(width: u8, current_index_in_u64: u8, num_elements: usize) -> usize {
    let remainder = (64 - current_index_in_u64) / width;
    if remainder as usize >= num_elements {
        return 0;
    }

    let u64s_taken_up = 1+(((num_elements-remainder as usize) * width as usize) >> 6);

    u64s_taken_up
}

#[cfg(test)]
mod tests {
    use super::*;
    fn verify_position_of_element(width: u8, index: u32, expected_index_in_u64: u8) {
        let result = find_position_of_element(width, index);
        assert_eq!(expected_index_in_u64, result);
    }

    #[test]
    fn test_position_of_element() {
        verify_position_of_element(7, 0, 0);
        verify_position_of_element(7, 1, 7);
        verify_position_of_element(7, 2, 14);
        verify_position_of_element(7, 3, 21);
        verify_position_of_element(7, 4, 28);
        verify_position_of_element(7, 5, 35);
        verify_position_of_element(7, 6, 42);
        verify_position_of_element(7, 7, 49);
        verify_position_of_element(7, 8, 56);
        verify_position_of_element(7, 9, 63);
        verify_position_of_element(7, 10, 6);
    }

    fn verify_u64s_taken_up(width: u8, current_index_in_u64: u8, num_elements: usize, expected_result: usize) {
        let result = extra_u64s_taken_up(width, current_index_in_u64, num_elements);
        assert_eq!((width, current_index_in_u64, num_elements, result),
                   (width, current_index_in_u64, num_elements, expected_result));
    }

    #[test]
    fn test_u64s_taken_up() {
        verify_u64s_taken_up(7, 0, 9, 0);
        verify_u64s_taken_up(7, 0, 10, 1);
        verify_u64s_taken_up(7, 1, 9, 0);
        verify_u64s_taken_up(7, 2, 9, 1);
        verify_u64s_taken_up(7, 63, 0, 0);
        verify_u64s_taken_up(7, 63, 1, 1);
        verify_u64s_taken_up(7, 1, 18, 1);
        verify_u64s_taken_up(7, 2, 18, 2);
    }

    #[test]
    fn log_array_error() {
        // Display
        assert_eq!(
            "expected input buffer size (7) >= 8",
            LogArrayError::InputBufferTooSmall(7).to_string()
        );
        assert_eq!(
            "expected width (69) <= 64",
            LogArrayError::WidthTooLarge(69).to_string()
        );
        assert_eq!(
            "expected input buffer size (9) to be 8 for 0 elements and width 17",
            LogArrayError::UnexpectedInputBufferSize(9, 8, 0, 17).to_string()
        );

        // From<LogArrayError> for io::Error
        assert_eq!(
            io::Error::new(
                io::ErrorKind::InvalidData,
                LogArrayError::InputBufferTooSmall(7)
            )
            .to_string(),
            io::Error::from(LogArrayError::InputBufferTooSmall(7)).to_string()
        );
    }

    #[test]
    fn validate_input_buf_size() {
        let val = |buf_size| LogArrayError::validate_input_buf_size(buf_size);
        let err = |buf_size| Err(LogArrayError::InputBufferTooSmall(buf_size));
        assert_eq!(err(7), val(7));
        assert_eq!(Ok(()), val(8));
        assert_eq!(Ok(()), val(9));
        assert_eq!(Ok(()), val(usize::max_value()));
    }

    #[test]
    fn validate_len_and_width() {
        let val =
            |buf_size, len, width| LogArrayError::validate_len_and_width(buf_size, len, width);

        let err = |width| Err(LogArrayError::WidthTooLarge(width));

        // width: 65
        assert_eq!(err(65), val(0, 0, 65));

        let err = |buf_size, expected, len, width| {
            Err(LogArrayError::UnexpectedInputBufferSize(
                buf_size, expected, len, width,
            ))
        };

        // width: 0
        assert_eq!(err(0, 8, 0, 0), val(0, 0, 0));

        // width: 1
        assert_eq!(Ok(()), val(8, 0, 1));
        assert_eq!(err(9, 8, 0, 1), val(9, 0, 1));
        assert_eq!(Ok(()), val(16, 1, 1));

        // width: 64
        assert_eq!(Ok(()), val(16, 1, 64));
        assert_eq!(err(16, 24, 2, 64), val(16, 2, 64));
        assert_eq!(err(24, 16, 1, 64), val(24, 1, 64));

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Ok(()),
            val(
                usize::try_from(u64::from(u32::max_value()) + 1 << 3).unwrap(),
                u32::max_value(),
                64
            )
        );

        // width: 5
        assert_eq!(err(16, 24, 13, 5), val(16, 13, 5));
        assert_eq!(Ok(()), val(24, 13, 5));
    }

    #[test]
    pub fn empty() {
        let logarray = LogArray::parse(Bytes::from([0u8; 8].as_ref())).unwrap();
        assert!(logarray.is_empty());
        assert!(MonotonicLogArray::from_logarray(logarray).is_empty());
    }

    #[test]
    fn log_array_file_builder_error() {
        let mut builder = LogArrayBuilder::new(3);
        let err = builder.push(8).unwrap_err();
        assert_eq!(LogArrayBuilderError::PushedNumberTooLargeForWidth{number:8, width:3}, err);
    }

    #[test]
    fn generate_then_parse_works() {
        let mut builder = LogArrayBuilder::new(5);
        builder.push_slice([1, 3, 2, 5, 12, 31, 18].as_slice()).unwrap();
        let bytes = builder.finalize();
        let logarray = LogArray::parse(bytes).unwrap();

        assert_eq!(1, logarray.entry(0));
        assert_eq!(3, logarray.entry(1));
        assert_eq!(2, logarray.entry(2));
        assert_eq!(5, logarray.entry(3));
        assert_eq!(12, logarray.entry(4));
        assert_eq!(31, logarray.entry(5));
        assert_eq!(18, logarray.entry(6));
    }

    const TEST0_DATA: [u8; 8] = [
        0b00000000,
        0b00000000,
        0b1_0000000,
        0b00000000,
        0b10_000000,
        0b00000000,
        0b011_00000,
        0b00000000,
    ];
    const TEST0_CONTROL: [u8; 8] = [0, 0, 0, 3, 17, 0, 0, 0];

    fn test0_logarray() -> LogArray {
        let mut content = Vec::new();
        content.extend_from_slice(&TEST0_DATA);
        content.extend_from_slice(&TEST0_CONTROL);
        LogArray::parse(Bytes::from(content)).unwrap()
    }

    #[test]
    #[should_panic(expected = "expected index (3) < length (3)")]
    fn entry_panic() {
        let _ = test0_logarray().entry(3);
    }

    #[test]
    #[should_panic(expected = "expected slice offset (2) + length (2) <= source length (3)")]
    fn slice_panic1() {
        let _ = test0_logarray().slice(2, 2);
    }

    #[test]
    #[should_panic(expected = "expected 32-bit slice offset (4294967296)")]
    #[cfg(target_pointer_width = "64")]
    fn slice_panic2() {
        let _ = test0_logarray().slice(usize::try_from(u32::max_value()).unwrap() + 1, 2);
    }

    #[test]
    #[should_panic(expected = "expected 32-bit slice length (4294967296)")]
    #[cfg(target_pointer_width = "64")]
    fn slice_panic3() {
        let _ = test0_logarray().slice(0, usize::try_from(u32::max_value()).unwrap() + 1);
    }

    #[test]
    #[should_panic(expected = "overflow from slice offset (4294967295) + length (1)")]
    fn slice_panic4() {
        let _ = test0_logarray().slice(usize::try_from(u32::max_value()).unwrap(), 1);
    }

    #[test]
    #[should_panic(expected = "expected index (2) < length (2)")]
    fn slice_entry_panic() {
        let _ = test0_logarray().slice(1, 2).entry(2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not monotonic: expected predecessor (2) <= successor (1)")]
    fn monotonic_panic() {
        let content = [0u8, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 32, 0, 0, 0].as_ref();
        MonotonicLogArray::from_logarray(LogArray::parse(Bytes::from(content)).unwrap());
    }

    #[test]
    fn test_slice() {
        let mut builder = LogArrayBuilder::new(5);
        builder.push_slice([2,0,3,8,5,9,12].as_slice()).unwrap();
        let bytes = builder.finalize();
        let logarray = LogArray::parse(bytes).unwrap();

        let slice = logarray.slice(2,4);
        assert_eq!(4, slice.len());
        assert_eq!(vec![3,8,5,9], slice.iter().collect::<Vec<_>>());
        assert_eq!(3, slice.entry(0));
        assert_eq!(8, slice.entry(1));
        assert_eq!(5, slice.entry(2));
        assert_eq!(9, slice.entry(3));

        let slice2 = slice.slice(1,2);
        assert_eq!(2, slice2.len());
        assert_eq!(vec![8,5], slice2.iter().collect::<Vec<_>>());
        assert_eq!(8, slice2.entry(0));
        assert_eq!(5, slice2.entry(1));
    }

    #[test]
    #[should_panic(expected = "expected slice offset (8) + length (10) <= source length (7)")]
    fn slice_start_out_of_bounds_should_panic() {
        let mut builder = LogArrayBuilder::new(5);
        builder.push_slice([2,0,3,8,5,9,12].as_slice()).unwrap();
        let bytes = builder.finalize();
        let logarray = LogArray::parse(bytes).unwrap();

        let _slice = logarray.slice(8,10);
    }

    #[test]
    #[should_panic(expected = "expected slice offset (4) + length (10) <= source length (7)")]
    fn slice_end_out_of_bounds_should_panic() {
        let mut builder = LogArrayBuilder::new(5);
        builder.push_slice([2,0,3,8,5,9,12].as_slice()).unwrap();
        let bytes = builder.finalize();
        let logarray = LogArray::parse(bytes).unwrap();

        let _slice = logarray.slice(4,10);
    }

    #[test]
    fn empty_slice() {
        let mut builder = LogArrayBuilder::new(5);
        builder.push_slice([2,0,3,8,5,9,12].as_slice()).unwrap();
        let bytes = builder.finalize();
        let logarray = LogArray::parse(bytes).unwrap();

        let slice = logarray.slice(4,0);
        assert!(slice.is_empty());
        assert_eq!(0, slice.len());
    }
}
