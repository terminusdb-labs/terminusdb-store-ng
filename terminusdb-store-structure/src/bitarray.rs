use byteorder::{BigEndian, ByteOrder};
use bytes::{BufMut, Bytes, BytesMut};
use std::error;
use std::fmt;
use std::io;

/// A thread-safe, reference-counted, compressed bit sequence.
///
/// A `BitArray` is a wrapper around a [`Bytes`] that provides a view of the underlying data as a
/// compressed sequence of bits.
///
/// [`Bytes`]: ../../../bytes/struct.Bytes.html
///
/// As with other types in [`structures`], a `BitArray` is created from an existing buffer, rather
/// than constructed from parts. The buffer may be read from a file or other source and may be very
/// large. A `BitArray` preserves the buffer to save memory but provides a simple abstraction of
/// being a vector of `bool`s.
///
/// [`structures`]: ../index.html
#[derive(Clone)]
pub struct BitArray {
    /// Number of usable bits in the array.
    len: u64,

    /// Shared reference to the buffer containing the sequence of bits.
    ///
    /// The buffer does not contain the control word.
    buf: Bytes,
}

/// An error that occurred during a bit array operation.
#[derive(Debug, PartialEq)]
pub enum BitArrayError {
    InputBufferTooSmall(usize),
    UnexpectedInputBufferSize(u64, u64, u64),
}

impl BitArrayError {
    /// Validate the input buffer size.
    ///
    /// It must have at least the control word.
    fn validate_input_buf_size(input_buf_size: usize) -> Result<(), Self> {
        if input_buf_size < 8 {
            return Err(BitArrayError::InputBufferTooSmall(input_buf_size));
        }
        Ok(())
    }

    /// Validate the length.
    ///
    /// The input buffer size should be the appropriate multiple of 8 to include the number of bits
    /// plus the control word.
    fn validate_len(input_buf_size: usize, len: u64) -> Result<(), Self> {
        // Calculate the expected input buffer size. This includes the control word.
        let expected_buf_size = {
            // The following steps are necessary to avoid overflow. If we add first and shift
            // second, the addition might result in a value greater than `u64::max_value()`.
            // Therefore, we right-shift first to produce a value that cannot overflow, check how
            // much we need to add, and add it.
            let after_shifting = len >> 6 << 3;
            if len & 63 == 0 {
                // The number of bits fit evenly into 64-bit words. Add only the control word.
                after_shifting + 8
            } else {
                // The number of bits do not fit evenly into 64-bit words. Add a word for the
                // leftovers plus the control word.
                after_shifting + 16
            }
        };
        let input_buf_size = u64::try_from(input_buf_size).unwrap();

        if input_buf_size != expected_buf_size {
            return Err(BitArrayError::UnexpectedInputBufferSize(
                input_buf_size,
                expected_buf_size,
                len,
            ));
        }

        Ok(())
    }
}

impl fmt::Display for BitArrayError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use BitArrayError::*;
        match self {
            InputBufferTooSmall(input_buf_size) => {
                write!(f, "expected input buffer size ({}) >= 8", input_buf_size)
            }
            UnexpectedInputBufferSize(input_buf_size, expected_buf_size, len) => write!(
                f,
                "expected input buffer size ({}) to be {} for {} bits",
                input_buf_size, expected_buf_size, len
            ),
        }
    }
}

impl error::Error for BitArrayError {}

impl From<BitArrayError> for io::Error {
    fn from(err: BitArrayError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, err)
    }
}

/// Read the length from the control word buffer. `buf` must start at the first word after the data
/// buffer. `input_buf_size` is used for validation.
fn read_and_validate_control_word(buf: &[u8], input_buf_size: usize) -> Result<u64, BitArrayError> {
    let len = read_control_word(buf);
    BitArrayError::validate_len(input_buf_size, len)?;
    Ok(len)
}

pub fn read_control_word(buf: &[u8]) -> u64 {
    BigEndian::read_u64(buf)
}

impl BitArray {
    /// Construct a `BitArray` by parsing a `Bytes` buffer.
    pub fn parse(buf: Bytes) -> Result<BitArray, BitArrayError> {
        let input_buf_size = buf.len();
        BitArrayError::validate_input_buf_size(input_buf_size)?;

        let len = read_and_validate_control_word(&buf[input_buf_size - 8..], input_buf_size)?;

        Ok(BitArray { buf, len })
    }

    /// Returns the number of usable bits in the bit array.
    pub fn len(&self) -> usize {
        usize::try_from(self.len).unwrap_or_else(|_| {
            panic!(
                "expected length ({}) to fit in {} bytes",
                self.len,
                std::mem::size_of::<usize>()
            )
        })
    }

    /// Returns `true` if there are no usable bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Reads the data buffer and returns the logical value of the bit at the bit `index`.
    ///
    /// Panics if `index` is >= the length of the bit array.
    pub fn get(&self, index: usize) -> bool {
        let len = self.len();
        assert!(index < len, "expected index ({}) < length ({})", index, len);

        let byte = self.buf[index / 8];
        let mask = 0b1000_0000 >> index % 8;

        byte & mask != 0
    }

    pub fn iter(&self) -> BitArrayIterator {
        BitArrayIterator::new(self.clone())
    }

    pub fn rank0(&self, mut index: usize) -> usize {
        if index >= self.len() {
            panic!(
                "index {} out of range for bitarray with length {}",
                index,
                self.len()
            );
        }

        let mut it = self.iter();
        let mut count = 0;
        loop {
            let elt = it.next().unwrap();

            if !elt {
                count += 1;
            }

            if index == 0 {
                return count;
            }
            index -= 1;
        }
    }

    pub fn rank1(&self, mut index: usize) -> usize {
        if index >= self.len() {
            panic!(
                "index {} out of range for bitarray with length {}",
                index,
                self.len()
            );
        }
        let mut it = self.iter();
        let mut count = 0;
        loop {
            let elt = it.next().unwrap();

            if elt {
                count += 1;
            }

            if index == 0 {
                return count;
            }
            index -= 1;
        }
    }

    pub fn select0(&self, mut rank: usize) -> Option<usize> {
        if rank == 0 {
            if self.get(0) {
                return Some(0);
            } else {
                return None;
            }
        }

        for (index, elt) in self.iter().enumerate() {
            if !elt {
                rank -= 1;
                if rank == 0 {
                    return Some(index);
                }
            }
        }

        None
    }

    pub fn select1(&self, mut rank: usize) -> Option<usize> {
        if rank == 0 {
            if !self.get(0) {
                return Some(0);
            } else {
                return None;
            }
        }

        for (index, elt) in self.iter().enumerate() {
            if elt {
                rank -= 1;
                if rank == 0 {
                    return Some(index);
                }
            }
        }

        None
    }
}

pub struct BitArrayIterator {
    bitarray: BitArray,
    pos: usize,
    current: u64,
    current_mask: u64,
}

impl BitArrayIterator {
    fn new(bitarray: BitArray) -> Self {
        Self {
            bitarray,
            pos: 0,
            current: 0,
            current_mask: 0x8000000000000000,
        }
    }
}

impl Iterator for BitArrayIterator {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.pos >= self.bitarray.len() {
            return None;
        }

        if self.pos % 64 == 0 {
            let byte_pos = self.pos / 8;
            self.current = BigEndian::read_u64(&self.bitarray.buf[byte_pos..byte_pos + 8]);
        }

        let result = (self.current & self.current_mask) != 0;

        self.current_mask = self.current_mask.rotate_right(1);
        self.pos += 1;

        Some(result)
    }
}

pub struct BitArrayBuilder {
    bytes: BytesMut,
    count: usize,
}

impl BitArrayBuilder {
    pub fn new() -> Self {
        Self {
            bytes: BytesMut::new(),
            count: 0,
        }
    }

    fn required_capacity_bytes(capacity: usize) -> usize {
        let num_words = (capacity + 63) / 64;
        let num_bytes = num_words * 8;

        // adding 8 for the control word
        num_bytes + 8
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: BytesMut::with_capacity(Self::required_capacity_bytes(capacity)),
            count: 0,
        }
    }

    pub fn push(&mut self, bit: bool) {
        let len = self.bytes.len();
        let current_byte_ix = self.count / 8;
        if current_byte_ix >= len {
            self.bytes.put_u64(0);
        }

        let current_byte = self
            .bytes
            .get_mut(current_byte_ix)
            .expect("can't borrow a byte that was just allocated");

        if bit {
            let index_in_byte = self.count % 8;
            *current_byte |= 0x80 >> index_in_byte;
        }

        self.count += 1;
    }

    pub fn push_slice(&mut self, bits: &[bool]) {
        self.reserve(bits.len());

        for &bit in bits {
            self.push(bit);
        }
    }

    pub fn push_iter<I: Iterator<Item = bool>>(&mut self, iter: I) {
        for bit in iter {
            self.push(bit);
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        let capacity = self.count + additional;
        let required_bytes = Self::required_capacity_bytes(capacity);

        self.bytes.reserve(required_bytes);
    }

    pub fn finalize(mut self) -> Bytes {
        self.bytes.put_u64(self.count as u64);

        self.bytes.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_array_error() {
        // Display
        assert_eq!(
            "expected input buffer size (7) >= 8",
            BitArrayError::InputBufferTooSmall(7).to_string()
        );
        assert_eq!(
            "expected input buffer size (9) to be 8 for 0 bits",
            BitArrayError::UnexpectedInputBufferSize(9, 8, 0).to_string()
        );

        // From<BitArrayError> for io::Error
        assert_eq!(
            io::Error::new(
                io::ErrorKind::InvalidData,
                BitArrayError::InputBufferTooSmall(7)
            )
            .to_string(),
            io::Error::from(BitArrayError::InputBufferTooSmall(7)).to_string()
        );
    }

    #[test]
    fn validate_input_buf_size() {
        let val = |buf_size| BitArrayError::validate_input_buf_size(buf_size);
        let err = |buf_size| Err(BitArrayError::InputBufferTooSmall(buf_size));
        assert_eq!(err(7), val(7));
        assert_eq!(Ok(()), val(8));
        assert_eq!(Ok(()), val(9));
        assert_eq!(Ok(()), val(usize::max_value()));
    }

    #[test]
    fn validate_len() {
        let val = |buf_size, len| BitArrayError::validate_len(buf_size, len);
        let err = |buf_size, expected, len| {
            Err(BitArrayError::UnexpectedInputBufferSize(
                buf_size, expected, len,
            ))
        };

        assert_eq!(err(0, 8, 0), val(0, 0));
        assert_eq!(Ok(()), val(16, 1));
        assert_eq!(Ok(()), val(16, 2));

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            Ok(()),
            val(
                usize::try_from(u128::from(u64::max_value()) + 65 >> 6 << 3).unwrap(),
                u64::max_value()
            )
        );
    }

    #[test]
    fn empty() {
        assert!(BitArray::parse(Bytes::from([0u8; 8].as_ref()))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn construct_and_parse_small_bitarray() {
        let contents = vec![true, true, false, false, true];

        let mut builder = BitArrayBuilder::new();
        builder.push_slice(&contents);
        let bytes = builder.finalize();

        let bitarray = BitArray::parse(bytes).unwrap();

        assert_eq!(true, bitarray.get(0));
        assert_eq!(true, bitarray.get(1));
        assert_eq!(false, bitarray.get(2));
        assert_eq!(false, bitarray.get(3));
        assert_eq!(true, bitarray.get(4));
    }

    #[test]
    fn construct_and_parse_large_bitarray() {
        let contents = (0..).map(|n| n % 3 == 0).take(123456);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bytes = builder.finalize();

        let bitarray = BitArray::parse(bytes).unwrap();

        for i in 0..bitarray.len() {
            assert_eq!(i % 3 == 0, bitarray.get(i));
        }
    }

    #[test]
    fn construct_and_iterate_through_large_bitarray() {
        let contents = (0..).map(|n| n % 3 == 0).take(123456);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bytes = builder.finalize();

        let bitarray = BitArray::parse(bytes).unwrap();

        for (index, element) in bitarray.iter().enumerate() {
            assert_eq!(index % 3 == 0, element);
        }
    }

    fn verify_capacity_calculation(count: usize, expected_capacity: usize) {
        let actual_capacity = BitArrayBuilder::required_capacity_bytes(count);
        assert_eq!((count, expected_capacity), (count, actual_capacity));
    }

    #[test]
    fn test_capacity_calculation() {
        verify_capacity_calculation(0, 8);
        verify_capacity_calculation(1, 16);
        verify_capacity_calculation(64, 16);
        verify_capacity_calculation(65, 24);
    }

    #[test]
    fn rank0() {
        let contents = vec![true, true, false, false, true, true];

        let mut builder = BitArrayBuilder::new();
        builder.push_slice(&contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        assert_eq!(0, bitarray.rank0(0));
        assert_eq!(0, bitarray.rank0(1));
        assert_eq!(1, bitarray.rank0(2));
        assert_eq!(2, bitarray.rank0(3));
        assert_eq!(2, bitarray.rank0(4));
        assert_eq!(2, bitarray.rank0(5));
    }

    #[test]
    fn rank0_large() {
        let contents = (0..).map(|n| n % 3 == 0).take(1234);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bytes = builder.finalize();

        let bitarray = BitArray::parse(bytes).unwrap();

        let mut expected_rank = 0;
        for i in 0..bitarray.len() {
            if i % 3 != 0 {
                expected_rank += 1;
            }
            let rank = bitarray.rank0(i);
            assert_eq!(expected_rank, rank);
        }
    }

    #[test]
    fn rank1() {
        let contents = vec![true, true, false, false, true, true];

        let mut builder = BitArrayBuilder::new();
        builder.push_slice(&contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        assert_eq!(1, bitarray.rank1(0));
        assert_eq!(2, bitarray.rank1(1));
        assert_eq!(2, bitarray.rank1(2));
        assert_eq!(2, bitarray.rank1(3));
        assert_eq!(3, bitarray.rank1(4));
        assert_eq!(4, bitarray.rank1(5));
    }

    #[test]
    fn rank1_large() {
        let contents = (0..).map(|n| n % 3 == 0).take(1234);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bytes = builder.finalize();

        let bitarray = BitArray::parse(bytes).unwrap();

        let mut expected_rank = 0;
        for i in 0..bitarray.len() {
            if i % 3 == 0 {
                expected_rank += 1;
            }
            let rank = bitarray.rank1(i);
            assert_eq!(expected_rank, rank);
        }
    }

    #[test]
    fn select0() {
        let contents = vec![true, true, false, false, true, true];

        let mut builder = BitArrayBuilder::new();
        builder.push_slice(&contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        assert_eq!(Some(0), bitarray.select0(0));
        assert_eq!(Some(2), bitarray.select0(1));
        assert_eq!(Some(3), bitarray.select0(2));
        assert_eq!(None, bitarray.select0(3));
    }

    #[test]
    fn select0_large() {
        let contents = (0..).map(|n| n % 3 == 0).take(1234);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        let mut skip = 3;
        let mut expected_index = 0;
        for i in 0..823 {
            let s = bitarray.select0(i);

            assert_eq!(Some(expected_index), s);
            expected_index += 1;
            skip -= 1;
            if skip == 0 {
                skip = 2;
                expected_index += 1;
            }
        }
    }

    #[test]
    fn select1() {
        let contents = vec![true, true, false, false, true, true];

        let mut builder = BitArrayBuilder::new();
        builder.push_slice(&contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        assert_eq!(None, bitarray.select1(0));
        assert_eq!(Some(0), bitarray.select1(1));
        assert_eq!(Some(1), bitarray.select1(2));
        assert_eq!(Some(4), bitarray.select1(3));
        assert_eq!(Some(5), bitarray.select1(4));
        assert_eq!(None, bitarray.select1(5));
    }

    #[test]
    fn select1_large() {
        let contents = (0..).map(|n| n % 3 == 0).take(1234);

        let mut builder = BitArrayBuilder::new();
        builder.push_iter(contents);
        let bitarray = BitArray::parse(builder.finalize()).unwrap();

        assert_eq!(None, bitarray.select1(0));
        let mut expected_index = 0;
        for i in 1..413 {
            let s = bitarray.select1(i);
            assert_eq!(Some(expected_index), s);

            expected_index += 3;
        }
    }
}
