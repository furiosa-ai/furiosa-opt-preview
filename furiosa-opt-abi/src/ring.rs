//! One ring of words with two index registers, as the driver lays every queue out.
//!
//! - One slot stays empty so full and empty differ.
//! - [`Producer`] and [`Consumer`] each remember their own index and read the other side's register
//!   only when they must: across PCIe that read is the expensive step.
//! - A [`Ring`] becomes exactly one side, so it is not `Copy`: two live sides would each remember
//!   an index the other moves. A side made anew re-reads its register.
//! - Every entry opens with a [`header`]; a [`Completion`] is a header and fixed payload words.

use core::cell::Cell;
use core::task::Poll;

use crate::mmio::Register;

#[derive(Debug)]
pub struct Ring {
    entries: Register<u64>,
    producer: Register<u32>,
    consumer: Register<u32>,
    capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    /// A queue index is outside its valid range.
    #[error("queue index is out of range")]
    Index,
    /// More words were asked for than the ring holds.
    #[error("a ring was asked for more words than it holds")]
    Words,
}

impl Ring {
    /// Panics below two entries: the empty slot would leave no room to hold one.
    ///
    /// # Safety
    ///
    /// `entries` must reach `capacity` words, and all three must stay mapped while the ring is used.
    pub(crate) const unsafe fn new(
        entries: Register<u64>,
        producer: Register<u32>,
        consumer: Register<u32>,
        capacity: usize,
    ) -> Self {
        assert!(capacity >= 2, "a ring keeps one slot empty, so it holds two at least");
        Self {
            entries,
            producer,
            consumer,
            capacity,
        }
    }

    /// The completion ring a page holds: its producer and consumer indices, then `capacity` entries.
    ///
    /// # Safety
    ///
    /// `page` must map the whole ring for as long as it is used.
    pub const unsafe fn completion(page: *mut u8, capacity: usize) -> Self {
        // SAFETY: the caller maps the page these registers lie in.
        unsafe {
            Self::new(
                Register::new(page.wrapping_add(8).cast()),
                Register::new(page.cast()),
                Register::new(page.wrapping_add(4).cast()),
                capacity,
            )
        }
    }

    /// Entries from `consumer` up to `producer`, both below the capacity.
    const fn distance(&self, producer: usize, consumer: usize) -> usize {
        (producer + self.capacity - consumer) % self.capacity
    }

    /// Entries a producer at `producer` may still write with a consumer at `consumer`.
    const fn space(&self, producer: usize, consumer: usize) -> usize {
        self.capacity - self.distance(producer, consumer) - 1
    }

    const fn wrap(&self, index: usize) -> usize {
        index % self.capacity
    }

    fn checked(&self, index: u32) -> Result<usize, Error> {
        let index = index as usize;
        (index < self.capacity).then_some(index).ok_or(Error::Index)
    }

    fn read(&self, index: usize) -> u64 {
        // SAFETY: `new` promised `capacity` entries and `index` is below it.
        unsafe { self.entries.offset(index) }.read()
    }

    fn write(&self, index: usize, word: u64) {
        // SAFETY: as in `read`.
        unsafe { self.entries.offset(index) }.write(word)
    }
}

/// The writing side of a ring. It owns the producer index and remembers the space it last saw
/// below the consumer, so a write that fits costs no register read.
#[derive(Debug)]
pub struct Producer {
    ring: Ring,
    index: Cell<usize>,
    space: Cell<usize>,
}

impl Producer {
    /// Takes the producer side over, reading both indices once.
    pub fn new(ring: Ring) -> Result<Self, Error> {
        let index = ring.checked(ring.producer.read())?;
        let consumer = ring.checked(ring.consumer.read())?;
        let space = ring.space(index, consumer);
        Ok(Self {
            ring,
            index: Cell::new(index),
            space: Cell::new(space),
        })
    }

    /// Words that fit right now, reading the consumer's register.
    pub fn space(&self) -> Result<usize, Error> {
        let consumer = self.ring.checked(self.ring.consumer.read())?;
        let space = self.ring.space(self.index.get(), consumer);
        self.space.set(space);
        Ok(space)
    }

    /// Whether `words` fit, reading the consumer's register only when the remembered space says
    /// they do not.
    fn fits(&self, words: usize) -> Result<bool, Error> {
        Ok(words <= self.space.get() || words <= self.space()?)
    }

    /// Writes every word and moves the index once; pending, with the ring unchanged, while they
    /// do not fit.
    pub fn write_all(&self, words: &[u64]) -> Poll<Result<(), Error>> {
        self.write(words)
            .map(|result| result.map(|index| self.publish(index, words.len())))
    }

    /// Writes words for later publication. Dropping the result leaves the producer index unchanged;
    /// its exclusive borrow prevents another write from reusing the unpublished entries.
    pub fn prepare(&mut self, words: &[u64]) -> Poll<Result<Publication<'_>, Error>> {
        self.write(words).map(|result| {
            result.map(|index| Publication {
                producer: self,
                index,
                words: words.len(),
            })
        })
    }

    fn write(&self, words: &[u64]) -> Poll<Result<usize, Error>> {
        match self.fits(words.len()) {
            Ok(true) => {}
            Ok(false) => return Poll::Pending,
            Err(error) => return Poll::Ready(Err(error)),
        }
        let mut index = self.index.get();
        for &word in words {
            self.ring.write(index, word);
            index = self.ring.wrap(index + 1);
        }
        if !words.is_empty() {
            crate::mmio::release();
        }
        Poll::Ready(Ok(index))
    }

    fn publish(&self, index: usize, words: usize) {
        if words != 0 {
            self.ring.producer.write(index as u32);
            self.index.set(index);
            self.space.set(self.space.get() - words);
        }
    }
}

/// Unpublished words in one ring; consuming this value commits them exactly once.
#[must_use = "words become visible only when published"]
pub struct Publication<'a> {
    producer: &'a mut Producer,
    index: usize,
    words: usize,
}

impl Publication<'_> {
    /// Makes the prepared words visible. Entry writes were ordered before this store by `prepare`.
    pub fn publish(self) {
        self.producer.publish(self.index, self.words);
    }
}

/// The reading side of a ring. It owns the consumer index; the producer's register is read on
/// every look, since new words arrive through it.
#[derive(Debug)]
pub struct Consumer {
    ring: Ring,
    index: Cell<usize>,
}

impl Consumer {
    /// Takes the consumer side over, reading its index once.
    pub fn new(ring: Ring) -> Result<Self, Error> {
        let index = ring.checked(ring.consumer.read())?;
        Ok(Self {
            ring,
            index: Cell::new(index),
        })
    }

    /// Words published and not yet consumed.
    pub fn available(&self) -> Result<usize, Error> {
        let producer = self.ring.checked(self.ring.producer.read())?;
        crate::mmio::acquire();
        Ok(self.ring.distance(producer, self.index.get()))
    }

    /// Copies published words from `offset` past the index, without moving it.
    pub fn copy(&self, offset: usize, output: &mut [u64]) -> Result<(), Error> {
        let available = self.available()?;
        if offset > available || output.len() > available - offset {
            return Err(Error::Words);
        }
        for (step, word) in output.iter_mut().enumerate() {
            *word = self.ring.read(self.ring.wrap(self.index.get() + offset + step));
        }
        Ok(())
    }

    /// Moves the index past `words` copied words.
    pub fn consume(&self, words: usize) -> Result<(), Error> {
        if words > self.available()? {
            return Err(Error::Words);
        }
        if words != 0 {
            let index = self.ring.wrap(self.index.get() + words);
            crate::mmio::fence();
            self.ring.consumer.write(index as u32);
            self.index.set(index);
        }
        Ok(())
    }
}

const COMPLETION_WORDS: usize = 3;

/// One completion; the kernel driver fixes every completion's payload width.
pub struct Completion {
    pub id: u32,
    pub payload: [u64; COMPLETION_WORDS],
}

impl Completion {
    /// The completions carrying `frame` to request `id`, the last padded with zeros.
    pub fn encode(id: u32, frame: &[u64]) -> impl Iterator<Item = [u64; 1 + COMPLETION_WORDS]> + '_ {
        frame.chunks(COMPLETION_WORDS).map(move |fragment| {
            let mut words = [header::pack(id, COMPLETION_WORDS as u8), 0, 0, 0];
            words[1..=fragment.len()].copy_from_slice(fragment);
            words
        })
    }

    pub fn decode([header, payload @ ..]: [u64; 1 + COMPLETION_WORDS]) -> Option<Self> {
        let (id, words) = header::unpack(header)?;
        (usize::from(words) == COMPLETION_WORDS).then_some(Self { id, payload })
    }
}

/// The kernel driver's queue entry header: `{id, reserved, payload words, opcode}` little-endian.
pub mod header {
    const OPCODE: u64 = 1;
    const WORDS_SHIFT: u32 = 48;
    const OPCODE_SHIFT: u32 = 56;

    pub const fn pack(id: u32, words: u8) -> u64 {
        id as u64 | (words as u64) << WORDS_SHIFT | OPCODE << OPCODE_SHIFT
    }

    /// `(id, payload words)` of a header carrying the driver's opcode.
    pub const fn unpack(word: u64) -> Option<(u32, u8)> {
        if word >> OPCODE_SHIFT != OPCODE {
            return None;
        }
        Some((word as u32, (word >> WORDS_SHIFT) as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(entries: &mut [u64], producer: &mut u32, consumer: &mut u32) -> Ring {
        // SAFETY: the test owns this memory for the ring's lifetime.
        unsafe {
            Ring::new(
                Register::new(entries.as_mut_ptr()),
                Register::new(producer),
                Register::new(consumer),
                entries.len(),
            )
        }
    }

    /// Index registers move past exactly the words written or read, and one slot stays empty.
    #[test]
    fn fill_drain_across_wrap() {
        // Each side is a party of its own, so each gets its own ring over the same registers.
        let (mut entries, mut producer, mut consumer) = ([0; 4], 2, 2);
        let writer = Producer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();
        let reader = Consumer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();

        assert_eq!(writer.write_all(&[7, 8, 9]), Poll::Ready(Ok(())));
        assert_eq!(writer.write_all(&[10]), Poll::Pending);
        assert_eq!(reader.available(), Ok(3));
        let mut words = [0; 2];
        reader.copy(1, &mut words).unwrap();
        assert_eq!(words, [8, 9]);
        assert_eq!(reader.copy(2, &mut words), Err(Error::Words));
        reader.consume(3).unwrap();
        assert_eq!(reader.available(), Ok(0));
        assert_eq!(writer.write_all(&[10]), Poll::Ready(Ok(())));
        assert_eq!(consumer, 1);
        assert_eq!(producer, 2);
        assert_eq!(writer.write_all(&[11, 12]), Poll::Ready(Ok(())));
        // An index register past the capacity is refused, wherever either side reads it.
        consumer = 4;
        assert_eq!(writer.write_all(&[13]), Poll::Ready(Err(Error::Index)));
        assert_eq!(consumer, 4);
        producer = 4;
        assert_eq!(reader.available(), Err(Error::Index));
        assert_eq!(
            Producer::new(ring(&mut entries, &mut producer, &mut consumer)).err(),
            Some(Error::Index)
        );
    }

    /// A reader sees prepared words only once published; a dropped publication changes nothing.
    #[test]
    fn publication_is_atomic() {
        let (mut entries, mut producer, mut consumer) = ([0; 4], 2, 2);
        let mut writer = Producer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();
        let reader = Consumer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();
        {
            let Poll::Ready(Ok(_publication)) = writer.prepare(&[1, 2, 3]) else {
                panic!("empty ring has capacity");
            };
            assert_eq!(reader.available(), Ok(0));
        }
        let Poll::Ready(Ok(publication)) = writer.prepare(&[7, 8, 9]) else {
            panic!("dropped publication leaves capacity available");
        };
        assert_eq!(reader.available(), Ok(0));
        publication.publish();
        assert!(writer.prepare(&[10]).is_pending());
        let mut words = [0; 3];
        reader.copy(0, &mut words).unwrap();
        assert_eq!(words, [7, 8, 9]);
        reader.consume(3).unwrap();
        assert_eq!(writer.write_all(&[10]), Poll::Ready(Ok(())));
    }
}
