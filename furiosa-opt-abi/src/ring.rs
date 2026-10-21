//! One ring of words with two index registers, as the driver lays every queue out.
//!
//! - A ring owns the [`Memory`] it lies in, so it cannot outlive it, and checks once that its
//!   entries and index registers lie inside it.
//! - One slot stays empty so full and empty differ.
//! - [`Producer`] and [`Consumer`] each remember their own index and read the other side's register
//!   only when they must: across PCIe that read is the expensive step.
//! - A [`Ring`] becomes exactly one side, so it is not `Copy`: two live sides would each remember
//!   an index the other moves. A side made anew re-reads its register.
//! - Every entry opens with a [`header`]; a [`Completion`] is a header and fixed payload words.

use core::cell::Cell;
use core::mem::offset_of;
use core::task::Poll;

use crate::mmio::{Memory, Register};

/// A ring over the memory it lies in: its entries and two index registers, each at an offset the
/// memory maps.
#[derive(Debug)]
pub struct Ring<M> {
    memory: M,
    entries: usize,
    producer: usize,
    consumer: usize,
    capacity: usize,
}

impl<M: Memory> Ring<M> {
    /// The ring whose `capacity` entries start at byte offset `entries` of `memory`, with its
    /// index registers at `producer` and `consumer`. A ring keeps one slot empty, so it holds two
    /// entries at least.
    pub fn new(memory: M, entries: usize, producer: usize, consumer: usize, capacity: usize) -> Result<Self, Error> {
        let mapped = memory.mapped();
        let within = |offset: usize, len: usize| {
            offset >= mapped.start && offset.checked_add(len).is_some_and(|end| end <= mapped.end)
        };
        let fits = capacity >= 2
            && within(entries, capacity * size_of::<u64>())
            && within(producer, size_of::<u32>())
            && within(consumer, size_of::<u32>());
        fits.then_some(Self {
            memory,
            entries,
            producer,
            consumer,
            capacity,
        })
        .ok_or(Error::Unmapped)
    }

    /// The completion ring a page holds: its producer and consumer indices, then `capacity` entries.
    pub fn completion(page: M, capacity: usize) -> Result<Self, Error> {
        Self::new(
            page,
            offset_of!(CompletionPage, entries),
            offset_of!(CompletionPage, producer),
            offset_of!(CompletionPage, consumer),
            capacity,
        )
    }

    fn producer(&self) -> Register<u32> {
        self.register(self.producer)
    }

    fn consumer(&self) -> Register<u32> {
        self.register(self.consumer)
    }

    fn read(&self, index: usize) -> u64 {
        self.register(self.entries + index * size_of::<u64>()).read()
    }

    fn write(&self, index: usize, word: u64) {
        self.register(self.entries + index * size_of::<u64>()).write(word)
    }

    fn register<T: Copy>(&self, offset: usize) -> Register<T> {
        // SAFETY: `new` checked the offset lies in the memory, which this ring owns.
        unsafe { Register::new(self.memory.origin().wrapping_add(offset).cast()) }
    }

    fn checked(&self, index: u32) -> Result<usize, Error> {
        let index = index as usize;
        (index < self.capacity).then_some(index).ok_or(Error::Index)
    }

    /// Entries a producer at `producer` may still write with a consumer at `consumer`.
    const fn space(&self, producer: usize, consumer: usize) -> usize {
        self.capacity - self.distance(producer, consumer) - 1
    }

    /// Entries from `consumer` up to `producer`, both below the capacity.
    const fn distance(&self, producer: usize, consumer: usize) -> usize {
        (producer + self.capacity - consumer) % self.capacity
    }

    const fn wrap(&self, index: usize) -> usize {
        index % self.capacity
    }
}

/// The writing side of a ring. It owns the producer index and remembers the space it last saw
/// below the consumer, so a write that fits costs no register read.
#[derive(Debug)]
pub struct Producer<M> {
    ring: Ring<M>,
    index: Cell<usize>,
    space: Cell<usize>,
}

impl<M: Memory> Producer<M> {
    /// Takes the producer side over, reading both indices once.
    pub fn new(ring: Ring<M>) -> Result<Self, Error> {
        let index = ring.checked(ring.producer().read())?;
        let consumer = ring.checked(ring.consumer().read())?;
        let space = ring.space(index, consumer);
        Ok(Self {
            ring,
            index: Cell::new(index),
            space: Cell::new(space),
        })
    }

    /// Writes every word and moves the index once; pending, with the ring unchanged, while they
    /// do not fit.
    pub fn write_all(&self, words: &[u64]) -> Poll<Result<(), Error>> {
        self.write(words)
            .map(|result| result.map(|index| self.publish(index, words.len())))
    }

    /// Writes words for later publication. Dropping the result leaves the producer index unchanged;
    /// its exclusive borrow prevents another write from reusing the unpublished entries.
    pub fn prepare(&mut self, words: &[u64]) -> Poll<Result<Publication<'_, M>, Error>> {
        self.write(words).map(|result| {
            result.map(|index| Publication {
                producer: self,
                index,
                words: words.len(),
            })
        })
    }

    /// Words that fit right now, reading the consumer's register.
    pub fn space(&self) -> Result<usize, Error> {
        let consumer = self.ring.checked(self.ring.consumer().read())?;
        let space = self.ring.space(self.index.get(), consumer);
        self.space.set(space);
        Ok(space)
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
        Poll::Ready(Ok(index))
    }

    /// Whether `words` fit, reading the consumer's register only when the remembered space says
    /// they do not.
    fn fits(&self, words: usize) -> Result<bool, Error> {
        Ok(words <= self.space.get() || words <= self.space()?)
    }

    fn publish(&self, index: usize, words: usize) {
        if words != 0 {
            self.ring.producer().write_release(index as u32);
            self.index.set(index);
            self.space.set(self.space.get() - words);
        }
    }
}

/// Unpublished words in one ring; consuming this value commits them exactly once.
#[must_use = "words become visible only when published"]
pub struct Publication<'a, M: Memory> {
    producer: &'a mut Producer<M>,
    index: usize,
    words: usize,
}

impl<M: Memory> Publication<'_, M> {
    /// Makes the prepared words visible: the producer index is written after the entries.
    pub fn publish(self) {
        self.producer.publish(self.index, self.words);
    }
}

/// The reading side of a ring. It owns the consumer index; the producer's register is read on
/// every look, since new words arrive through it.
#[derive(Debug)]
pub struct Consumer<M> {
    ring: Ring<M>,
    index: Cell<usize>,
}

impl<M: Memory> Consumer<M> {
    /// Takes the consumer side over, reading its index once.
    pub fn new(ring: Ring<M>) -> Result<Self, Error> {
        let index = ring.checked(ring.consumer().read())?;
        Ok(Self {
            ring,
            index: Cell::new(index),
        })
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
            self.ring.consumer().write_release(index as u32);
            self.index.set(index);
        }
        Ok(())
    }

    /// Words published and not yet consumed.
    pub fn available(&self) -> Result<usize, Error> {
        let producer = self.ring.checked(self.ring.producer().read_acquire())?;
        Ok(self.ring.distance(producer, self.index.get()))
    }
}

/// One completion; the kernel driver fixes every completion's payload width.
pub struct Completion {
    pub id: u32,
    pub payload: [u64; COMPLETION_WORDS],
}

impl Completion {
    pub fn decode([header, payload @ ..]: [u64; 1 + COMPLETION_WORDS]) -> Option<Self> {
        let (id, words) = header::unpack(header)?;
        (usize::from(words) == COMPLETION_WORDS).then_some(Self { id, payload })
    }

    /// The completions carrying `frame` to request `id`, the last padded with zeros.
    pub fn encode(id: u32, frame: &[u64]) -> impl Iterator<Item = [u64; 1 + COMPLETION_WORDS]> + '_ {
        frame.chunks(COMPLETION_WORDS).map(move |fragment| {
            let mut words = [header::pack(id, COMPLETION_WORDS as u8), 0, 0, 0];
            words[1..=fragment.len()].copy_from_slice(fragment);
            words
        })
    }
}

/// The kernel driver's queue entry header, the first word of every entry.
pub mod header {
    use zerocopy::byteorder::little_endian::{U16, U32};
    use zerocopy::{FromBytes, IntoBytes};

    pub const fn pack(id: u32, words: u8) -> u64 {
        let header = Header {
            id: U32::new(id),
            reserved: U16::ZERO,
            words,
            opcode: OPCODE,
        };
        u64::from_le_bytes(zerocopy::transmute!(header))
    }

    /// `(id, payload words)` of a header carrying the driver's opcode.
    pub const fn unpack(word: u64) -> Option<(u32, u8)> {
        let header: Header = zerocopy::transmute!(word.to_le_bytes());
        if header.opcode != OPCODE {
            return None;
        }
        Some((header.id.get(), header.words))
    }

    /// The header as the driver lays it out, little-endian.
    #[derive(FromBytes, IntoBytes)]
    #[repr(C)]
    struct Header {
        id: U32,
        reserved: U16,
        words: u8,
        opcode: u8,
    }

    const OPCODE: u8 = 1;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    /// A queue index is outside its valid range.
    #[error("queue index is out of range")]
    Index,
    /// More words were asked for than the ring holds.
    #[error("a ring was asked for more words than it holds")]
    Words,
    /// A ring's entries or index registers lie outside the memory that holds it.
    #[error("a ring lies outside its memory")]
    Unmapped,
}

const COMPLETION_WORDS: usize = 3;

/// The page a completion ring lies in, as the kernel driver lays it out.
#[repr(C)]
pub(crate) struct CompletionPage {
    pub(crate) producer: u32,
    pub(crate) consumer: u32,
    pub(crate) entries: [u64; 0],
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A completion page the test owns: the two indices in its first word, four entries after.
    struct Page(core::cell::UnsafeCell<[u64; 5]>);

    impl Page {
        fn new(producer: u32, consumer: u32) -> Self {
            Self(core::cell::UnsafeCell::new([
                u64::from(producer) | u64::from(consumer) << 32,
                0,
                0,
                0,
                0,
            ]))
        }

        fn index(&self, which: usize) -> u32 {
            // SAFETY: the page outlives this read; the rings over it read it volatile too.
            unsafe { core::ptr::read_volatile(self.0.get().cast::<u32>().add(which)) }
        }

        fn set(&self, which: usize, index: u32) {
            // SAFETY: as in `index`.
            unsafe { core::ptr::write_volatile(self.0.get().cast::<u32>().add(which), index) }
        }
    }

    struct View<'a>(&'a Page);

    // SAFETY: the page lives for `'a` and is all mapped.
    unsafe impl Memory for View<'_> {
        fn origin(&self) -> *mut u8 {
            self.0.0.get().cast()
        }

        fn mapped(&self) -> core::ops::Range<usize> {
            0..size_of::<[u64; 5]>()
        }
    }

    const PRODUCER: usize = 0;
    const CONSUMER: usize = 1;

    // Each side is a party of its own, so each gets its own ring over the same page.
    fn ring(page: &Page) -> Ring<View<'_>> {
        Ring::completion(View(page), 4).unwrap()
    }

    /// Index registers move past exactly the words written or read, and one slot stays empty.
    #[test]
    fn fill_drain_across_wrap() {
        let page = Page::new(2, 2);
        let writer = Producer::new(ring(&page)).unwrap();
        let reader = Consumer::new(ring(&page)).unwrap();

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
        assert_eq!(page.index(CONSUMER), 1);
        assert_eq!(page.index(PRODUCER), 2);
        assert_eq!(writer.write_all(&[11, 12]), Poll::Ready(Ok(())));
        // An index register past the capacity is refused, wherever either side reads it.
        page.set(CONSUMER, 4);
        assert_eq!(writer.write_all(&[13]), Poll::Ready(Err(Error::Index)));
        assert_eq!(page.index(CONSUMER), 4);
        page.set(PRODUCER, 4);
        assert_eq!(reader.available(), Err(Error::Index));
        assert_eq!(Producer::new(ring(&page)).err(), Some(Error::Index));
    }

    /// A reader sees prepared words only once published; a dropped publication changes nothing.
    #[test]
    fn publication_is_atomic() {
        let page = Page::new(2, 2);
        let mut writer = Producer::new(ring(&page)).unwrap();
        let reader = Consumer::new(ring(&page)).unwrap();
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

    /// A ring whose entries or index registers reach past its memory is refused.
    #[test]
    fn refuses_rings_outside_their_memory() {
        let page = Page::new(0, 0);
        assert_eq!(Ring::completion(View(&page), 5).err(), Some(Error::Unmapped));
        assert_eq!(Ring::new(View(&page), 8, 40, 4, 4).err(), Some(Error::Unmapped));
        assert_eq!(Ring::completion(View(&page), 1).err(), Some(Error::Unmapped));
    }
}
