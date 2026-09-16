use std::ops::Range;

const ALIGN: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    at: usize,
    len: usize,
}

impl Span {
    const fn end(self) -> usize {
        self.at + self.len
    }
}

/// First-fit allocator over one span of a device memory window, tracking free spans in address
/// order.
pub(crate) struct Allocator {
    capacity: usize,
    free: Vec<Span>,
}

impl Allocator {
    /// Hands out `free`, narrowed to whole aligned units.
    pub(crate) fn new(free: Range<usize>) -> Self {
        let at = free.start.next_multiple_of(ALIGN);
        let end = free.end & !(ALIGN - 1);
        let capacity = end.saturating_sub(at);
        let free = (capacity > 0)
            .then_some(Span { at, len: capacity })
            .into_iter()
            .collect();
        Self { capacity, free }
    }

    pub(crate) fn memory(&self) -> crate::Memory {
        crate::Memory {
            capacity: self.capacity,
            available: self.free.iter().map(|span| span.len).sum(),
            largest: self.free.iter().map(|span| span.len).max().unwrap_or(0),
        }
    }

    pub(crate) fn alloc(&mut self, len: usize) -> Option<usize> {
        let len = Self::rounded(len)?;
        let index = self.free.iter().position(|span| span.len >= len)?;
        let span = self.free[index];
        if span.len == len {
            self.free.remove(index);
        } else {
            self.free[index] = Span {
                at: span.at + len,
                len: span.len - len,
            };
        }
        Some(span.at)
    }

    pub(crate) fn release(&mut self, at: usize, len: usize) {
        let span = Span {
            at,
            len: Self::rounded(len).expect("valid prior allocation length"),
        };
        let index = self.free.partition_point(|other| other.at < at);

        debug_assert!(
            index
                .checked_sub(1)
                .and_then(|previous| self.free.get(previous))
                .is_none_or(|previous| previous.end() <= at)
                && self.free.get(index).is_none_or(|next| span.end() <= next.at),
            "{at:#x}..{:#x} was already free",
            span.end(),
        );
        self.free.insert(index, span);
        self.merge(index);
    }

    const fn rounded(len: usize) -> Option<usize> {
        let len = if len == 0 { 1 } else { len };
        match len.checked_add(ALIGN - 1) {
            Some(len) => Some(len & !(ALIGN - 1)),
            None => None,
        }
    }

    fn merge(&mut self, index: usize) {
        if let Some(next) = self.free.get(index + 1).copied()
            && self.free[index].end() == next.at
        {
            self.free[index].len += next.len;
            self.free.remove(index + 1);
        }
        if let Some(previous) = index.checked_sub(1)
            && self.free[previous].end() == self.free[index].at
        {
            self.free[previous].len += self.free[index].len;
            self.free.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_aligned_capacity() {
        let allocator = Allocator::new(1000..4097);
        let memory = allocator.memory();

        assert_eq!(memory.capacity, 3072);
        assert_eq!(memory.available, 3072);
        assert_eq!(memory.largest, 3072);
    }

    #[test]
    fn reports_fragmentation() {
        let mut allocator = Allocator::new(1024..5120);
        let first = allocator.alloc(1024).unwrap();
        let second = allocator.alloc(1024).unwrap();
        allocator.release(first, 1024);

        let memory = allocator.memory();

        assert_eq!(memory.capacity, 4096);
        assert_eq!(memory.available, 3072);
        assert_eq!(memory.largest, 2048);
        assert!(allocator.alloc(memory.largest + 1).is_none());
        assert_eq!(allocator.alloc(memory.largest), Some(second + 1024));
    }

    #[test]
    fn reports_coalesced_space() {
        let mut allocator = Allocator::new(0..4096);
        let first = allocator.alloc(1024).unwrap();
        let second = allocator.alloc(1024).unwrap();
        allocator.release(first, 1024);

        allocator.release(second, 1024);

        assert_eq!(allocator.memory().available, 4096);
        assert_eq!(allocator.memory().largest, 4096);
    }

    #[test]
    fn reports_exhaustion() {
        let mut allocator = Allocator::new(0..256);
        allocator.alloc(256).unwrap();

        let memory = allocator.memory();

        assert_eq!(memory.capacity, 256);
        assert_eq!(memory.available, 0);
        assert_eq!(memory.largest, 0);
    }

    #[test]
    fn reuses_released_span() {
        let mut memory = Allocator::new(0..4096);
        let at = memory.alloc(256).expect("an allocation");

        memory.release(at, 256);

        assert_eq!(memory.alloc(256), Some(at));
    }

    #[test]
    fn allocates_from_the_start_of_its_span() {
        let mut memory = Allocator::new(1000..2048);

        assert_eq!(memory.alloc(256), Some(1024));
        assert_eq!(memory.alloc(1024), None);
        assert_eq!(memory.alloc(768), Some(1280));
    }
}
