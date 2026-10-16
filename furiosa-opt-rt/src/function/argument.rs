//! Borrowed call arguments and retained operand storage.

use std::sync::{Arc, Weak};

use crate::buffer::{Holds, Owner};
use crate::driver::NO_FIRMWARE_DMA;
use crate::pinned::{HostLease, Mapping};
use crate::{Access, Buffer, BufferSlice, BufferSliceMut, Device, Error, HostBufferSliceMut, Result};

/// A launch's arguments; device memory is checked against other work by the runtime's access
/// tracking when the launch is admitted, not by the view's borrow.
pub enum Argument<'a> {
    /// A word device code receives as it is.
    Value(u64),
    /// Memory the launch only reads.
    Read(BufferSlice<'a>),
    /// Memory the launch may write.
    Write(BufferSlice<'a>),
    /// Mapped pinned host memory the launch's device-issued DMA may read or write; the launch
    /// holds it as a whole, excluding everything except distinct ordered launches on the same mapping.
    Host(HostBufferSliceMut<'a>),
}

impl<'a> From<BufferSlice<'a>> for Argument<'a> {
    fn from(view: BufferSlice<'a>) -> Self {
        Self::Read(view)
    }
}
impl<'a> From<BufferSliceMut<'a>> for Argument<'a> {
    fn from(view: BufferSliceMut<'a>) -> Self {
        Self::Write(view.0)
    }
}

impl Argument<'_> {
    /// The device storage this argument names, and whether the launch writes it.
    fn device(&self) -> Option<(BufferSlice<'_>, Access)> {
        match self {
            Self::Read(view) => Some((*view, Access::Read)),
            Self::Write(view) => Some((*view, Access::Write)),
            _ => None,
        }
    }
}

/// Operand storage allocated once at load and empty when the bind callback fills it.
pub struct Arguments {
    pub(super) values: Vec<Operand>,
    access: Holds,
    pub(super) owner: Owner,
    limit: usize,
    /// Retires the launches and transfers a host operand's lease conflicts with.
    device: Weak<Device>,
}

impl Arguments {
    pub(crate) fn new(capacity: usize, device: Weak<Device>) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            access: Holds::new(capacity),
            owner: Owner::Unordered,
            limit: capacity,
            device,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn clear(&mut self) {
        self.access.clear();
        self.values.clear();
    }

    pub(super) fn reserve(&mut self, regions: usize) {
        if self.limit + regions > self.access.capacity() {
            self.access = Holds::new(self.limit + regions);
        }
    }

    pub(super) fn acquire(&mut self, held: &[(BufferSlice<'_>, Access)]) -> Result<()> {
        self.access.owner = self.owner;
        self.access.acquire(
            self.values
                .iter()
                .filter_map(|operand| match operand {
                    Operand::Device { access, .. } => Some((operand.buffer().expect("device operand"), *access)),
                    _ => None,
                })
                .chain(held.iter().copied()),
        )
    }

    pub(super) fn poison(&self) {
        self.access.poison();
        for operand in &self.values {
            if let Operand::Host { lease, .. } = operand {
                lease.poison();
            }
        }
    }

    /// Binds operand `index`; operands that are never bound are zero, and [`Error::Capacity`]
    /// past the function's argument count. Binding an index again must name the same operand,
    /// for memory its start, chips and access, and keeps the first length, so an image may list
    /// one slot twice. A host operand's memory is [`Error::Busy`] while another device access
    /// holds it.
    pub fn bind(&mut self, index: usize, argument: Argument<'_>) -> Result<()> {
        if index >= self.limit {
            return Err(Error::Capacity);
        }
        if index >= self.len() {
            self.values.resize_with(index + 1, || Operand::Unbound);
        }
        match &self.values[index] {
            Operand::Unbound => {
                let device = self.device.upgrade().expect("a function keeps its device");
                if let Argument::Host(view) = &argument
                    && let Some(backing) = view.backing
                    && self.values.iter().any(|operand| match operand {
                        Operand::Host { lease, .. } => lease.belongs_to(&backing),
                        Operand::Unbound | Operand::Value(_) | Operand::Device { .. } => false,
                    })
                {
                    return Err(Error::Busy);
                }
                self.values[index] = Operand::new(argument, &device, self.owner)?;
                Ok(())
            }
            bound if bound.names(&argument) => Ok(()),
            _ => Err(Error::Invalid(format!("operand {index} is bound to different storage"))),
        }
    }
}

pub(crate) enum Operand {
    /// An index no argument was bound to, encoded as zero.
    Unbound,
    Value(u64),
    Device {
        buffer: Buffer,
        chips: u8,
        access: Access,
    },
    Host {
        mapping: Arc<Mapping>,
        offset: u32,
        lease: HostLease,
    },
}

impl Operand {
    /// `argument` as an operand of a launch on `device`.
    pub(crate) fn new(argument: Argument<'_>, device: &Device, owner: Owner) -> Result<Self> {
        Ok(match argument {
            Argument::Value(value) => Self::Value(value),
            Argument::Host(view) => {
                if !device.driver.firmware_dma() {
                    return Err(Error::Device(NO_FIRMWARE_DMA.into()));
                }
                let backing = view
                    .backing
                    .ok_or_else(|| Error::Invalid("a host operand must be pinned".into()))?;
                let mapping = backing
                    .mapping
                    .ok_or_else(|| Error::Invalid("host memory has not been mapped to a device".into()))?;
                Self::Host {
                    mapping: Arc::clone(mapping),
                    offset: backing.offset as u32,
                    lease: device.admit(|| match owner {
                        Owner::Ordered(_) => backing.ordered(),
                        Owner::Unordered => backing.lease(Access::Write),
                    })?,
                }
            }
            _ => {
                let (view, access) = argument.device().expect("device argument");
                Self::Device {
                    buffer: view.retain(),
                    chips: view.chips,
                    access,
                }
            }
        })
    }

    /// Whether `argument` names this operand: the same value, or the same start, chips and access
    /// of the same memory.
    fn names(&self, argument: &Argument<'_>) -> bool {
        match (self, argument) {
            (Self::Value(value), Argument::Value(other)) => value == other,
            (Self::Device { access, .. }, _) => match (self.buffer(), argument.device()) {
                (Some(bound), Some((view, other))) => bound.ptr_eq(&view) && *access == other,
                _ => false,
            },
            (Self::Host { mapping, offset, .. }, Argument::Host(view)) => view.backing.is_some_and(|backing| {
                backing.mapping.is_some_and(|other| Arc::ptr_eq(mapping, other)) && backing.offset == *offset as usize
            }),
            _ => false,
        }
    }

    fn buffer(&self) -> Option<BufferSlice<'_>> {
        match self {
            Self::Device { buffer, chips, .. } => {
                let mut view = buffer.slice(..).expect("retained range");
                view.chips = *chips;
                Some(view)
            }
            _ => None,
        }
    }

    pub(crate) fn encode(operands: &[Self], window: u64, column: usize, words: &mut [u64]) {
        for (word, operand) in words.iter_mut().zip(operands) {
            *word = match operand {
                Self::Unbound => 0,
                Self::Value(value) => *value,
                Self::Device { buffer, .. } => buffer.operand(window),
                Self::Host { mapping, offset, .. } => mapping.operand(window, column, *offset),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Kind;
    use std::time::Instant;

    /// Repeated admission of a prepared large indirect table stays below one millisecond.
    #[test]
    #[ignore = "CPU launch latency threshold; run in an optimized test build"]
    fn bounded_indirect_binding() {
        // Given: a large table whose blocks partition one allocation.
        const COUNT: usize = 612720;
        let buffer = crate::test::allocations(0..COUNT * 512, COUNT * 512)
            .alloc(Kind::Dram, COUNT * 512)
            .unwrap();
        let mut behind: Vec<_> = (0..COUNT)
            .map(|index| buffer.alias(index * 512..(index + 1) * 512).unwrap())
            .collect();
        behind.dedup_by(|next, previous| previous.join(next));
        let mut arguments = Arguments::new(1, Weak::new());
        arguments.owner = Owner::Ordered(1);
        let mut samples = [0; 11];
        // When: every launch collects, reserves, admits and retires the prepared backing.
        for sample in &mut samples {
            let started = Instant::now();
            let held: Vec<_> = behind
                .iter()
                .map(|buffer| (buffer.slice(..).unwrap(), Access::Write))
                .collect();
            arguments.reserve(held.len());
            arguments.acquire(&held).unwrap();
            arguments.clear();
            *sample = started.elapsed().as_nanos();
        }
        // Then: bind plus admission and retirement fit the host latency budget.
        samples.sort_unstable();
        println!(
            "blocks={COUNT} backing={} median_ns={} min_ns={} max_ns={}",
            behind.len(),
            samples[5],
            samples[0],
            samples[10]
        );
        assert!(
            samples[5] < 1_000_000,
            "indirect bind median {} ns exceeds 1 ms",
            samples[5]
        );
    }

    /// Reports host admission and retirement cost for independently allocated indirect regions.
    #[test]
    #[ignore = "CPU indirect-region benchmark"]
    fn indirect_admission_bench() {
        for count in [4096, 612720] {
            let buffers: Vec<_> = (0..count)
                .map(|_| crate::test::allocations(0..512, 512).alloc(Kind::Dram, 512).unwrap())
                .collect();
            let mut arguments = Arguments::new(1, Weak::new());
            arguments.owner = Owner::Ordered(1);
            let mut samples = [[0; 4]; 11];
            for sample in &mut samples {
                let started = Instant::now();
                let held: Vec<_> = buffers
                    .iter()
                    .map(|buffer| (buffer.slice(..).unwrap(), Access::Write))
                    .collect();
                sample[0] = started.elapsed().as_nanos();
                let started = Instant::now();
                arguments.reserve(held.len());
                sample[1] = started.elapsed().as_nanos();
                let started = Instant::now();
                arguments.acquire(&held).unwrap();
                sample[2] = started.elapsed().as_nanos();
                let started = Instant::now();
                arguments.clear();
                sample[3] = started.elapsed().as_nanos();
            }
            for (phase, name) in ["collect", "reserve", "acquire", "clear"].into_iter().enumerate() {
                let mut values = samples.map(|sample| sample[phase]);
                values.sort_unstable();
                println!(
                    "regions={count} phase={name} median_ns={} min_ns={} max_ns={}",
                    values[5], values[0], values[10]
                );
            }
            crate::test::without_allocations(|| {
                let held = buffers.iter().map(|buffer| (buffer.slice(..).unwrap(), Access::Write));
                arguments.access.owner = arguments.owner;
                arguments.access.acquire(held).unwrap();
                arguments.clear();
            });
        }
    }
}
