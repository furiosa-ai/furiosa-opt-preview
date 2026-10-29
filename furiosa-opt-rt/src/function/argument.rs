//! Borrowed call arguments and retained operand storage.

use std::sync::{Arc, Weak};

use crate::buffer::{Lease, Owner, Region};
use crate::pinned::{DmaBuf, Registration};
use crate::{Access, Buffer, BufferSlice, BufferSliceMut, Device, Error, Hold, HostBufferSliceMut, Result};

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
    /// holds it as a whole, excluding everything except distinct ordered launches on the same registration.
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
    values: Vec<Operand>,
    access: Lease,
    owner: Owner,
    limit: usize,
    /// Completes the launches and transfers a host operand's lease conflicts with.
    device: Weak<Device>,
}

impl Arguments {
    pub(crate) fn new(capacity: usize, device: Weak<Device>) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            access: Lease::new(capacity + IMAGE),
            owner: Owner::Unordered,
            limit: capacity,
            device,
        }
    }

    /// Binds operand `index`; operands that are never bound are zero, and [`Error::Capacity`]
    /// past the function's argument count. Binding an index again must name the same operand,
    /// for memory its start, length, chips and access, so an image may list
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
                self.values[index] = Operand::new(argument, &device)?;
                Ok(())
            }
            bound if bound.names(&argument) => Ok(()),
            _ => Err(Error::Invalid(format!("operand {index} is bound to different storage"))),
        }
    }

    /// Makes `owner` the owner of the memory the next binding holds.
    pub(super) fn set_owner(&mut self, owner: Owner) {
        self.owner = owner;
    }

    /// Room for `regions` the caller holds beside the operands and the staged image.
    pub(super) fn reserve(&mut self, regions: usize) {
        if self.limit + IMAGE + regions > self.access.capacity() {
            self.access = Lease::new(self.limit + IMAGE + regions);
        }
    }

    pub(super) fn acquire<'a>(&mut self, held: impl IntoIterator<Item = Hold<'a>>) -> Result<()> {
        self.access.set_owner(self.owner);
        self.access.admit(
            self.values
                .iter()
                .filter_map(|operand| match operand {
                    Operand::Device { access, .. } => {
                        Some(Region::from((operand.buffer().expect("device operand"), *access)))
                    }
                    Operand::Host {
                        allocation,
                        registration,
                        offset,
                        len,
                    } => Some(Region::host(
                        crate::pinned::Backing {
                            dmabuf: allocation,
                            registration: Some(registration),
                            offset: *offset as usize,
                        },
                        *len,
                        Access::Write,
                    )),
                    Operand::Unbound | Operand::Value(_) => None,
                })
                .chain(held.into_iter().map(Region::from)),
        )
    }

    pub(super) fn poison(&self) {
        self.access.poison();
    }

    pub(crate) fn clear(&mut self) {
        self.access.clear();
        self.values.clear();
    }

    /// The operands bound so far, in argument order.
    pub(super) fn operands(&self) -> &[Operand] {
        &self.values
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
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
        registration: Arc<Registration>,
        offset: u32,
        len: usize,
        allocation: Arc<DmaBuf>,
    },
}

impl Operand {
    /// `argument` as an operand of a launch on `device`.
    pub(crate) fn new(argument: Argument<'_>, device: &Device) -> Result<Self> {
        Ok(match argument {
            Argument::Value(value) => Self::Value(value),
            Argument::Host(view) => {
                device.driver.pdma()?;
                let backing = view
                    .pinned
                    .ok_or_else(|| Error::Invalid("a host operand must be pinned".into()))?;
                let registration = backing
                    .mapped_to(&device.allocator)
                    .ok_or_else(|| Error::Invalid("host memory has not been mapped to this device".into()))?;
                Self::Host {
                    registration: Arc::clone(registration),
                    offset: backing.offset() as u32,
                    len: view.len(),
                    allocation: Arc::clone(backing.dmabuf),
                }
            }
            _ => {
                let (view, access) = argument.device().expect("device argument");
                Self::Device {
                    buffer: view.retain(),
                    chips: view.chip_mask(),
                    access,
                }
            }
        })
    }

    pub(crate) fn encode(operands: &[Self], window: u64, column: usize, words: &mut [u64]) {
        for (word, operand) in words.iter_mut().zip(operands) {
            *word = match operand {
                Self::Unbound => 0,
                Self::Value(value) => *value,
                Self::Device { buffer, .. } => buffer.operand(window),
                Self::Host {
                    registration, offset, ..
                } => registration.operand(window, column, *offset),
            };
        }
    }

    /// Whether `argument` names this operand: the same value, or the same start, chips and access
    /// of the same memory.
    fn names(&self, argument: &Argument<'_>) -> bool {
        match (self, argument) {
            (Self::Value(value), Argument::Value(other)) => value == other,
            (Self::Device { access, .. }, _) => match (self.buffer(), argument.device()) {
                (Some(bound), Some((view, other))) => {
                    bound.ptr_eq(&view) && bound.len() == view.len() && *access == other
                }
                _ => false,
            },
            (
                Self::Host {
                    registration,
                    offset,
                    len,
                    ..
                },
                Argument::Host(view),
            ) => view.pinned.is_some_and(|backing| {
                backing
                    .registration
                    .is_some_and(|other| Arc::ptr_eq(registration, other))
                    && backing.offset() == *offset as usize
                    && view.len() == *len
            }),
            _ => false,
        }
    }

    fn buffer(&self) -> Option<BufferSlice<'_>> {
        match self {
            Self::Device { buffer, chips, .. } => Some(buffer.slice(..).expect("retained range").selecting(*chips)),
            _ => None,
        }
    }
}

/// Regions a launch holds beside its operands and the caller's: the staged image.
const IMAGE: usize = 1;

#[cfg(test)]
mod tests {
    use super::*;

    /// Rebinding an operand cannot silently narrow the memory declared by the caller.
    #[test]
    fn refuses_different_extent() {
        let device = Arc::new(crate::test::device(&[0], 8));
        let buffer = device.alloc(crate::memory::Kind::Dram, 512).unwrap();
        let mut arguments = Arguments::new(1, Arc::downgrade(&device));
        arguments
            .bind(0, Argument::Write(buffer.slice(..100).unwrap()))
            .unwrap();
        assert!(matches!(
            arguments.bind(0, Argument::Write(buffer.slice(..200).unwrap())),
            Err(Error::Invalid(_)),
        ));
    }
}
