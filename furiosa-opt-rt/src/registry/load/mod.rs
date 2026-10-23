//! Loading an [`Entry`]: argument binding and completion processing of the compiler's images.
//!
//! - Implements the bind callback [`Function::load`] takes.

use std::sync::Arc;

use super::image::{Image, Parts, Slot};
use super::{Entry, Key};
use crate::memory::Kind;
use crate::{Argument, Arguments, Buffer, Device, Error, Function, Result};

mod profile;
use profile::{Profile, Report};

impl Entry {
    /// Loads the image onto `device` and registers every launch through host completion. A
    /// profiled function's launches report the spans the image names on [`TARGET`](super::TARGET),
    /// in device cycles.
    ///
    /// # Safety
    /// The image's code must access only its slots, with the modes its inputs and outputs give, as
    /// the compiler's images of [`entries`](super::entries) do. Private stacks must remain
    /// inaccessible to callers and their reuse must be ordered by the generated synchronization.
    pub async unsafe fn load(&self, device: &Arc<Device>, profile: Option<super::Profile>) -> Result<Function> {
        let image = Image::parse(self.image).map_err(Error::Invalid)?;
        if image.topology != device.topology() {
            return Err(Error::Invalid(format!(
                "image topology {:?} differs from device topology {:?}",
                image.topology,
                device.topology()
            )));
        }
        let weight = match image.weight {
            Some(weight) => {
                let shard = weight.bytes.len() / usize::from(image.topology.chips);
                let mut buffer = device.alloc(weight.kind.into(), shard)?;
                device.write([(weight.bytes.into(), buffer.slice_mut(..)?)]).await?;
                Some(buffer)
            }
            None => None,
        };
        let [first, second] = image.stacks.map(|slot| {
            (slot.size != 0)
                .then(|| device.alloc(slot.kind.into(), slot.size as usize))
                .transpose()
        });
        let stacks = [first?, second?];
        let profile = profile.and_then(|profile| {
            // Spans name the kernel as the driver names its artifact: the path, then any key; a
            // type-keyed instantiation by its function-item type name.
            let name = match self.key() {
                Key::Numeric([]) => self.path.to_string(),
                Key::Numeric(key) => format!(
                    "{}[{}]",
                    self.path,
                    key.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
                ),
                Key::Type(instance) => instance.to_string(),
            };
            Profile::new(&image, profile.level, name)
                .inspect_err(|error| (profile.refused)(profile.level, error))
                .ok()?
                .map(Arc::new)
        });
        let Parts {
            inputs,
            outputs,
            slots,
            executable,
            sram_end,
            ..
        } = image.into_parts();
        let binder = Binder {
            weight,
            stacks,
            signature: Signature { inputs, outputs, slots },
            profile,
        };
        let capacity = operand::SLOTS + binder.signature.slots.len();
        // SAFETY: binding checks caller regions and modes; generated code confines its effects to
        // those operands and private compiler storage ordered by its DRAM-reuse synchronization.
        unsafe {
            Function::load(
                device,
                executable.bytes(),
                sram_end as usize,
                capacity,
                move |args, arguments| binder.bind(args, arguments),
            )
            .await
            .map(|function| function.ordered())
        }
    }
}

/// A loaded compiler image's binder; admitted calls retain its device resources through completion.
struct Binder {
    weight: Option<Buffer>,
    /// The compiler's two stacks; a stack of size zero has no buffer.
    stacks: [Option<Buffer>; 2],
    signature: Signature,
    profile: Option<Arc<Profile>>,
}

impl Binder {
    fn bind(&self, args: &mut crate::Supplied<'_, '_>, arguments: &mut Arguments) -> Result<Report> {
        // Stacks are read operands so concurrent calls share them; lir_to_rlir drains every DRAM
        // reuse sync target before a later launch can reuse their bytes.
        let stacks = (operand::STACKS..).zip(&self.stacks);
        for (index, buffer) in std::iter::once((operand::WEIGHT, &self.weight)).chain(stacks) {
            if let Some(buffer) = buffer {
                arguments.bind(index, buffer.slice(..)?.into())?;
            }
        }
        self.signature.bind(args, arguments)?;
        Ok(Report::new(self.profile.as_ref()))
    }
}

/// A launch's argument bindings and memory requirements.
struct Signature {
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    slots: Vec<Slot>,
}

impl Signature {
    fn bind(&self, supplied: &mut crate::Supplied<'_, '_>, arguments: &mut Arguments) -> Result<()> {
        let mut bindings = self
            .inputs
            .iter()
            .chain(self.outputs.iter().filter(|slot| !self.inputs.contains(slot)));
        supplied(&mut |argument| {
            let &index = bindings
                .next()
                .ok_or_else(|| Error::Invalid("the supplied args exceed the image signature".into()))?;
            let slot = &self.slots[index as usize];
            let (view, write) = match argument {
                Argument::Read(view) => (view, false),
                Argument::Write(view) => (view, true),
                _ => return Err(Error::Invalid("a tensor slot requires an allocation".into())),
            };
            let has = view.kind();
            if slot.kind != has && !(slot.kind == Kind::Dram && has == Kind::DramShared) {
                return Err(Error::Invalid(format!(
                    "slot {index} expects a {:?} buffer, the argument is a {has:?} one",
                    slot.kind
                )));
            }
            if (view.len() as u64) < slot.size {
                return Err(Error::Invalid(format!(
                    "slot {index} needs {} bytes on every chip, the buffer has {}",
                    slot.size,
                    view.len()
                )));
            }
            if has == Kind::Sram && !(view.addr() as u64).is_multiple_of(furiosa_opt_abi::arch::rngd::sram::ALIGNMENT) {
                return Err(Error::Invalid(format!(
                    "SRAM slot {index} is not aligned to the resident address alignment"
                )));
            }
            if self.outputs.contains(&index) && !write {
                return Err(Error::Invalid("an output slot requires write access".into()));
            }
            let view = view.slice(..slot.size as usize)?;
            let argument = if write {
                Argument::Write(view)
            } else {
                Argument::Read(view)
            };
            arguments.bind(operand::SLOTS + index as usize, argument)?;
            Ok(())
        })?;
        if bindings.next().is_some() {
            return Err(Error::Invalid(
                "the supplied args are fewer than the image signature".into(),
            ));
        }
        Ok(())
    }
}

/// The operands ahead of a launch's slots, in the order the compiler's device code reads them
/// (`enum dram_kind` in its `primitive_functions.c`); its `DRAM_RESERVED`, operand 1, stays unbound.
mod operand {
    pub const WEIGHT: usize = 0;
    pub const STACKS: usize = 2;
    pub const SLOTS: usize = 4;
}
