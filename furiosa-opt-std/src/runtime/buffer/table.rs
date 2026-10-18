use super::{HbmBuffer, HbmBufferSlice, HbmBufferSliceMut, HostBufferSlice};
use crate::Error;
use crate::backend::{Allocation, Backend};
use crate::runtime::{Access, DeviceSend};

impl<B: Backend> DeviceSend<B> for HbmBuffer<B> {
    fn prepare(&self, device: &B::Device) -> Result<(), Error> {
        if self.1.get().is_some() {
            return Ok(());
        }
        let mut table = B::alloc(device, crate::runtime::Kind::Dram, 64)?;
        let image: Vec<_> = [
            self.size() as u64,
            (self.addr() as u64).wrapping_sub(table.addr() as u64),
            1,
            0,
            0,
            1,
            1,
            0,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
        let image = image.repeat(usize::from(B::topology(device).chips));
        futures_executor::block_on(tokio::task::unconstrained(B::write(
            device,
            vec![(HostBufferSlice::from(&image), HbmBufferSliceMut::whole(&mut table))],
        )))?;
        // ponytail: racing first launches publish one immutable table; the other is dropped.
        let _ = self.1.set(table);
        Ok(())
    }

    fn arguments(&self, access: Access) -> Result<impl Iterator<Item = (HbmBufferSlice<'_, B>, Access)>, Error> {
        let table = self.1.get().ok_or(Error::Invalid(
            "a buffer's address table must be prepared before binding",
        ))?;
        Ok(std::iter::once((
            HbmBufferSlice {
                behind: std::slice::from_ref(&self.0),
                ..HbmBufferSlice::whole(table)
            },
            access,
        )))
    }
}
