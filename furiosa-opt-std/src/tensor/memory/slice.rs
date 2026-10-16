use std::any::Any;
use std::ops::{Deref, DerefMut, Range};
use std::sync::OnceLock;

use furiosa_mapping::{Ident, M, Mapping, MappingExt};

use super::HbmTensor;
use crate::Error;
use crate::backend::{Allocation, Backend};
use crate::context::{Dma, DmaContext};
use crate::runtime::{Access, DeviceSend, HbmBufferSlice, HbmBufferSliceMut, HostBufferSlice};
use crate::scalar::{MaterializableScalar, Scalar};
use furiosa_opt_macro::primitive;

mod table;
use table::Table;

/// The axis that counts `len` blocks, named apart from every axis of a block.
pub(super) fn outer<Chip: M, Element: M>(len: usize) -> Mapping {
    let (chip, element) = (Chip::to_value(), Element::to_value());
    let mut name = String::from("HbmOuter");
    while chip
        .axes()
        .iter()
        .chain(element.axes().iter())
        .any(|axis| axis.symbol.as_str() == name)
    {
        name.push('_');
    }
    Mapping::Symbol {
        symbol: Ident::try_from(name.as_str()).unwrap(),
        size: len,
    }
}

pub(super) fn mapping<Chip: M, Element: M>(len: usize) -> Mapping {
    Chip::to_value().pair(outer::<Chip, Element>(len).pair(Element::to_value()))
}

pub(super) fn storage<const DMA: Dma, D: MaterializableScalar, Chip: M, Element: M, B: Backend>(
    dma: &DmaContext<DMA, B>,
    blocks: &[HbmTensor<D, Chip, Element, B>],
) -> B::Storage<D> {
    let values: Vec<_> = blocks.iter().map(|block| block.read(dma).into_vec()).collect();
    B::from_vec(
        &mapping::<Chip, Element>(blocks.len()),
        (0..Chip::SIZE).flat_map(|chip| {
            values
                .iter()
                .flat_map(move |block| block[chip * Element::SIZE..(chip + 1) * Element::SIZE].iter().copied())
        }),
    )
}

/// An owned runtime-length collection. Tensor elements may have independent device allocations.
#[primitive(Slice)]
#[derive(Debug)]
pub struct Slice<T> {
    pub(super) values: Vec<T>,
    table: OnceLock<Box<dyn Any + Send + Sync>>,
}

impl<T> From<Vec<T>> for Slice<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            values,
            table: OnceLock::new(),
        }
    }
}

impl<T> FromIterator<T> for Slice<T> {
    fn from_iter<I: IntoIterator<Item = T>>(values: I) -> Self {
        values.into_iter().collect::<Vec<_>>().into()
    }
}

impl<T> Deref for Slice<T> {
    type Target = [T];

    #[primitive(Slice::deref)]
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> DerefMut for Slice<T> {
    #[primitive(Slice::deref)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.invalidate();
        &mut self.values
    }
}

impl<T> Slice<T> {
    /// Retains at most the first `len` elements.
    pub fn truncate(&mut self, len: usize) {
        if len < self.values.len() {
            self.invalidate();
            self.values.truncate(len);
        }
    }

    fn invalidate(&mut self) {
        self.table.take();
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> Slice<HbmTensor<D, Chip, Element, B>> {
    /// Aliases `range` of these blocks, retaining their current device allocations.
    ///
    /// # Safety
    /// Both slices name the same memory: an access through one must complete before an
    /// overlapping access through the other starts.
    ///
    /// # Panics
    /// Panics if `range` is reversed or extends past the end, as slicing does.
    pub unsafe fn alias(&self, range: Range<usize>) -> Self {
        self.values[range]
            .iter()
            .map(|block| HbmTensor::from_buffer(block.buffer.alias(..).unwrap_or_else(|error| panic!("{error}"))))
            .collect()
    }
}

impl<T> Extend<T> for Slice<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, values: I) {
        self.invalidate();
        self.values.extend(values);
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend<B> for Slice<HbmTensor<D, Chip, Element, B>> {
    fn prepare(&self, device: &B::Device) -> Result<(), Error> {
        if self.table.get().is_some() {
            return Ok(());
        }
        let mut behind = self
            .values
            .iter()
            .map(|block| block.buffer.alias(..))
            .collect::<Result<Vec<_>, _>>()?;
        let blocks: Vec<_> = behind.iter().map(|buffer| (buffer.addr(), buffer.size())).collect();
        // Overlapping entries stay separate so writable aliases still conflict within a launch.
        behind.dedup_by(|next, previous| previous.join(next));
        let behind = behind.into_boxed_slice();
        let mut buffer = B::alloc(device, Table::<B>::len(&blocks))?;
        let image = Table::<B>::encode(buffer.addr(), &blocks).repeat(usize::from(B::topology(device).chips));
        futures_executor::block_on(tokio::task::unconstrained(B::write(
            device,
            vec![(HostBufferSlice::from(&image), HbmBufferSliceMut::whole(&mut buffer))],
        )))?;
        // ponytail: racing first launches build separate tables; only the published one survives.
        let _ = self.table.set(Box::new(Table::<B> { buffer, behind }));
        Ok(())
    }

    fn arguments(&self, access: Access) -> Result<impl Iterator<Item = (HbmBufferSlice<'_, B>, Access)>, Error> {
        let table = self
            .table
            .get()
            .and_then(|table| table.downcast_ref::<Table<B>>())
            .ok_or(Error::Invalid(
                "a slice's address table must be prepared before binding",
            ))?;
        Ok(std::iter::once((
            HbmBufferSlice {
                behind: &table.behind,
                ..HbmBufferSlice::whole(&table.buffer)
            },
            access,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cached backing compression preserves the address-table ABI, gaps and overlapping entries.
    #[test]
    fn preserves_compacted_tables() {
        use crate::backend::Cpu;
        use crate::prelude::*;
        let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();
        let buffer = HbmBuffer::alloc(&device, 1536).unwrap();
        for (ranges, count) in [
            (vec![0..512, 512..1024], 1),
            (vec![0..512, 1024..1536], 2),
            (vec![0..512, 256..768], 2),
        ] {
            let values: Slice<HbmTensor<u8, m![2], m![512], Cpu>> = ranges
                .into_iter()
                .map(|range| HbmTensor::from_buffer(buffer.0.alias(range).unwrap()))
                .collect();
            let blocks: Vec<_> = values.iter().map(|block| (block.buffer.addr(), 512)).collect();
            values.prepare(device.pdma.device()).unwrap();
            let table = values.table.get().unwrap().downcast_ref::<Table<Cpu>>().unwrap();
            assert_eq!(table.behind.len(), count);
            let mut image = vec![0; table.buffer.size() * 2];
            futures_executor::block_on(async {
                HbmBufferSlice::whole(&table.buffer)
                    .to_host(&mut device.pdma)
                    .output(&mut image)
                    .await
            })
            .unwrap();
            assert_eq!(image, Table::<Cpu>::encode(table.buffer.addr(), &blocks).repeat(2));
            values.prepare(device.pdma.device()).unwrap();
            assert!(std::ptr::eq(
                table,
                values.table.get().unwrap().downcast_ref::<Table<Cpu>>().unwrap()
            ));
        }
    }

    #[test]
    fn invalidates_mutable_access() {
        let mut values = Slice::from(vec![1, 2, 3]);
        values.table.set(Box::new(())).unwrap();
        values[1] = 4;
        assert!(values.table.get().is_none());
        assert_eq!(&*values, [1, 4, 3]);

        values.table.set(Box::new(())).unwrap();
        values.swap(1, 2);
        assert!(values.table.get().is_none());
        assert_eq!(&*values, [1, 3, 4]);

        values.table.set(Box::new(())).unwrap();
        values.iter_mut().for_each(|value| *value += 1);
        assert!(values.table.get().is_none());
        assert_eq!(&*values, [2, 4, 5]);

        values.table.set(Box::new(())).unwrap();
        values.extend([6, 7]);
        assert!(values.table.get().is_none());
        assert_eq!(&*values, [2, 4, 5, 6, 7]);

        values.table.set(Box::new(())).unwrap();
        values.truncate(3);
        assert!(values.table.get().is_none());
        assert_eq!(&*values, [2, 4, 5]);

        values.table.set(Box::new(())).unwrap();
        let borrowed: &mut [i32] = &mut values;
        assert_eq!(borrowed.len(), 3);
        assert!(values.table.get().is_none());
    }

    #[test]
    fn preserves_moved_binding() {
        let values = Slice::from(vec![1]);
        values.table.set(Box::new(())).unwrap();
        let moved = Box::new(values);
        assert!(moved.table.get().unwrap().is::<()>());
    }

    #[test]
    fn scatter_preserves_membership() {
        use crate::prelude::*;

        axes![Count = 64];
        let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 1 }).unwrap();
        let block = HbmTensor::<i32, m![1], m![Count], Cpu>::from_value(&device.tdma, &Tensor::from_vec([7; 64]));
        let indices = HbmTensor::<i32, m![1], m![Count], Cpu>::from_value(
            &device.tdma,
            &Tensor::from_vec((0..64).map(|i| i * 4)),
        );
        let updates = DmTensor::<i32, m![1], m![1], m![Count], m![1], Cpu>::from_parts(Tensor::from_vec([9; 64]));
        let mut values = Slice::from(vec![block]);
        values.table.set(Box::new(())).unwrap();
        values
            .scatter::<m![Count], m![1]>()
            .by_byte_offsets(indices.view())
            .from_dm(&mut device.tdma, updates);
        assert_eq!(values[0].load().into_vec(), [9; 64]);
        assert!(values.table.get().unwrap().is::<()>());
    }

    #[test]
    fn keeps_chip_indices_independent() {
        use crate::prelude::*;

        axes![Chips = 2, Rows = 2, Width = 2, Count = 2];
        type Block = HbmTensor<i32, m![Chips], m![Rows, Width], Cpu>;
        let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 4 }).unwrap();
        let mut blocks = Slice::from(vec![
            Block::from_value(&device.tdma, &Tensor::zeroed()),
            Block::from_value(&device.tdma, &Tensor::zeroed()),
        ]);
        let updates =
            DmTensor::<i32, m![Chips], m![1], m![1 # 64], m![Count, Width], Cpu>::from_parts(Tensor::from_vec(
                (0..2).flat_map(|chip| (0..64).flat_map(move |_| [10, 11, 20, 21].map(|value| value + chip * 100))),
            ));
        let offsets =
            HbmTensor::<i32, m![Chips], m![Count], Cpu>::from_value(&device.tdma, &Tensor::from_vec([24, 0, 0, 16]));
        blocks
            .scatter::<m![Rows], m![1 # 64, Width]>()
            .by_byte_offsets(offsets.view())
            .from_dm(&mut device.tdma, updates);
        assert_eq!(blocks[0].load().into_vec(), [20, 21, 0, 0, 110, 111, 0, 0]);
        assert_eq!(blocks[1].load().into_vec(), [0, 0, 10, 11, 120, 121, 0, 0]);

        let offsets =
            HbmTensor::<i32, m![Chips], m![Count], Cpu>::from_value(&device.tdma, &Tensor::from_vec([16, 0, 0, 16]));
        let lengths = HbmTensor::<i32, m![Chips], m![1], Cpu>::from_value(&device.tdma, &Tensor::from_vec([1, 2]));
        let gathered = blocks
            .gather::<m![1], m![Rows, Width]>()
            .by_byte_offsets(&offsets)
            .prefix::<m![Count], _>(&lengths, 0)
            .to_dm::<m![1], m![1 # 64], m![Count, Rows, Width]>(&mut device.tdma)
            .to_hbm::<m![Count, Rows, Width]>(&mut device.tdma);
        let actual = gathered.load().into_vec();
        assert_eq!(&actual[..4], &[0, 0, 10, 11]);
        assert_eq!(&actual[8..], &[110, 111, 0, 0, 120, 121, 0, 0]);
    }

    mod entry {
        use crate::prelude::*;

        axes![Chips = 2, Rows = 2, Width = 2, Count = 2];
        type Block = HbmTensor<i32, m![Chips], m![Rows, Width], Cpu>;
        type Rows2 = DmTensor<i32, m![Chips], m![1], m![1 # 64], m![Count, Width], Cpu>;

        fn blocks(device: &Device<Cpu>, count: usize) -> Slice<Block> {
            (0..count)
                .map(|_| Block::from_value(&device.tdma, &Tensor::zeroed()))
                .collect()
        }

        fn offsets(device: &Device<Cpu>, values: [i32; 4]) -> HbmTensor<i32, m![Chips], m![Count], Cpu> {
            HbmTensor::from_value(&device.tdma, &Tensor::from_vec(values))
        }

        fn updates() -> Rows2 {
            DmTensor::from_parts(Tensor::from_vec((0..2).flat_map(|chip| {
                (0..64).flat_map(move |_| [10, 11, 20, 21].map(|value| value + chip * 100))
            })))
        }

        #[test]
        fn addresses_blocks_from_the_entry() {
            let mut device = Device::<Cpu>::new(Topology { chips: 2, pes: 4 }).unwrap();
            let mut blocks = blocks(&device, 3);
            // A row is 8 bytes and a block 16: chip 0 writes block 2 row 1 and block 1 row 0, chip 1
            // writes block 1 row 0 and block 2 row 0.
            let offsets = offsets(&device, [24, 0, 0, 16]);
            blocks
                .scatter::<m![Rows], m![1 # 64, Width]>()
                .entry(1)
                .by_byte_offsets(offsets.view())
                .from_dm(&mut device.tdma, updates());
            assert_eq!(blocks[0].load().into_vec(), [0; 8]);
            assert_eq!(blocks[1].load().into_vec(), [20, 21, 0, 0, 110, 111, 0, 0]);
            assert_eq!(blocks[2].load().into_vec(), [0, 0, 10, 11, 120, 121, 0, 0]);

            let gathered: Rows2 = blocks
                .gather::<m![Rows], m![Width]>()
                .entry(1)
                .by_byte_offsets(&offsets)
                .to_dm(&mut device.tdma);
            assert_eq!(
                gathered.to_hbm::<m![Count, Width]>(&mut device.tdma).load().into_vec(),
                updates().to_hbm::<m![Count, Width]>(&mut device.tdma).load().into_vec()
            );
        }
    }

    mod rows {
        use crate::prelude::*;

        axes![Block = 4, Cell = 4, Key = 2];
        type Table = Slice<HbmTensor<i32, m![1], m![Block, Cell], Cpu>>;
        type Updates = DmTensor<i32, m![1], m![1], m![1 # 64], m![Key, Cell], Cpu>;

        fn table(device: &Device<Cpu>, rows: usize) -> Table {
            (0..rows)
                .map(|row| {
                    let start = i32::try_from(row * 16).unwrap();
                    HbmTensor::from_value(&device.tdma, &Tensor::from_vec(start..start + 16))
                })
                .collect()
        }

        fn updates() -> Updates {
            DmTensor::from_parts(Tensor::<i32, m![Key, Cell], Cpu>::from_vec(100..108).transpose(true))
        }

        fn cells(table: &Table) -> Vec<i32> {
            table.iter().flat_map(|block| block.load().into_vec()).collect()
        }

        #[test]
        fn addresses_last_entry() {
            let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
            for rows in [1, 3, 8193] {
                let table = table(&device, rows);
                let indices = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(
                    &device.tdma,
                    &Tensor::from_vec([0, ((rows - 1) * 64) as i32]),
                );
                let lengths = HbmTensor::<i32, m![1], m![1], Cpu>::from_value(&device.tdma, &Tensor::from_vec([2]));
                let gathered: DmTensor<i32, m![1], m![1], m![1 # 64], m![Key, Block, Cell], Cpu> = table
                    .gather::<m![1], m![Block, Cell]>()
                    .by_byte_offsets(&indices)
                    .prefix::<m![Key], _>(&lengths, 0)
                    .to_dm(&mut device.tdma);
                assert_eq!(
                    gathered.inner.transpose::<m![Key, Block, Cell]>(false).into_vec(),
                    (0..16)
                        .chain(((rows - 1) * 16) as i32..(rows * 16) as i32)
                        .collect::<Vec<_>>()
                );
                assert!(table.get(rows).is_none());
                assert!(table[rows..rows].is_empty());
            }
        }

        #[test]
        fn writes_preserve_other_cells() {
            let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
            let mut table = table(&device, 3);
            let indices = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&device.tdma, &Tensor::from_vec([16, 176]));
            table
                .scatter::<m![Block], m![1 # 64, Cell]>()
                .by_byte_offsets(indices.view())
                .from_dm(&mut device.tdma, updates());
            let mut expected: Vec<_> = (0..48).collect();
            expected[4..8].copy_from_slice(&[100, 101, 102, 103]);
            expected[44..48].copy_from_slice(&[104, 105, 106, 107]);
            assert_eq!(cells(&table), expected);
        }

        #[test]
        fn scatters_relative_to_entry() {
            let mut device = Device::<Cpu>::new(Topology { chips: 1, pes: 8 }).unwrap();
            let mut table = table(&device, 3);
            let indices = HbmTensor::<i32, m![1], m![Key], Cpu>::from_value(&device.tdma, &Tensor::from_vec([16, 48]));
            table
                .scatter::<m![Block], m![1 # 64, Cell]>()
                .entry(1)
                .by_byte_offsets(indices.view())
                .from_dm(&mut device.tdma, updates());
            let mut expected: Vec<_> = (0..48).collect();
            expected[20..24].copy_from_slice(&[100, 101, 102, 103]);
            expected[28..32].copy_from_slice(&[104, 105, 106, 107]);
            assert_eq!(cells(&table), expected);
        }
    }

    #[test]
    fn reuses_mutable_binding() {
        use crate::prelude::*;

        let values = Slice::<HbmTensor<i32, m![1], m![1], Cpu>>::from(Vec::new());
        assert!(matches!(values.arguments(Access::Write), Err(Error::Invalid(_))));
        let device = Device::<Cpu>::new(Topology { chips: 1, pes: 1 }).unwrap();
        values.prepare(device.pdma.device()).unwrap();
        let first = values.arguments(Access::Write).unwrap().next().unwrap().0.buffer.addr();
        values.prepare(device.pdma.device()).unwrap();
        assert_eq!(
            values.arguments(Access::Write).unwrap().next().unwrap().0.buffer.addr(),
            first
        );
    }
}
