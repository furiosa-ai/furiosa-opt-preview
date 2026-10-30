use crate::backend::Backend;

pub(super) struct Table<B: Backend> {
    pub(super) buffer: B::Buffer,
    pub(super) behind: Box<[B::Buffer]>,
}

impl<B: Backend> Table<B> {
    fn stride(blocks: &[(usize, usize)]) -> usize {
        blocks
            .first()
            .map(|block| block.1)
            .filter(|&size| {
                size > 0
                    && blocks.iter().enumerate().all(|(index, block)| {
                        block.1 == size
                            && index
                                .checked_mul(size)
                                .and_then(|offset| blocks[0].0.checked_add(offset))
                                == Some(block.0)
                    })
            })
            .unwrap_or(0)
    }

    pub(super) fn len(blocks: &[(usize, usize)]) -> usize {
        64 + if Self::stride(blocks) == 0 {
            12 * blocks.len()
        } else {
            0
        }
    }

    pub(super) fn encode(table: usize, blocks: &[(usize, usize)]) -> Vec<u8> {
        let stride = Self::stride(blocks);
        let origin = blocks.iter().map(|block| block.0).min().unwrap_or(table);
        let compact = blocks.iter().all(|block| {
            block
                .0
                .checked_sub(origin)
                .and_then(|offset| offset.checked_add(block.1))
                .is_some_and(|end| end <= (1usize << 32))
        });
        let header = [
            blocks.len() as u64,
            (origin as u64).wrapping_sub(table as u64),
            u64::from(compact),
            0,
            0,
            stride as u64,
            0,
            0,
        ];
        header
            .into_iter()
            .chain(
                blocks
                    .iter()
                    .filter(|_| stride == 0)
                    .map(|block| (block.0 as u64).wrapping_sub(table as u64)),
            )
            .flat_map(u64::to_le_bytes)
            // The C ABI reads offsets only when compact is set; otherwise they are unused.
            .chain(
                blocks
                    .iter()
                    .filter(|_| stride == 0)
                    .flat_map(|block| u32::try_from(block.0 - origin).unwrap_or(0).to_le_bytes()),
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    /// Address tables match the C header, signed-relative addresses and compact-offset ABI.
    #[test]
    fn encodes_address_tables() {
        for (blocks, expected) in [
            (vec![], vec![0, 0, 1, 0, 0, 0, 0, 0]),
            (vec![(512, 32), (544, 32)], vec![2, 256, 1, 0, 0, 32, 0, 0]),
        ] {
            let image = Table::<Cpu>::encode(256, &blocks);
            assert_eq!(
                image,
                expected.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>()
            );
            assert_eq!(image.len(), Table::<Cpu>::len(&blocks));
        }
        let image = Table::<Cpu>::encode(256, &[(128, 32), (512, 64)]);
        let expected = [2u64, u64::MAX - 127, 1, 0, 0, 0, 0, 0, u64::MAX - 127, 256]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .chain([0u32, 384].into_iter().flat_map(u32::to_le_bytes))
            .collect::<Vec<_>>();
        assert_eq!(image, expected);
        for (end, compact) in [(1usize << 32, 1), ((1usize << 32) + 1, 0)] {
            let image = Table::<Cpu>::encode(256, &[(128, 1), (128 + end - 1, 1)]);
            assert_eq!(u64::from_le_bytes(image[16..24].try_into().unwrap()), compact);
        }
    }
}
