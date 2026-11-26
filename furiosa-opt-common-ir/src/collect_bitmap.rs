//! The packet positions a collect compaction keeps, shared by the eDSL, the lowering checks and the hardware IR.

use std::fmt::{self, Formatter};

use serde::{Deserialize, Serialize};

/// The packet positions a collect compaction keeps, bit `i` for position `i`; zero keeps nothing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CollectBitmap(u8);

impl CollectBitmap {
    /// The number of packet positions the bitmap can select.
    pub const BITS: u32 = u8::BITS;

    /// The bits, bit `i` for position `i`.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// The kept packet positions, counted from zero, ascending.
    pub fn positions(self) -> impl Iterator<Item = usize> {
        (0..Self::BITS as usize).filter(move |position| self.0 >> position & 1 == 1)
    }
}

impl fmt::Debug for CollectBitmap {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "CollectBitmap({}u8)", self.0)
    }
}

impl From<u8> for CollectBitmap {
    fn from(bits: u8) -> Self {
        Self(bits)
    }
}

impl From<CollectBitmap> for u8 {
    fn from(bitmap: CollectBitmap) -> Self {
        bitmap.bits()
    }
}

impl serde_lite::Deserialize for CollectBitmap {
    fn deserialize(val: &serde_lite::Intermediate) -> Result<Self, serde_lite::Error> {
        <u8 as serde_lite::Deserialize>::deserialize(val).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bitmap_keeps_the_positions_its_bits_name() {
        let bitmap = CollectBitmap::from(0b1000_1001);
        assert_eq!(bitmap.positions().collect::<Vec<_>>(), [0, 3, 7]);
        assert_eq!(CollectBitmap::from(0).positions().count(), 0);
    }
}
