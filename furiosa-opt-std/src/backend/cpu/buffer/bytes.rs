use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Host bytes shared until written: a clone shares them, and the first write to a shared value
/// copies it. A value read from a whole allocation shares the allocation's bytes, so reading a
/// tensor copies nothing until one side writes.
#[derive(Clone, Debug, Default)]
pub struct Bytes(Arc<Vec<u8>>);

impl From<Vec<u8>> for Bytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }
}

impl From<Bytes> for Vec<u8> {
    fn from(bytes: Bytes) -> Self {
        Arc::unwrap_or_clone(bytes.0)
    }
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.0).as_mut_slice()
    }
}
