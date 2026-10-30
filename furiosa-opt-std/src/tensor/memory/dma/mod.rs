//! The families that move data between placements.
//!
//! `direct` copies between two placements that are both known. `gather` and `scatter` copy
//! through an index. `layout` holds the rule a DM destination must satisfy, so every `to_dm`
//! here checks it. `scatter` reads DM and writes HBM, and takes no DM destination at all.
//!
//! [`super::redistribute`] is not a family here: it owns no transfer of its own and composes
//! `direct` copies instead.

mod direct;
mod gather;
mod layout;
mod scatter;

pub use direct::{ToHbm, ToHost};
pub use gather::*;
pub use scatter::*;
