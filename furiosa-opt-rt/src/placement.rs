//! Where a device runs: a topology resolved to physical chips and the PEs used on each.

use std::ops::Range;

pub use furiosa_opt_abi::Topology;

use crate::arch::rngd;
use crate::{Error, Result};

/// A topology resolved to physical chips, in rank order, and the PEs used on each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub chips: Vec<u8>,
    pub pes: Range<u8>,
}

/// The placements `Device::new` tries, in order.
pub trait ToPlacements {
    fn to_placements(&self) -> Result<Vec<Placement>>;
}

impl ToPlacements for Placement {
    fn to_placements(&self) -> Result<Vec<Placement>> {
        Ok(vec![self.clone()])
    }
}

impl ToPlacements for &[Placement] {
    fn to_placements(&self) -> Result<Vec<Placement>> {
        Ok(self.to_vec())
    }
}

impl ToPlacements for Topology {
    /// Every placement of `topology` on the installed chips `FURIOSA_VISIBLE_CHIPS` lists (all of
    /// them when it is unset), lowest chips and PEs first.
    fn to_placements(&self) -> Result<Vec<Placement>> {
        let visible = match std::env::var("FURIOSA_VISIBLE_CHIPS") {
            Err(std::env::VarError::NotPresent) => String::new(),
            spec => spec.map_err(|why| Error::Invalid(format!("FURIOSA_VISIBLE_CHIPS: {why}")))?,
        };
        let visible = visible
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(|id| {
                id.parse()
                    .map_err(|_| Error::Invalid(format!("FURIOSA_VISIBLE_CHIPS: `{id}` is not a chip index")))
            })
            .collect::<Result<Vec<u8>>>()?;
        let installed = std::fs::read_dir(crate::driver::kernel::NODES)
            .map_err(|why| Error::Device(format!("listing {}: {why}", crate::driver::kernel::NODES)))?
            .filter_map(|entry| {
                let name = entry.ok()?.file_name();
                name.to_str()?.strip_prefix("npu")?.strip_suffix("mgmt")?.parse().ok()
            })
            .filter(|chip| visible.is_empty() || visible.contains(chip));
        Ok(Placement::place(*self, installed.collect()))
    }
}

impl Placement {
    pub(crate) fn place(Topology { chips: count, pes }: Topology, mut chips: Vec<u8>) -> Vec<Self> {
        fn subsets(chips: &[u8], count: usize) -> Vec<Vec<u8>> {
            match chips {
                _ if count == 0 => vec![Vec::new()],
                [] => Vec::new(),
                [first, rest @ ..] => subsets(rest, count - 1)
                    .into_iter()
                    .map(|tail| [&[*first], &tail[..]].concat())
                    .chain(subsets(rest, count))
                    .collect(),
            }
        }
        chips.sort_unstable();
        chips.dedup();
        let slices = (0..rngd::PES)
            .step_by(pes.max(1).into())
            .map(|first| first..first + pes);
        subsets(&chips, count.into())
            .into_iter()
            .flat_map(|chips| {
                slices.clone().map(move |pes| Self {
                    chips: chips.clone(),
                    pes,
                })
            })
            .collect()
    }
}
