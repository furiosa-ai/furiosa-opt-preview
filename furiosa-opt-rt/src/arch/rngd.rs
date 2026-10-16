//! RNGD's kernel-driver interface: device nodes, ioctl payloads, mapping offsets and PE fusion.

use std::ops::Range;
use std::time::Duration;

pub(crate) use furiosa_opt_abi::arch::rngd::*;

use crate::{Error, Result, Topology};

/// Where every chip's device nodes are.
pub(crate) const NODES: &str = "/dev/rngd";

const DOORBELL: u64 = 0x80_0000;
const PE_WINDOW: u64 = 512 << 20;
/// A PE's BAR2 window from its first doorbell ring through the doorbell registers.
pub(crate) const BAR2_LEN: usize = (DOORBELL + 0x1000) as usize - DOORBELLS.entries;
pub(crate) const CQ_REGION: i64 = 1 << 16;
pub(crate) const CQ_LEN: usize = 4096;
/// DRAM's offset in BAR4.
pub(crate) const BAR4_DRAM: u64 = 256 << 20;

/// Device memory the firmware's PEs address, past the driver's own reservation.
pub(crate) const DRAM_FIRMWARE: u64 = 512 << 20;
pub(crate) const WINDOW_DRAM: u8 = 0;

/// Where a chip's memory window may start in a cluster's address space; the bootloader tells
/// which by finding the firmware image at it.
pub(crate) const CANDIDATES: [u64; 3] = [0, 256 << 20, 512 << 20];

/// The PCIe outbound translation entries one chip has, each mapping this many bytes.
pub(crate) const ATU_ENTRIES: usize = 16;
pub(crate) const ATU_ENTRY: usize = 4 << 30;
/// Where the firmware sees the PDMA registers through the first translation entry.
pub(crate) const PDMA_REGISTERS: u64 = 0x80_4000_0000;
/// RNGD PDMA BAR0 register bank, including the driver's reserved register page.
pub(crate) const PDMA_EXTENT: u64 = (PDMA_REGISTERS - 0x80_0000_0000) + 0x1000;
/// RNGD PDMA reserves eight 8 KiB descriptor chains, one per physical channel.
pub(crate) const PDMA_DESCRIPTORS: usize = 8 * (8 << 10);
/// Where the firmware sees its own chip's memory for PDMA.
pub(crate) const LOCAL_DRAM: u64 = 0xc0_1000_0000;

pub(crate) const CAPS_DMABUF: i32 = 1;
pub(crate) const CAPS_RESERVE: i32 = 2;
pub(crate) const URING_CMD_SCATTER_READ: u32 = 0;
pub(crate) const URING_CMD_SCATTER_WRITE: u32 = 1;
pub(crate) const URING_CMD_DMABUF_READ: u32 = 3;
pub(crate) const URING_CMD_DMABUF_WRITE: u32 = 4;

pub(crate) fn node(chip: u8, name: &str) -> String {
    format!("{NODES}/npu{chip}{name}")
}

/// The firmware PDMA channels (4 to 7) cluster `cluster` of `clusters` on a chip may use.
pub(crate) fn channels(cluster: usize, clusters: usize) -> u8 {
    if clusters == 1 { 0xf0 } else { 0x30 << (cluster * 2) }
}

/// The driver's IPC deadline for `chip`, when sysfs tells it.
pub(crate) fn timeout(chip: u8) -> Option<Duration> {
    let path = format!("/sys/class/rngd_mgmt/rngd!npu{chip}mgmt/pe_ipc_timeout_ms");
    let millis = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| text.trim().parse().ok());
    if millis.is_none() {
        log::debug!("{path} is unreadable; the launch limit goes unchecked against the driver's");
    }
    millis.map(Duration::from_millis)
}

/// The signed bootloader to verify and start on the PEs in `pe_ids` once their node opens.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct Stage {
    pub(crate) pe_ids: u32,
    pub(crate) size: u32,
    pub(crate) data: u64,
}

nix::ioctl_readwrite!(stage, b'N', 0x04, Stage);

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct Window {
    pub(crate) kind: u8,
    pub(crate) pes: u8,
    pub(crate) base: u64,
    pub(crate) size: u64,
}

nix::ioctl_readwrite!(set_window, b'N', 0x03, Window);

#[repr(C)]
pub(crate) struct DmaMap {
    pub(crate) fd: i32,
    pub(crate) capacity: u32,
    pub(crate) segments: u64,
    pub(crate) count: u32,
    pub(crate) reserved: u32,
    pub(crate) channels: [i32; 4],
    pub(crate) size: u64,
}

nix::ioctl_readwrite!(dma_map, b'N', 1, DmaMap);

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Peer {
    pub(crate) file: u32,
    pub(crate) npu: u32,
    pub(crate) dram: u64,
    pub(crate) size: u64,
}

#[repr(C, packed)]
pub(crate) struct GroupConfig {
    pub(crate) local: u32,
    pub(crate) ats: u16,
    pub(crate) remotes: u16,
    pub(crate) peers: [Peer; MAX_CHIPS as usize],
}

nix::ioctl_readwrite!(set_group, b'N', 0x05, GroupConfig);
nix::ioctl_readwrite!(physical_id, b'N', 0x04, ());

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub(crate) struct BarInfo {
    pub(crate) phys: u64,
    pub(crate) size: u64,
}

nix::ioctl_readwrite!(available_dram, b'N', 0x02, BarInfo);

#[repr(C, packed)]
pub(crate) struct Region {
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) fd: i32,
}

#[repr(C, packed)]
pub(crate) struct Entry {
    pub(crate) fd: i32,
    pub(crate) offset: u32,
    pub(crate) index: u16,
    pub(crate) ats: u16,
}

nix::ioctl_readwrite!(bar_info, b'N', 0, BarInfo);
nix::ioctl_readwrite!(export, b'N', 1, Region);
nix::ioctl_readwrite!(set_entry, b'N', 7, Entry);
nix::ioctl_readwrite!(unset_entry, b'N', 8, u16);
nix::ioctl_none!(reserve, b'N', 5);
nix::ioctl_none!(get_caps, b'N', 0x00);

/// One contiguous piece of a scatter list, in the driver's layout.
#[repr(C)]
pub(crate) struct Segment {
    pub(crate) src: u64,
    pub(crate) dst: u64,
    pub(crate) len: u64,
}

const _: () = assert!(std::mem::size_of::<Segment>() == 24);

/// The PEs one cluster fuses on a chip: `count` (1, 2 or 4) from `lead`, a multiple of `count`,
/// exactly the driver's fusion nodes. The lead PE's queue and control window are the cluster's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PeRange {
    lead: u8,
    count: u8,
}

impl PeRange {
    /// The clusters that run `pes` on a chip: an aligned slice of 1, 2 or 4 PEs, or for all
    /// eight, both four-PE clusters.
    pub(crate) fn clusters(pes: Range<u8>) -> Result<Vec<Self>> {
        let count = pes.len() as u8;
        if !matches!(count, 1 | 2 | 4 | 8) || !pes.start.is_multiple_of(count) || pes.end > PES {
            return Err(Error::Invalid(format!(
                "invalid PEs {pes:?}: expected an aligned slice of 1, 2, 4 or 8"
            )));
        }
        let count = count / clusters(Topology { chips: 1, pes: count }) as u8;
        Ok(pes.step_by(count.into()).map(|lead| Self { lead, count }).collect())
    }

    /// The doorbell queue of this cluster on the chip ranked `member` in its device.
    pub(crate) fn queue(self, member: u8) -> u8 {
        member * PES + self.lead
    }

    pub(crate) fn window(self) -> u64 {
        u64::from(self.lead) * PE_WINDOW
    }

    /// The driver's bitmask of these PEs.
    pub(crate) fn mask(self) -> u8 {
        ((1 << self.count) - 1) << self.lead
    }

    pub(crate) fn node(self) -> String {
        match self.count {
            1 => format!("pe{}", self.lead),
            count => format!("pe{}-{}", self.lead, self.lead + count - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cluster is a kernel driver fusion node, with the driver's name, PE mask, queue and window.
    #[test]
    fn clusters_are_driver_fusion_nodes() {
        for (pes, nodes) in [
            (0..8, vec![(0x0f, 0, 0, "pe0-3"), (0xf0, 4, 0x8000_0000, "pe4-7")]),
            (4..8, vec![(0xf0, 4, 0x8000_0000, "pe4-7")]),
            (6..8, vec![(0xc0, 6, 0xc000_0000, "pe6-7")]),
            (5..6, vec![(0x20, 5, 0xa000_0000, "pe5")]),
        ] {
            let clusters = PeRange::clusters(pes).unwrap();
            assert_eq!(clusters.len(), nodes.len());
            for (cluster, node) in clusters.into_iter().zip(nodes) {
                let route = (cluster.mask(), cluster.queue(0), cluster.window(), &*cluster.node());
                assert_eq!(route, node);
            }
        }
        for pes in [0..0, 0..3, 2..6, 1..3, 8..16] {
            assert!(PeRange::clusters(pes.clone()).is_err(), "{pes:?}");
        }
    }
}
