//! The kernel driver's interface: device nodes, mapping offsets, ioctl payloads and calls, and
//! sysfs.

use std::time::Duration;

use crate::arch::rngd::{DOORBELLS, MAX_CHIPS};

pub(crate) mod map;
pub(crate) mod window;

/// Where every chip's device nodes are.
pub(crate) const NODES: &str = "/dev/rngd";

const DOORBELL: u64 = 0x80_0000;
pub(crate) const PE_WINDOW: u64 = 512 << 20;
/// A PE's BAR2 window from its first doorbell ring through the doorbell registers.
pub(crate) const BAR2_LEN: usize = (DOORBELL + 0x1000) as usize - DOORBELLS.entries;
pub(crate) const CQ_REGION: i64 = 1 << 16;
pub(crate) const CQ_LEN: usize = 4096;
/// DRAM's offset in BAR4: the SoC shared region a PF build of the kernel driver skips
/// (`NPU_DRAM_BASE` in its `npu_address_map.h`). A VF build skips nothing, which this runtime does
/// not support.
pub(crate) const BAR4_DRAM: u64 = 256 << 20;

pub(crate) const WINDOW_DRAM: u8 = 0;

pub(crate) const CAPS_DMABUF: i32 = 1;
pub(crate) const CAPS_RESERVE: i32 = 2;
pub(crate) const URING_CMD_SCATTER_READ: u32 = 0;
pub(crate) const URING_CMD_SCATTER_WRITE: u32 = 1;
pub(crate) const URING_CMD_DMABUF_READ: u32 = 3;
pub(crate) const URING_CMD_DMABUF_WRITE: u32 = 4;

/// The signed bootloader to verify and start on the PEs in `pe_ids` once their node opens.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct Stage {
    pub(crate) pe_ids: u32,
    pub(crate) size: u32,
    pub(crate) data: u64,
}

nix::ioctl_readwrite!(stage, b'N', 0x04, Stage);

/// Where the PEs in `pes` see device memory: `size` bytes of the chip's `kind` memory at their
/// address `base`.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub(crate) struct PeWindow {
    pub(crate) kind: u8,
    pub(crate) pes: u8,
    pub(crate) base: u64,
    pub(crate) size: u64,
}

nix::ioctl_readwrite!(set_window, b'N', 0x03, PeWindow);

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

/// A scatter command's bytes: `count` segments at `desc_addr`, in pageable memory the driver pins
/// for the command.
#[repr(C)]
pub(crate) struct ScatterCommand {
    pub(crate) count: u64,
    pub(crate) desc_addr: u64,
}

/// A dma-buf command's bytes: `count` segments at `desc_addr`, in the dma-buf `fd`.
#[repr(C)]
pub(crate) struct DmabufCommand {
    pub(crate) fd: i32,
    pub(crate) count: u32,
    pub(crate) desc_addr: u64,
}

pub(crate) fn node(chip: u8, name: &str) -> String {
    format!("{NODES}/npu{chip}{name}")
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
