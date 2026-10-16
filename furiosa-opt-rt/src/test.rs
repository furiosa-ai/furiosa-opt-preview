use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::Waker;

use crate::buffer::Allocations;
use crate::driver::dma::{DmaQueue, Transport, fake::Fake};
use crate::pinned::Mapping;
use crate::{Device, Pinned};

thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Bytes of device DRAM that [`device`] and its siblings give a test device.
const DRAM: usize = 4096;

struct Allocator;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

// SAFETY: allocation delegates to System; the default zeroing and reallocation call `alloc`.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get().map(|count| count + 1)));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

/// Runs `run`, then asserts that the calling thread allocated nothing during it. Other threads
/// are not counted; use an all-thread region where a worker thread matters.
pub(crate) fn without_allocations<T>(run: impl FnOnce() -> T) -> T {
    ALLOCATIONS.set(Some(0));
    let result = run();
    assert_eq!(ALLOCATIONS.replace(None), Some(0), "runtime hot path allocated");
    result
}

/// A one-chip allocator over `dram`, shared DRAM ending at `shared_end`, and the device's SRAM.
pub(crate) fn allocations(dram: Range<usize>, shared_end: usize) -> Arc<Allocations> {
    let ranges = crate::buffer::Ranges::new(dram, shared_end);
    Arc::new(Allocations::new((1, 8).into(), ranges))
}

pub(crate) fn pinned() -> Pinned {
    sized(4096)
}

fn sized(len: usize) -> Pinned {
    let file = tempfile::tempfile().unwrap();
    file.set_len(len as u64).unwrap();
    Pinned::new(file.into(), len).unwrap()
}

/// Host memory with one descriptor table per device cluster.
pub(crate) fn mapped(device: &Device, offsets: Vec<usize>, len: usize) -> Pinned {
    let mut host = sized(len);
    let table = device.alloc(crate::memory::Kind::Dram, offsets.len() * 256).unwrap();
    host.mapping = Some(Arc::new(Mapping { table, offsets }));
    host
}

/// A link whose rings live in a file the test reads and writes: the completion page first, then
/// BAR2 from the doorbell queues on.
pub(crate) fn link() -> (crate::driver::link::Link, File) {
    use crate::arch::rngd::{BAR2_LEN, CQ_LEN};
    use crate::driver::map::Mapping;
    let file = tempfile::tempfile().unwrap();
    file.set_len((CQ_LEN + BAR2_LEN) as u64).unwrap();
    let page = Mapping::new(&file, 0, CQ_LEN).unwrap();
    let bar2 = Mapping::new(&file, CQ_LEN as i64, BAR2_LEN).unwrap();
    let cluster = crate::driver::cluster::Cluster::new(bar2, page, file.try_clone().unwrap(), 0).unwrap();
    (crate::driver::link::Link::new(cluster), file)
}

/// A chip whose lock, BAR4 window and firmware PDMA mapping are temporary files, with the
/// device's memory at `window` in its clusters' address space.
pub(crate) fn chip(index: u8, window: u64) -> crate::driver::chip::Chip {
    use nix::fcntl::{Flock, FlockArg};
    // Small host writes land in BAR4 from its DRAM offset on, over the DRAM a test device has.
    let bar4 = tempfile::tempfile().unwrap();
    bar4.set_len(crate::arch::rngd::BAR4_DRAM + DRAM as u64).unwrap();
    crate::driver::chip::Chip {
        index,
        physical: index,
        peers: [0; 8],
        base: crate::arch::rngd::DEVICE_MEMORY.start + window,
        window: crate::driver::window::Window::new(bar4),
        dma: Some(crate::driver::chip::Dma {
            channels: Vec::new(),
            dmar: tempfile::tempfile().unwrap(),
            entry: 0,
        }),
        lock: Flock::lock(tempfile::tempfile().unwrap(), FlockArg::LockSharedNonblock).unwrap(),
    }
}

/// A driver with no hardware behind it, where `links` stand in for the clusters of `chips`.
pub(crate) fn driver(
    chips: Vec<crate::driver::chip::Chip>,
    pes: u8,
    links: Vec<crate::driver::link::Link>,
) -> crate::driver::Driver {
    let links = Arc::new(Mutex::new(links));
    crate::driver::Driver {
        topology: (chips.len() as u8, pes).into(),
        chips,
        transfers: DmaQueue::new(
            Transport::Fake(Fake::default()),
            Arc::clone(&links),
            Waker::noop().clone(),
        ),
        links,
        failure: Default::default(),
        tokens: Default::default(),
        owners: Default::default(),
    }
}

/// A device of `chips` with `pes` PEs each and no hardware behind it; only the test progresses
/// its work, and the device finishes every launch at once.
pub(crate) fn device(chips: &[u8], pes: u8) -> crate::Device {
    with(chips, pes, Vec::new(), DRAM)
}

/// [`device`] with one cluster, whose launches run until [`answer`] finishes them through the
/// returned file.
pub(crate) fn running(chips: &[u8], pes: u8) -> (Arc<crate::Device>, File) {
    let (link, file) = link();
    (Arc::new(with(chips, pes, vec![link], DRAM)), file)
}

/// [`running`] on a kernel driver that cannot reserve PDMA channels for the firmware.
pub(crate) fn unreserved(chips: &[u8], pes: u8) -> (Arc<crate::Device>, File) {
    let (link, file) = link();
    let mut device = with(chips, pes, vec![link], DRAM);
    let driver = Arc::get_mut(&mut device.driver).expect("a new device's only driver handle");
    driver.chips.iter_mut().for_each(|chip| chip.dma = None);
    (Arc::new(device), file)
}

/// A one-chip [`running`] device with a link for each of its clusters, over `dram` bytes.
pub(crate) fn clusters(pes: u8, dram: usize) -> (Arc<crate::Device>, Vec<File>) {
    let clusters = crate::arch::rngd::clusters((1, pes).into());
    let (links, files) = (0..clusters).map(|_| link()).unzip();
    (Arc::new(with(&[0], pes, links, dram)), files)
}

/// The cluster of a [`running`] device finishes its first `count` launches.
pub(crate) fn answer(file: &File, count: u32) {
    let launches = (1..=count).map(|id| (id, furiosa_opt_ipc::Response::Launch));
    respond(file, &launches.collect::<Vec<_>>());
}

/// The cluster behind a [`link`] file answers with `answers`, in order, after none before. The
/// completion page holds the producer index at byte 0, which publishes the entries from byte 8.
pub(crate) fn respond(file: &File, answers: &[(u32, furiosa_opt_ipc::Response)]) {
    use std::os::unix::fs::FileExt;
    let mut response = [0; 3];
    let frames = (answers.iter())
        .flat_map(|(id, answer)| {
            let words = furiosa_opt_ipc::wire::encode(answer, &mut response).unwrap();
            furiosa_opt_abi::ring::Completion::encode(*id, &response[..words])
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<u64>>();
    file.write_all_at(bytemuck::cast_slice(&frames), 8).unwrap();
    file.write_all_at(&(frames.len() as u32).to_le_bytes(), 0).unwrap();
}

fn with(chips: &[u8], pes: u8, links: Vec<crate::driver::link::Link>, dram: usize) -> crate::Device {
    let driver = driver(chips.iter().map(|&index| chip(index, 0)).collect(), pes, links);
    let ranges = crate::buffer::Ranges::new(0..dram, dram);
    crate::Device {
        worker: crate::device::Worker::new(),
        allocations: Arc::new(Allocations::new(driver.topology, ranges)),
        driver: Arc::new(driver),
        placement: crate::Placement {
            chips: chips.to_vec(),
            pes: 0..pes,
        },
    }
}
