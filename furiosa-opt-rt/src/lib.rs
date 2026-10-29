#![doc = include_str!("../README.md")]

mod arch {
    pub(crate) mod rngd;
}
mod buffer;
mod device;
mod driver;
mod function;
pub mod memory;
mod pinned;
mod placement;
pub mod registry;

pub use buffer::{Access, Blocks, Buffer, BufferSlice, BufferSliceMut, Hold, Lease};
pub use device::{Clock, ClockReading, Device, Transfer};
pub use function::{Argument, Arguments, Each, Epilogue, Function, Launch, Supplied, Supply};
pub use pinned::{Bytes, BytesMut, HostBufferSlice, HostBufferSliceMut, Pinned};
pub use placement::{Placement, ToPlacements, Topology};

/// Why the runtime refused or failed a device operation.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Another launch or transfer still uses the bytes; it is admitted once that work finishes.
    #[error("the buffer range is in use")]
    Busy,
    /// The memory was held by device work that failed, so its contents are unknown.
    #[error("the memory is unavailable after a device failure")]
    Poisoned,
    /// A launch bound more arguments than its function holds.
    #[error("more arguments than the function's signature holds")]
    Capacity,
    /// The caller asked for something the device or the program cannot do.
    #[error("invalid request: {0}")]
    Invalid(String),
    /// The hardware or its driver failed.
    #[error("device error: {0}")]
    Device(String),
    /// The device or its driver lacks a capability the request needs; the message names it.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A device task panicked; carries its message so the caller can raise it as its own panic.
    #[error("a device task {0}")]
    Panicked(String),
    /// Another process holds the PEs this device needs on chip `npu{0}`.
    #[error("npu{0} is already reserved by another process")]
    Reserved(u8),
    /// The buffer belongs to another device.
    #[error("the buffer belongs to another device")]
    ForeignBuffer,
    /// No free span of `kind` memory holds `requested` bytes.
    #[error("{kind:?} has no contiguous space for {requested} bytes")]
    OutOfMemory { kind: memory::Kind, requested: usize },
    /// SRAM residents and a launch's temporaries would overlap: the lowest resident starts at
    /// `resident`, below where the temporaries end.
    #[error("SRAM residents from {resident} overlap launch temporaries up to {temporaries}")]
    SramOverlap { resident: usize, temporaries: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn reserved(chip: u8, error: nix::errno::Errno) -> Self {
        match error {
            nix::errno::Errno::EBUSY | nix::errno::Errno::EAGAIN => Self::Reserved(chip),
            error => Self::Device(format!("reserving npu{chip}: {error}")),
        }
    }
}

/// Devices, chips and mailboxes with no hardware behind them, for every module's tests.
#[cfg(test)]
mod test {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::fs::File;
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::task::{Context, Poll, Waker};

    use crate::buffer::Allocator;
    use crate::driver::dma::{DmaQueue, Transport, fake::Fake};
    use crate::pinned::Registration;
    use crate::{Arguments, Device, Function, Pinned};

    thread_local! {
        static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// Counts its wakes.
    #[derive(Default)]
    pub(crate) struct Wakes(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Polls `wait` once with a waker that counts into `wakes`.
    pub(crate) fn poll<T>(wait: &mut (impl Future<Output = T> + Unpin), wakes: &Arc<Wakes>) -> Poll<T> {
        let waker = Waker::from(Arc::clone(wakes));
        std::pin::Pin::new(wait).poll(&mut Context::from_waker(&waker))
    }

    pub(crate) fn woken(wakes: &Wakes) -> usize {
        wakes.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A function of `capacity` operands on `device`, whose launches run no code; `bind` turns
    /// their arguments into operands.
    pub(crate) fn create<F: Future<Output = crate::Result<()>> + Send + 'static>(
        device: &Arc<Device>,
        capacity: usize,
        bind: impl for<'a> Fn(&mut crate::Supplied<'_, 'a>, &mut Arguments) -> crate::Result<F> + Send + Sync + 'static,
    ) -> Function {
        // SAFETY: stub executables do not access memory; these callbacks only exercise runtime ownership.
        let load = unsafe {
            Function::load(
                device,
                &[0],
                crate::arch::rngd::sram::WINDOW.end as usize,
                capacity,
                bind,
            )
        };
        match std::pin::pin!(load).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(function) => function.unwrap(),
            Poll::Pending => panic!("the fixture's small BAR4 write must finish synchronously"),
        }
    }

    /// [`create`] with eight operands, bound in argument order, and no epilogue.
    pub(crate) fn function(device: &Arc<Device>) -> Function {
        create(device, 8, |args, bound| {
            in_order(args, bound)?;
            Ok(std::future::ready(Ok(())))
        })
    }

    /// Binds a launch's arguments to operands 0, 1, ... in the order they come.
    pub(crate) fn in_order(args: &mut crate::Supplied<'_, '_>, bound: &mut Arguments) -> crate::Result<()> {
        let mut index = 0;
        args(&mut |argument| {
            bound.bind(index, argument)?;
            index += 1;
            Ok(())
        })
    }

    /// Bytes of device DRAM that [`device`] and its siblings give a test device.
    const DRAM: usize = 4096;

    struct Counting;

    #[global_allocator]
    static ALLOCATOR: Counting = Counting;

    // SAFETY: allocation delegates to System; the default zeroing and reallocation call `alloc`.
    unsafe impl GlobalAlloc for Counting {
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
    pub(crate) fn allocator(dram: Range<usize>, shared_end: usize) -> Arc<Allocator> {
        let ranges = crate::buffer::Ranges::new(dram, shared_end);
        Arc::new(Allocator::over((1, 8).into(), ranges))
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
        host.set_registration(Some(Arc::new(Registration { tables: table, offsets })));
        host
    }

    /// A mailbox whose rings live in a file the test reads and writes: the completion page first, then
    /// BAR2 from the doorbell cluster on.
    pub(crate) fn mailbox() -> (crate::driver::mailbox::Mailbox, File) {
        use crate::driver::kernel::map::Mapping;
        use crate::driver::kernel::{BAR2_LEN, CQ_LEN};
        let file = tempfile::tempfile().unwrap();
        file.set_len((CQ_LEN + BAR2_LEN) as u64).unwrap();
        let page = Mapping::new(&file, 0, CQ_LEN).unwrap();
        let bar2 = Mapping::new(&file, CQ_LEN as i64, BAR2_LEN).unwrap();
        let link = crate::driver::link::Link::new(bar2, page, 0).unwrap();
        let cluster = crate::driver::cluster::Cluster::new(file.try_clone().unwrap());
        (
            crate::driver::mailbox::Mailbox::new(cluster, link, furiosa_opt_ipc::DRIVER_DEADLINE),
            file,
        )
    }

    /// A chip whose lock, BAR4 window and firmware PDMA mapping are temporary files, with the
    /// device's memory at `window` in its clusters' address space.
    pub(crate) fn chip(index: u8, window: u64) -> crate::driver::chip::Chip {
        use nix::fcntl::{Flock, FlockArg};
        // Small host writes land in BAR4 from its DRAM offset on; the file is sparse, so it covers
        // the DRAM of any test device.
        let bar4 = tempfile::tempfile().unwrap();
        bar4.set_len(crate::driver::kernel::BAR4_DRAM + (1 << 30)).unwrap();
        crate::driver::chip::Chip {
            index,
            peers: [0; 8],
            base: crate::arch::rngd::DEVICE_MEMORY.start + window,
            window: crate::driver::kernel::window::Window::new(bar4),
            lock: Flock::lock(tempfile::tempfile().unwrap(), FlockArg::LockSharedNonblock).unwrap(),
        }
    }

    /// A driver with no hardware behind it, where `mailboxes` stand in for the clusters of `chips`.
    pub(crate) fn driver(
        chips: Vec<crate::driver::chip::Chip>,
        pes: u8,
        mailboxes: Vec<crate::driver::mailbox::Mailbox>,
    ) -> crate::driver::Driver {
        let mailboxes = Arc::new(Mutex::new(mailboxes));
        let queue = DmaQueue::new(
            Transport::Fake(Fake::default()),
            Arc::clone(&mailboxes),
            Waker::noop().clone(),
        );
        let dma = |_| crate::driver::chip::Dma {
            channels: Vec::new(),
            dmar: tempfile::tempfile().unwrap(),
            entry: 0,
        };
        crate::driver::Driver {
            topology: (chips.len() as u8, pes).into(),
            pdma: Some(crate::driver::chip::Pdma {
                chips: chips.iter().map(dma).collect(),
            }),
            chips,
            poison: Arc::clone(queue.poison()),
            dma: queue,
            mailboxes,
            allocatable: crate::buffer::Ranges::new(0..DRAM, DRAM),
            patience: furiosa_opt_ipc::DRIVER_DEADLINE,
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
        let (mailbox, file) = mailbox();
        (Arc::new(with(chips, pes, vec![mailbox], DRAM)), file)
    }

    /// [`running`] on a kernel driver that cannot reserve PDMA channels for the firmware.
    pub(crate) fn unreserved(chips: &[u8], pes: u8) -> (Arc<crate::Device>, File) {
        let (mailbox, file) = mailbox();
        let mut device = with(chips, pes, vec![mailbox], DRAM);
        let driver = Arc::get_mut(&mut device.driver).expect("a new device's only driver handle");
        driver.pdma = None;
        (Arc::new(device), file)
    }

    /// A one-chip [`running`] device with a mailbox for each of its clusters, over `dram` bytes.
    pub(crate) fn clusters(pes: u8, dram: usize) -> (Arc<crate::Device>, Vec<File>) {
        let clusters = crate::Topology { chips: 1, pes }.clusters();
        let (mailboxes, files) = (0..clusters).map(|_| mailbox()).unzip();
        (Arc::new(with(&[0], pes, mailboxes, dram)), files)
    }

    /// The cluster of a [`running`] device finishes its first `count` launches.
    pub(crate) fn answer(file: &File, count: u32) {
        let launches = (1..=count).map(|id| (id, furiosa_opt_ipc::Answer(Ok(furiosa_opt_ipc::Records::new()))));
        respond(file, &launches.collect::<Vec<_>>());
    }

    /// The cluster behind a [`mailbox`] file answers with `answers`, in order, after none before. The
    /// completion page holds the producer index at byte 0, which publishes the entries from byte 8.
    pub(crate) fn respond<T: furiosa_opt_ipc::Payload>(file: &File, answers: &[(u32, furiosa_opt_ipc::Answer<T>)]) {
        use std::os::unix::fs::FileExt;
        let mut response = vec![0; furiosa_opt_ipc::MAX_RESPONSE_WORDS];
        let frames = (answers.iter())
            .flat_map(|(id, answer)| {
                let words = furiosa_opt_ipc::frame::encode(answer, &mut response).unwrap();
                furiosa_opt_abi::ring::Completion::encode(*id, &response[..words])
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<u64>>();
        file.write_all_at(bytemuck::cast_slice(&frames), 8).unwrap();
        file.write_all_at(&(frames.len() as u32).to_le_bytes(), 0).unwrap();
    }

    pub(crate) fn with(
        chips: &[u8],
        pes: u8,
        mailboxes: Vec<crate::driver::mailbox::Mailbox>,
        dram: usize,
    ) -> crate::Device {
        let mut driver = driver(chips.iter().map(|&index| chip(index, 0)).collect(), pes, mailboxes);
        driver.allocatable = crate::buffer::Ranges::new(0..dram, dram);
        let allocator = Arc::new(Allocator::over(driver.topology, driver.allocatable.clone()));
        let driver = Arc::new(driver);
        crate::Device {
            worker: crate::device::Worker::new(),
            launches: Arc::default(),
            transfers: Arc::new(crate::device::Transfers::new(Waker::noop().clone())),
            allocator,
            driver,
            placement: crate::Placement {
                chips: chips.to_vec(),
                pes: 0..pes,
            },
        }
    }
}
