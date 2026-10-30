//! Device-function execution runtime: a device, its memory, and the launches and transfers that
//! use it.

mod bind;
mod buffer;
mod launch;
mod transfer;

pub(crate) use bind::launch_memory;
pub use bind::{Access, Bind, DeviceOutput, DeviceSend};
pub(crate) use buffer::Region;
pub use buffer::{
    Chips, HbmBuffer, HbmBufferSlice, HbmBufferSliceMut, HostBuffer, HostBufferSlice, HostBufferSliceMut, Memory,
};
pub use launch::{Kernel, Launch, Launched, Loadable, Prepared, Prepend, Returned, Returns, Submitted, Work, launch};

use std::fmt::{self, Debug};

use cfg_if::cfg_if;
use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::{Backend, Runtime};
use crate::context::{Dma, DmaContext, Tu, TuContext};

use crate::backend::{Cpu, Npu};

cfg_if! {
    if #[cfg(backend = "npu")] {
        /// Backend selected for an NPU build.
        pub type CurrentBackend = Npu;
    } else {
        /// Backend selected for a host build.
        pub type CurrentBackend = Cpu;
    }
}

/// The chips a device function runs on, declared by `#[device(chip, pe)]`.
pub use furiosa_opt_rt::Topology;

/// Physical chips in rank order and the PEs used on each.
pub use furiosa_opt_rt::Placement;

/// Host bytes the CPU reads, as [`HostBufferSlice::bytes`] gives them; their borrow keeps device
/// writes off them.
pub use furiosa_opt_rt::Bytes;

/// Host bytes the CPU writes, as [`HostBufferSliceMut::into_bytes`] gives them; their borrow keeps
/// device accesses off them.
pub use furiosa_opt_rt::BytesMut;

/// The placements `Device::new` tries, in order.
pub trait ToPlacements<B: Backend = CurrentBackend> {
    /// Resolves candidates using the backend's placement policy.
    fn to_placements(&self) -> Result<Vec<Placement>, Error>;
}

impl<B: Backend> ToPlacements<B> for Placement {
    fn to_placements(&self) -> Result<Vec<Placement>, Error> {
        Ok(vec![self.clone()])
    }
}

impl ToPlacements<Npu> for Topology {
    fn to_placements(&self) -> Result<Vec<Placement>, Error> {
        Ok(furiosa_opt_rt::ToPlacements::to_placements(self)?)
    }
}

impl ToPlacements<Cpu> for Topology {
    fn to_placements(&self) -> Result<Vec<Placement>, Error> {
        Ok(vec![Placement {
            chips: (0..self.chips).collect(),
            pes: 0..self.pes,
        }])
    }
}

/// A kind of device memory, as `DeviceError::OutOfMemory` names the kind it ran out of.
pub use furiosa_opt_rt::memory::Kind;

/// How much of a device's memory is free, as [`Memory::usage`] reports it.
pub use furiosa_opt_rt::memory::Usage;

/// Why the runtime refused or failed a device operation, carried by [`Error::Device`].
pub use furiosa_opt_rt::Error as DeviceError;

/// Device context.
#[primitive(Device)]
pub struct Device<B: Runtime = CurrentBackend> {
    /// Tensor unit for the main context.
    pub main: TuContext<{ Tu::Main }>,
    /// Tensor unit for the sub context.
    pub sub: TuContext<{ Tu::Sub }>,
    /// Tensor DMA context.
    pub tdma: DmaContext<{ Dma::Tensor }, B>,
    /// PCIe DMA context.
    pub pdma: DmaContext<{ Dma::Pcie }, B>,
    // The kernel translator reads `main`, `sub`, `tdma`, `pdma` as MIR fields 0..=3.
}

impl<B: Backend> Debug for Device<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Device").finish_non_exhaustive()
    }
}

impl<B: Backend> Device<B> {
    /// Opens the first free placement of a `Topology`, or exactly the supplied `Placement`.
    pub fn new(placements: impl ToPlacements<B>) -> Result<Self, Error> {
        let device = B::open(placements.to_placements()?)?;
        Ok(Self {
            main: TuContext::<{ Tu::Main }>::on(),
            sub: TuContext::<{ Tu::Sub }>::on(),
            tdma: DmaContext::<{ Dma::Tensor }, B>::on(device.clone()),
            pdma: DmaContext::<{ Dma::Pcie }, B>::on(device),
        })
    }
}

impl<B: Backend> Device<B> {
    /// Prepares a launch of `function` on this device, with `args` as the parameters after it:
    /// `device.launch(f, (&a, &b))` is `launch(f, (&mut device, &a, &b))`.
    pub fn launch<'c, F, Args>(&'c mut self, function: F, args: Args) -> Prepared<Launch<F, Args::Output>>
    where
        Args: Prepend<&'c mut Self>,
        Args::Output: DeviceSend,
        F: DeviceFn<Args::Output>,
    {
        launch(function, args.prepend(self))
    }

    /// The chips and PEs this device holds.
    pub fn topology(&self) -> Topology {
        B::topology(self.pdma.device())
    }

    /// The placement this device opened.
    pub fn placement(&self) -> &Placement {
        B::placement(self.pdma.device())
    }

    /// Resolves once every launch and transfer submitted on this device before this call ended,
    /// whether or not anyone awaits them; the device's failure once it failed.
    pub fn synchronize(&self) -> impl Future<Output = Result<(), Error>> + Send + use<B> {
        B::synchronize(self.pdma.device())
    }

    /// The memory this device's buffers take.
    pub fn memory(&self) -> Memory<'_, B> {
        Memory {
            device: self.pdma.device(),
        }
    }
}

/// The tensor DMA engine, for an operation generic over `const DMA: Dma`: it bounds
/// `Device: AsMut<DmaContext<DMA>>` and transfers on `device.as_mut()`.
impl<B: Backend> AsMut<DmaContext<{ Dma::Tensor }, B>> for Device<B> {
    fn as_mut(&mut self) -> &mut DmaContext<{ Dma::Tensor }, B> {
        &mut self.tdma
    }
}

/// The PCIe DMA engine, for an operation generic over `const DMA: Dma`.
impl<B: Backend> AsMut<DmaContext<{ Dma::Pcie }, B>> for Device<B> {
    fn as_mut(&mut self) -> &mut DmaContext<{ Dma::Pcie }, B> {
        &mut self.pdma
    }
}

/// Executes a `#[device]` function on the selected backend with device-representable arguments
/// and returns. CPU builds execute its Rust body; NPU builds load its compiled registry entry.
// The kernel translator finds launches through this trait's path,
// `furiosa_opt_std::runtime::DeviceFn` (npu-visa-translate `device_marker`).
pub trait DeviceFn<Args: DeviceSend> {
    /// The topology of this argument specialization.
    const TOPOLOGY: Topology;
    /// The function's return value.
    type Output: DeviceOutput;
    /// Submits work whose returns go where `returns` says.
    fn launch<R: Returns<Self::Output>>(self, args: Args, returns: R) -> Submitted<Launched<R::Value>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Dma, DmaContext};
    #[cfg(not(backend = "npu"))]
    use crate::prelude::*;

    /// CPU topology resolution retains ascending logical chips and the requested PE count.
    #[test]
    fn reports_topology_placement() {
        let device = Device::<Cpu>::new(Topology { chips: 2, pes: 8 }).unwrap();
        assert_eq!(
            device.placement(),
            &Placement {
                chips: vec![0, 1],
                pes: 0..8
            }
        );
    }

    /// An explicit CPU placement keeps its identity while opening only its shape.
    #[test]
    fn preserves_explicit_placement() {
        let placement = Placement {
            chips: vec![3],
            pes: 0..4,
        };
        let device = Device::<Cpu>::new(placement.clone()).unwrap();
        assert_eq!(device.topology(), Topology { chips: 1, pes: 4 });
        assert_eq!(device.placement(), &placement);
    }

    /// A caller's resolver can supply no candidates, which is an invalid open request.
    #[test]
    fn rejects_empty_placements() {
        struct Empty;
        impl ToPlacements<Cpu> for Empty {
            fn to_placements(&self) -> Result<Vec<Placement>, Error> {
                Ok(Vec::new())
            }
        }
        assert!(matches!(Device::<Cpu>::new(Empty), Err(Error::Invalid(_))));
    }

    /// A CPU device opens only a topology a device can have: one to eight chips of one to eight PEs.
    #[test]
    fn rejects_unsupported_topologies() {
        for (chips, pes) in [(0, 8), (9, 8), (17, 8), (255, 8), (1, 0), (1, 9)] {
            let opened = Device::<Cpu>::new(Topology { chips, pes });
            assert!(
                matches!(opened, Err(Error::Invalid(_))),
                "{chips} chips, {pes} PEs: {opened:?}"
            );
        }
        assert!(Device::<Cpu>::new(Topology { chips: 8, pes: 8 }).is_ok());
    }

    #[test]
    fn defaults_to_current_backend() {
        let topology = Topology { chips: 1, pes: 8 };
        let _: Device = Device::new(topology).unwrap();
        let context: Device = Device::new(topology).unwrap();
        let _: &DmaContext<{ Dma::Pcie }> = &context.pdma;
    }

    #[test]
    fn kernel_visible_fields_lead() {
        let (_, body) = include_str!("mod.rs").split_once("pub struct Device<").unwrap();
        let (body, _) = body.split_once("\n}").unwrap();
        let fields: Vec<_> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("pub ")?.split_once(':'))
            .map(|(name, _)| name)
            .collect();
        assert_eq!(fields, ["main", "sub", "tdma", "pdma"]);
    }

    #[tokio::test]
    #[cfg(not(backend = "npu"))]
    async fn pcie_dma_round_trips() {
        axes![A = 4];

        let mut context = Device::new(Topology { chips: 1, pes: 8 }).expect("CPU context");
        let values = vec![3, 1, 4, 1];

        let hbm = HostTensor::<i32, m![A]>::from_vec(values.clone())
            .to_hbm::<m![1], m![A]>(&mut context.pdma)
            .await
            .unwrap();
        let host = hbm.to_host::<m![A]>(&mut context.pdma).await.unwrap();

        assert_eq!(host.into_vec(), values);
    }
}
