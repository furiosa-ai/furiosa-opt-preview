//! The Vector Engine's Filter stage: which output layouts each filter can emit from a stream.

mod bitmap;
mod tag;

use furiosa_mapping::{Mapping, MappingExt};
use furiosa_opt_common_ir::CollectBitmap;

pub use self::bitmap::config_filter_by_bitmap;
pub use self::tag::config_filter_by_tag;
use crate::verify::ONE_FLIT_ELEMENTS;

/// The layouts a filter reads and emits, per slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInput {
    pub in_time: Mapping,
    pub in_packet: Mapping,
    pub out_time: Mapping,
    pub out_packet: Mapping,
}

/// Why a filter's input or output layout is not realizable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FilterError {
    #[error("the bitmap keeps no packet position")]
    EmptyBitmap,
    #[error(
        "a tag filter takes no padded input, since padding passes the guard like an element; got {}",
        layouts.input_layout()
    )]
    TagFilterInputPadding { layouts: Box<FilterInput> },
    #[error(
        "a filter reads and emits packets of {ONE_FLIT_ELEMENTS} elements; got Packet = {} of {} and Packet2 = {} of {}",
        layouts.in_packet,
        layouts.in_packet.size(),
        layouts.out_packet,
        layouts.out_packet.size()
    )]
    PacketSize { layouts: Box<FilterInput> },
    #[error(
        "a tag filter fills its output from the front, so the output may pad only after its last element; got {}, which pads between its elements",
        layouts.output_layout()
    )]
    InteriorOutputPadding { layouts: Box<FilterInput> },
    #[error(
        "a tag filter emits at most the {} packets a slice reads, but Time2 holds {}; got {} and {}",
        layouts.in_time.size(),
        layouts.out_time.size(),
        layouts.input_layout(),
        layouts.output_layout()
    )]
    OutputCapacity { layouts: Box<FilterInput> },
    #[error(
        "a tag filter puts each slice's k-th survivor at output position k, an index the input does not have, so \
         the output names it with new axes; got {}, which reuses an axis of {}",
        layouts.output_layout(),
        layouts.input_layout()
    )]
    TagOutputNamesInputAxes { layouts: Box<FilterInput> },
    #[error(
        "a tag filter on padded chips or clusters ({clusters}) must hold every packet a slice reads, since the \
         padding's undefined values may pass the guard and the commit cannot skip a chip or a cluster, so a slice \
         there could emit past its output and stop the device; got Time = {} and Time2 = {}",
        layouts.in_time,
        layouts.out_time
    )]
    TagFilterPaddedCluster {
        clusters: Mapping,
        layouts: Box<FilterInput>,
    },
    #[error(
        "the filter keeps {kept_elements} elements per slice, which fill {} packets, but Time2 holds {}; got {}",
        kept_elements.div_ceil(ONE_FLIT_ELEMENTS),
        layouts.out_time.size(),
        layouts.output_layout()
    )]
    OutputVolume {
        kept_elements: usize,
        layouts: Box<FilterInput>,
    },
    /// A bitmap filter's output that is not the input axes' spelling of the kept positions, nor new
    /// axes for them.
    #[error("{}", layouts.mismatches_input_axes_message(*bitmap, spelling))]
    BitmapOutputMismatchesInputAxes {
        bitmap: CollectBitmap,
        spelling: Mapping,
        layouts: Box<FilterInput>,
    },
    /// A bitmap filter's output that does not name, with new axes, kept positions no input axis spells.
    #[error("{}", layouts.needs_new_axes_message(*bitmap))]
    BitmapOutputNeedsNewAxes {
        bitmap: CollectBitmap,
        layouts: Box<FilterInput>,
    },
    /// A bitmap filter that keeps positions no input axis spells, over an input whose padding rules
    /// out new axes.
    #[error("{}", layouts.new_axes_over_padded_input_message(*bitmap))]
    BitmapNewAxesOverPaddedInput {
        bitmap: CollectBitmap,
        layouts: Box<FilterInput>,
    },
}

impl FilterInput {
    /// The input as one mapping, `Time` outside `Packet`.
    pub(super) fn input(&self) -> Mapping {
        self.in_time.clone().pair(self.in_packet.clone())
    }

    /// The output as one mapping, `Time` outside `Packet`.
    pub(super) fn output(&self) -> Mapping {
        self.out_time.clone().pair(self.out_packet.clone())
    }

    /// Whether `mapping` names no axis of the input, so it gives the filter's output an index of
    /// its own.
    fn names_only_new_axes(&self, mapping: &Mapping) -> bool {
        let input_axes = self.input().normalize().axes();
        (mapping.axes().iter()).all(|axis| input_axes.iter().all(|input| input.symbol != axis.symbol))
    }

    /// Checks that the filter reads and emits packets of one flit.
    fn check_packets(&self) -> Result<(), FilterError> {
        let (input, output) = (self.in_packet.size(), self.out_packet.size());
        if input != ONE_FLIT_ELEMENTS || output != ONE_FLIT_ELEMENTS {
            return Err(FilterError::PacketSize {
                layouts: Box::new(self.clone()),
            });
        }
        Ok(())
    }

    /// The input as a diagnostic spells it.
    fn input_layout(&self) -> String {
        format!("Time = {}, Packet = {}", self.in_time, self.in_packet)
    }

    /// The output as a diagnostic spells it.
    fn output_layout(&self) -> String {
        format!("Time2 = {}, Packet2 = {}", self.out_time, self.out_packet)
    }
}
