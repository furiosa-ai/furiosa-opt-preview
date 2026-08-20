//! Vector-engine verifications, mirroring `furiosa-opt-std/src/engine/vector`.

mod operand;
mod tensor;

pub use operand::{UnreadableCause, VrfOperandError, config_vrf_operand};
pub use tensor::{
    VectorError, config_reduce_label, config_vector_intra_slice_unzip, config_vector_narrow_split,
    config_vector_narrow_trim, config_vector_widen_concat, config_vector_widen_pad,
};
