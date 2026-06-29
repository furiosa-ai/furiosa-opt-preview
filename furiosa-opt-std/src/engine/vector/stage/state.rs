use std::collections::HashSet;
use std::marker::PhantomData;

use furiosa_mapping::M;

use crate::{
    engine::vector::{alu::RngdAlu, scalar::VeScalar},
    tensor::Tensor,
    tensor_state::{HasTensor, NoTensor, TensorState},
};

/// VE state that tracks stash and ALU usage.
///
/// The type parameter `D` ties the stash's scalar type to the pipeline's current scalar type,
/// ensuring at compile time that stash reads match the pipeline's `D`.
///
/// - `VeState<D, NoTensor>` — stash is empty.
/// - `VeState<D, HasTensor<D, Mapping>>` — stash holds a `Tensor<D, Mapping>`.
#[derive(Debug)]
pub struct VeState<D: VeScalar, Stash: TensorState<D>> {
    /// Stash state — either [`NoTensor`] or [`HasTensor`] holding the actual tensor.
    pub(crate) stash: Stash,
    /// Set of ALUs that have been used.
    pub(crate) used_alus: HashSet<RngdAlu>,
    /// Marker for the scalar type `D`.
    _marker: PhantomData<D>,
}

impl<D: VeScalar, Stash: TensorState<D>> VeState<D, Stash> {
    /// Checks if ALU is available and marks it as used.
    /// Panics if ALU is already in use.
    pub(crate) fn use_alu(&mut self, alu: RngdAlu) {
        assert!(!self.used_alus.contains(&alu), "{alu:?} is already in use");
        self.used_alus.insert(alu);
    }

    /// Clones the stash data, transposing to target mapping if needed.
    ///
    /// TODO: This should checked at compile-time, or replaced with clone_stash_as.
    ///
    /// # Panics
    /// Panics if the target type D2 does not match the stashed type D.
    pub(crate) fn force_clone_stash_as<D2: VeScalar, TargetMapping: M>(&self) -> Option<Tensor<D2, TargetMapping>> {
        assert!(
            D::KIND == D2::KIND,
            "stash type mismatch: stashed as {:?}, requested as {:?}",
            D::KIND,
            D2::KIND
        );

        let cloned = self.stash.clone_tensor_as::<TargetMapping>();

        // SAFETY: Runtime assert above ensures D and D2 are the same concrete type
        // (both i32 or both f32), so Tensor<D, _> and Tensor<D2, _> have identical layout.
        cloned.map(|t| unsafe {
            // `transmute` is rejected because the compiler can't see that D and D2 have the same
            // size; `transmute_copy` + `forget` achieves the same move without that check.
            let converted = std::mem::transmute_copy(&t);
            std::mem::forget(t);
            converted
        })
    }
}

impl<D: VeScalar> VeState<D, NoTensor> {
    /// Creates a new empty VeState.
    pub(crate) fn new() -> Self {
        Self {
            stash: NoTensor,
            used_alus: HashSet::new(),
            _marker: PhantomData,
        }
    }

    /// Stores the tensor data in the stash, returning a new VeState with the stash's mapping.
    /// Consumes self to change the `TensorState` parameter.
    pub(crate) fn stash<NewMapping: M>(self, data: &Tensor<D, NewMapping>) -> VeState<D, HasTensor<D, NewMapping>> {
        VeState {
            stash: HasTensor::new(data.clone()),
            used_alus: self.used_alus,
            _marker: PhantomData,
        }
    }

    /// Consumes another VeState and merges its ALU usage into this one.
    /// Used when combining two groups in binary operations.
    /// Since both groups share the same ALU state (global resource),
    /// this simply performs a union of used ALUs.
    pub(crate) fn merge(&mut self, other: VeState<D, NoTensor>) {
        self.used_alus.extend(other.used_alus);
    }

    /// Converts VeState to different scalar type parameters, preserving ALU tracking.
    pub(crate) fn retype<NewD: VeScalar>(self) -> VeState<NewD, NoTensor> {
        VeState {
            stash: NoTensor,
            used_alus: self.used_alus,
            _marker: PhantomData,
        }
    }
}
