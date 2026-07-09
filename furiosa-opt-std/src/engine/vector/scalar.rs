//! VeScalar trait for Vector Engine scalar types.

use std::any::Any;

use crate::scalar::Scalar;

/// The kind of scalar type supported by Vector Engine.
#[derive(Debug, PartialEq, Eq)]
pub enum VeScalarKind {
    /// 32-bit floating point.
    F32,
    /// 32-bit integer.
    I32,
}

/// Marker trait for scalar types supported by Vector Engine.
/// Only i32 and f32 are supported.
///
/// This trait is sealed and cannot be implemented outside this module.
pub trait VeScalar: Scalar + Any {
    /// The kind of this scalar type.
    const KIND: VeScalarKind;

    /// Reinterprets `self` as `D2` when the two concrete scalar types match (equivalently
    /// `Self::KIND == D2::KIND`, since only i32/f32 exist), else `None`. Safe: a checked [`Any`]
    /// downcast, no `transmute`. Used by the cross-type stash read to relabel `Tensor<StashD>` as
    /// the pipeline's `Tensor<D>` once the KIND precondition holds.
    fn reinterpret<D2: VeScalar>(self) -> Option<D2> {
        (&self as &dyn Any).downcast_ref::<D2>().copied()
    }
}

impl VeScalar for i32 {
    const KIND: VeScalarKind = VeScalarKind::I32;
}

impl VeScalar for f32 {
    const KIND: VeScalarKind = VeScalarKind::F32;
}
