//! Tensors held in host memory.

use rand::Rng;
use rand::distr::StandardUniform;

use furiosa_mapping::*;
use furiosa_opt_macro::primitive;

use crate::Error;
use crate::backend::Backend;
use crate::runtime::{CurrentBackend, HostBuffer};
use crate::scalar::{MaterializableScalar, ScalarBytes};
use crate::tensor::Tensor;

/// Tensor stored in host memory.
///
/// `D: MaterializableScalar`, not just `Scalar`: a `HostTensor` is exactly the "host... carrier" the
/// bound's own doc names, so a non-materializable staging type (`i5`/`i9`) can never even be
/// CONSTRUCTED as one -- rejected at the type's own definition, not left to be caught only if some
/// particular method (`into_vec`, etc.) happens to touch the unsound whole-buffer path. `i5`/`i9`'s
/// entire legitimate lifetime is between `fetch_zero_point_sub` and `contract_outer`, both
/// engine-internal (`TuTensor`/bare `Tensor`), never through this public host-facing wrapper.
#[primitive(HostTensor)]
#[derive(Debug, Clone)]
pub struct HostTensor<D: MaterializableScalar, Element: M, B: Backend = CurrentBackend> {
    inner: Tensor<D, Element, B>,
}

impl<D: MaterializableScalar, Element: M, B: Backend> From<Tensor<D, Element, B>> for HostTensor<D, Element, B> {
    fn from(inner: Tensor<D, Element, B>) -> Self {
        Self { inner }
    }
}

/// A host buffer's bytes read as `D` values laid out as `Element`, once its length is the
/// tensor's; [`HostTensor::into_buffer`] gives the buffer back.
impl<D: MaterializableScalar, Element: M, B: Backend> TryFrom<HostBuffer> for HostTensor<D, Element, B> {
    type Error = Error;

    fn try_from(buffer: HostBuffer) -> Result<Self, Error> {
        if buffer.len() != D::size_in_bytes_from_length(Element::SIZE) {
            return Err(Error::Invalid("the host buffer's length is not the tensor's"));
        }
        Ok(Tensor::from_inner(buffer.into()).into())
    }
}

impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Logical shape (mapping) of this tensor.
    pub type Mapping = Element;

    pub(crate) fn inner(&self) -> &Tensor<D, Element, B> {
        &self.inner
    }

    pub(crate) fn storage(&self) -> &B::Storage<D> {
        &self.inner.inner
    }

    pub(crate) fn storage_mut(&mut self) -> &mut B::Storage<D> {
        &mut self.inner.inner
    }

    /// The same tensor in page-locked host memory, so a transfer reads or writes it directly and
    /// asynchronously. Already pinned memory stays.
    ///
    /// # Errors
    /// [`Error::Device`] when the system cannot page-lock the memory.
    pub fn pinned(self) -> Result<Self, Error> {
        Ok(Tensor::from_inner(B::pin(self.inner.inner)?).into())
    }

    /// Creates a tensor from an initialized buffer. Panics if the buffer length does not match the
    /// mapping size.
    pub fn from_vec(data: impl IntoIterator<Item = D>) -> Self {
        Tensor::from_vec(data).into()
    }

    /// Creates a tensor from a pre-packed device byte image ([`crate::scalar::Scalar::to_buf`]'s inverse), stored as-is.
    /// Contrast [`Self::from_vec`], which packs logical values; pre-packed fp4 weights come through here
    /// to avoid a decode + re-pack round-trip. Panics on a byte-length mismatch.
    pub fn from_buf(buf: Vec<u8>) -> Self {
        Tensor::from_buf(buf).into()
    }

    /// The host buffer this tensor's bytes live in, as the device holds them.
    pub fn into_buffer(self) -> HostBuffer {
        self.inner.inner.into()
    }

    /// Consumes self and returns the inner tensor.
    pub fn into_inner(self) -> Tensor<D, Self::Mapping, B> {
        self.inner
    }

    /// Returns the tensor data as a flat `Vec<D>`, consuming the tensor.
    pub fn into_vec(self) -> Vec<D> {
        self.inner.into_vec()
    }
}

/// Host-side `HostTensor` constructors. Bound to `Backend`; the value-iterating methods (`zero`,
/// `rand`) and `from_vec` / `from_safetensors` all bottom out in `Tensor::from_vec`, which
/// `BufStorage` implements as a real `Vec<D>` fill, so those work on Npu / Cpu host-side
/// staging too.
impl<D: MaterializableScalar, Element: M, B: Backend> HostTensor<D, Element, B> {
    /// Creates a tensor filled with zeros.
    pub fn zero() -> Self
    where
        D: num_traits::Zero,
    {
        Tensor::splat(D::zero()).into()
    }

    /// Creates a tensor filled with random values.
    #[primitive(HostTensor::rand)]
    pub fn rand(rng: &mut impl Rng) -> Self
    where
        StandardUniform: rand::distr::Distribution<D>,
    {
        Tensor::rand(rng).into()
    }

    /// Creates a tensor from a `safetensors` tensor view.
    ///
    /// The view's per-axis shape must match `Element`'s pair-flattened size list (e.g.
    /// `m![H, X]` expects safetensors shape `[H.size, X.size]`) and its bytes are decoded as
    /// little-endian `D` values — LE is mandated by the safetensors format spec, not our
    /// choice. Returns [`safetensors::SafeTensorError::TensorInvalidInfo`] on any mismatch.
    pub fn from_safetensors(view: &safetensors::tensor::TensorView<'_>) -> Result<Self, safetensors::SafeTensorError>
    where
        D: ScalarBytes,
    {
        fn flat_shape(mapping: &Mapping, out: &mut Vec<usize>) {
            match mapping {
                Mapping::Pair { left, right } => {
                    flat_shape(left, out);
                    flat_shape(right, out);
                }
                _ => out.push(mapping.size()),
            }
        }
        let mut expected_shape = Vec::new();
        flat_shape(&Element::to_value(), &mut expected_shape);
        if view.shape() != expected_shape.as_slice() {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        let stride = D::BITS / 8;
        if view.data().len() != Element::SIZE * stride {
            return Err(safetensors::SafeTensorError::TensorInvalidInfo);
        }
        // The view's LE bytes ARE the packed device image for a byte-multiple `D`, so store them directly
        // through `from_buf` rather than decoding each element and re-packing.
        Ok(Tensor::from_buf(view.data().to_vec()).into())
    }
}
