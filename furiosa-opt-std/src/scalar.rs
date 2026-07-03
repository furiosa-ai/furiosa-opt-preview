//! Scalar types.

use furiosa_opt_macro::primitive;
use num_traits::{Num, One, Zero};
use rand::distr::StandardUniform;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Rem, Sub};

/// A trait for scalar types.
pub trait Scalar: ndarray::LinalgScalar + Debug + Clone + Copy + PartialEq + Num + Send + Sync {
    /// Number of bits per element.
    const BITS: usize;

    /// Returns the byte size for `length` elements of this scalar type.
    ///
    /// Panics if the total bit count is not a multiple of 8.
    fn size_in_bytes_from_length(length: usize) -> usize {
        crate::constraints::size_in_bytes(Self::BITS, length)
    }

    /// Returns the number of elements that fit in `bytes` bytes.
    ///
    /// Panics if the byte count does not evenly divide into whole elements.
    fn length_from_bytes(bytes: usize) -> usize {
        assert_eq!(
            (bytes * 8) % Self::BITS,
            0,
            "bytes must correspond to a whole number of elements"
        );
        (bytes * 8) / Self::BITS
    }
}

/// A byte-aligned [`Scalar`] that can be decoded from its little-endian byte representation.
///
/// Excludes sub-byte scalars like [`i4`] for which a single element cannot be addressed at a
/// byte boundary. This is what [`crate::tensor::memory::HostTensor::from_safetensors`] requires,
/// matching the set of dtypes safetensors itself can carry.
pub trait ScalarBytes: Scalar {
    /// Decodes one element from `bytes`; `bytes.len()` must equal `Self::BITS / 8`.
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

/// Implements [`Scalar`] (`BITS = size_of * 8`) and [`ScalarBytes`] (delegating to the type's inherent
/// `from_le_bytes([u8; BITS/8])`) for each byte-aligned type. `f8`/`bf16`/`f8e4m3` are defined later in
/// the module and supply their own inherent `from_le_bytes`. Sub-byte `i4` is hand-written below.
macro_rules! impl_scalar {
    ($($t:ty);* $(;)?) => {
        $(
            impl Scalar for $t {
                const BITS: usize = ::std::mem::size_of::<$t>() * 8;
            }
            impl ScalarBytes for $t {
                fn from_le_bytes(bytes: &[u8]) -> Self {
                    // Convert the slice to the inherent `from_le_bytes`'s fixed-length array, which both
                    // checks the length and disambiguates from this trait method (the delegate below
                    // is the array-taking inherent, not a recursion).
                    let arr = bytes.try_into().unwrap_or_else(|_| {
                        panic!(
                            "{} expects {} bytes, got {}",
                            stringify!($t),
                            <$t as Scalar>::BITS / 8,
                            bytes.len()
                        )
                    });
                    <$t>::from_le_bytes(arr)
                }
            }
        )*
    };
}

// The byte-aligned scalars.
impl_scalar! {
    i8;
    u8;
    i16;
    i32;
    f8;
    f8e4m3;
    bf16;
    f32;
}

/// A data type that can be either initialized or uninitialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opt<D> {
    /// Initialized value.
    Init(D),
    /// Uninitialized value.
    Uninit,
}

impl<D> Opt<D> {
    /// Maps the initialized value using the provided function, or returns uninitialized.
    #[inline]
    pub fn map<D2>(self, f: impl FnOnce(D) -> D2) -> Opt<D2> {
        match self {
            Opt::Init(val) => Opt::Init(f(val)),
            Opt::Uninit => Opt::Uninit,
        }
    }

    /// Combines two Opt values with a function. Returns Init only if both are Init.
    #[inline]
    pub fn zip_map<D2, R>(self, other: Opt<D2>, f: impl FnOnce(D, D2) -> R) -> Opt<R> {
        match (self, other) {
            (Opt::Init(a), Opt::Init(b)) => Opt::Init(f(a, b)),
            _ => Opt::Uninit,
        }
    }

    /// Returns the initialized value, or panics if uninitialized.
    pub fn unwrap(self) -> D {
        let Opt::Init(val) = self else {
            panic!("Called unwrap on an uninitialized Opt value.");
        };
        val
    }
}

// `Opt` carries `Add` / `Mul` only for the `num_traits::Zero` / `One` supertraits the fold seed needs
// (`Opt::Init(zero())`); both compose via `zip_map`.
impl<D: Add<Output = D>> Add for Opt<D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Opt<D> {
        self.zip_map(rhs, |l, r| l + r)
    }
}

impl<D: Mul<Output = D>> Mul for Opt<D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Opt<D> {
        self.zip_map(rhs, |l, r| l * r)
    }
}

impl<D: Zero> Zero for Opt<D> {
    fn zero() -> Self {
        Opt::Init(D::zero())
    }

    fn is_zero(&self) -> bool {
        let Opt::Init(val) = self else {
            return false;
        };
        val.is_zero()
    }
}

impl<D: One> One for Opt<D> {
    fn one() -> Self {
        Opt::Init(D::one())
    }
}

/// 8-bit floating point type.
#[expect(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f8(::f8::f8);

impl Zero for f8 {
    fn zero() -> Self {
        f8(::f8::f8::from(0.0))
    }

    fn is_zero(&self) -> bool {
        self.0.float().is_zero()
    }
}

impl One for f8 {
    fn one() -> Self {
        f8(::f8::f8::from(1.0))
    }
}

impl Add<Self> for f8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        f8((self.0.float() + rhs.0.float()).into())
    }
}

impl Sub<Self> for f8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        f8((self.0.float() - rhs.0.float()).into())
    }
}

impl Mul<Self> for f8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        f8((self.0.float() * rhs.0.float()).into())
    }
}

impl Div<Self> for f8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        f8((self.0.float() / rhs.0.float()).into())
    }
}

impl Rem<Self> for f8 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        f8((self.0.float() % rhs.0.float()).into())
    }
}

impl Num for f8 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(f8(::f8::f8::from(f32::from_str_radix(str, radix)?)))
    }
}

impl rand::distr::Distribution<f8> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f8 {
        let val: f32 = rng.random_range(-1.0..1.0);
        f8(::f8::f8::from(val))
    }
}

impl f8 {
    /// Decodes one element from its little-endian byte; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 1]) -> Self {
        f8(::f8::f8::from(bytes[0]))
    }
}

/// 16-bit brain floating point type.
#[primitive(bf16)]
#[expect(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct bf16(half::bf16);

impl Zero for bf16 {
    fn zero() -> Self {
        bf16(half::bf16::from_f32(0.0))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for bf16 {
    fn one() -> Self {
        bf16(half::bf16::from_f32(1.0))
    }
}

impl Add<Self> for bf16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        bf16(self.0 + rhs.0)
    }
}

impl Sub<Self> for bf16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        bf16(self.0 - rhs.0)
    }
}

impl Mul<Self> for bf16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        bf16(self.0 * rhs.0)
    }
}

impl Div<Self> for bf16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        bf16(self.0 / rhs.0)
    }
}

impl Rem<Self> for bf16 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        bf16(self.0 % rhs.0)
    }
}

impl Num for bf16 {
    type FromStrRadixErr = <half::bf16 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(bf16(half::bf16::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<bf16> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bf16 {
        let val: f32 = rng.random_range(-1.0..1.0);
        bf16(half::bf16::from_f32(val))
    }
}

impl bf16 {
    /// Decodes one element from its little-endian bytes; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 2]) -> Self {
        bf16(half::bf16::from_bits(u16::from_le_bytes(bytes)))
    }

    /// Creates `bf16` from `f32`.
    pub fn from_f32(val: f32) -> Self {
        bf16(half::bf16::from_f32(val))
    }

    /// Converts to `f32`.
    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }
}

impl From<bf16> for f32 {
    fn from(val: bf16) -> Self {
        val.to_f32()
    }
}

/// 8-bit floating point type with 4-bit exponent (E4M3).
#[primitive(f8e4m3)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f8e4m3(u8);

impl Zero for f8e4m3 {
    fn zero() -> Self {
        f8e4m3(crate::float::F8E4_ZERO)
    }

    fn is_zero(&self) -> bool {
        crate::float::f8_e4_is_zero(self.0)
    }
}

impl One for f8e4m3 {
    fn one() -> Self {
        f8e4m3(crate::float::F8E4_ONE)
    }
}

impl Add<Self> for f8e4m3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub<Self> for f8e4m3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul<Self> for f8e4m3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div<Self> for f8e4m3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Rem<Self> for f8e4m3 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl Num for f8e4m3 {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_f32(f32::from_str_radix(str, radix)?))
    }
}

impl rand::distr::Distribution<f8e4m3> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f8e4m3 {
        let val: f32 = rng.random_range(-1.0..1.0);
        f8e4m3::from_f32(val)
    }
}

impl f8e4m3 {
    /// Decodes one element from its little-endian byte; the inherent peer the `impl_scalar!`-generated
    /// [`ScalarBytes`] delegates to (matching `std`'s `from_le_bytes([u8; N])` shape).
    pub(crate) fn from_le_bytes(bytes: [u8; 1]) -> Self {
        f8e4m3(bytes[0])
    }

    /// Creates `f8e4m3` from `f32`.
    pub fn from_f32(val: f32) -> Self {
        f8e4m3(crate::float::f8_e4_from_f32(val))
    }

    /// Converts to `f32`.
    pub fn to_f32(self) -> f32 {
        crate::float::f8_e4_to_f32(self.0)
    }
}

impl From<f8e4m3> for f32 {
    fn from(val: f8e4m3) -> Self {
        val.to_f32()
    }
}

/// 4-bit signed integer type.
///
/// Stored as `i8` with sign-extension: valid range is `[-8, 7]`.
#[primitive(i4)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct i4(i8);

impl Zero for i4 {
    fn zero() -> Self {
        i4(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for i4 {
    fn one() -> Self {
        i4(1)
    }
}

impl Add<Self> for i4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 + rhs.0)
    }
}

impl Sub<Self> for i4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 - rhs.0)
    }
}

impl Mul<Self> for i4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 * rhs.0)
    }
}

impl Div<Self> for i4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 / rhs.0)
    }
}

impl Rem<Self> for i4 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self {
        Self::from_lsb(self.0 % rhs.0)
    }
}

impl Num for i4 {
    type FromStrRadixErr = <i8 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::from_lsb(i8::from_str_radix(str, radix)?))
    }
}

// `BITS = 4` is explicit (`size_of` would round the `i8` storage up to 8); `i4` is not a `ScalarBytes`
// since a single 4-bit element is not byte-addressable.
impl Scalar for i4 {
    const BITS: usize = 4;
}

impl i4 {
    fn from_lsb(n: i8) -> Self {
        i4((n << 4) >> 4)
    }

    /// Creates `i4` from `i32`.
    pub fn from_i32(val: i32) -> Self {
        Self::from_lsb(val as i8)
    }

    /// Converts to `i32`.
    pub fn to_i32(self) -> i32 {
        i32::from(self.0)
    }
}

impl From<i4> for i32 {
    fn from(val: i4) -> Self {
        val.to_i32()
    }
}
