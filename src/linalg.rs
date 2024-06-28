use na::allocator::Allocator;
use na::{DefaultAllocator, DimName, OVector, SVector, Scalar};

use {Float, Hyperdual, OHyperdual, Zero};

/// Computes the norm of a vector of Hyperdual.
pub fn norm<T: Scalar + Float, const M: usize, const N: usize>(v: &SVector<Hyperdual<T, N>, M>) -> Hyperdual<T, N>
where
    Hyperdual<T, N>: Float,
{
    let mut val = Hyperdual::<T, N>::zero();

    for i in 0..M {
        val += v[i].powi(2);
    }

    val.sqrt()
}

/// Computes the norm of a vector of Hyperdual.
pub fn norm_owned<T: Scalar + Float, M: DimName, N: DimName>(v: &OVector<OHyperdual<T, N>, M>) -> OHyperdual<T, N>
where
    OHyperdual<T, N>: Float,
    DefaultAllocator: Allocator<M> + Allocator<N>,
    <DefaultAllocator as Allocator<N>>::Buffer<T>: Copy,
{
    let mut val = OHyperdual::<T, N>::zero();

    for i in 0..M::dim() {
        val += v[i].powi(2);
    }

    val.sqrt()
}
