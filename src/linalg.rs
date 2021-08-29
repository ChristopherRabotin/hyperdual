use na::{SVector, Scalar};

use {Float, Hyperdual, Zero};

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
