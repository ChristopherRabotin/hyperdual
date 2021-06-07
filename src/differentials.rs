use super::Float;
use na::{SMatrix, SVector};
use {Dual, DualN, One, Scalar, Zero};

/// Evaluates the function using dual numbers to get the partial derivative at the input point
#[inline]
pub fn differentiate<T: Copy + Scalar + One, F>(x: T, f: F) -> T
where
    F: Fn(Dual<T>) -> Dual<T>,
{
    f(Dual::new(x, T::one())).dual()
}

// Extracts Jacobian matrix and function value from a vector of dual numbers
#[inline]
pub fn extract_jacobian_and_result<T: Scalar + Zero + Float, const DIM_IN: usize, const DIM_OUT: usize, const DIM_HYPER: usize>(
    fx_dual: &SVector<DualN<T, DIM_HYPER>, DIM_OUT>,
) -> (SVector<T, DIM_OUT>, SMatrix<T, DIM_OUT, DIM_IN>) {
    let fx = super::vector_from_hyperspace(&fx_dual);
    let mut grad = SMatrix::<T, DIM_OUT, DIM_IN>::zeros();

    for i in 0..DIM_OUT {
        for j in 0..DIM_IN {
            grad[(i, j)] = fx_dual[i][j + 1];
        }
    }
    (fx, grad)
}
