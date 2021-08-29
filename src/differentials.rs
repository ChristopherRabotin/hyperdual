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
pub fn extract_jacobian_and_result<T: Scalar + Zero + Float, const DimIn: usize, const DimOut: usize, const DimHyper: usize>(
    fx_dual: &SVector<DualN<T, DimHyper>, DimOut>,
) -> (SVector<T, DimOut>, SMatrix<T, DimOut, DimIn>)
{
    let fx = super::vector_from_hyperspace(&fx_dual);
    let mut grad = SMatrix::<T, DimOut, DimIn>::zeros();

    for i in 0..DimOut {
        for j in 0..DimIn {
            grad[(i, j)] = fx_dual[i][j + 1];
        }
    }
    (fx, grad)
}
