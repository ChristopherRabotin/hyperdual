use super::Float;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName};
use na::{OMatrix, OVector};
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
pub fn extract_jacobian_and_result<T: Scalar + Zero + Float, DimIn: Dim + DimName, DimOut: Dim + DimName, DimHyper: Dim + DimName>(
    fx_dual: &OVector<DualN<T, DimHyper>, DimOut>,
) -> (OVector<T, DimOut>, OMatrix<T, DimOut, DimIn>)
where
    DefaultAllocator:
        Allocator<T, DimIn> + Allocator<T, DimOut> + Allocator<T, DimOut, DimIn> + Allocator<DualN<T, DimHyper>, DimOut> + Allocator<T, DimHyper>,
    <DefaultAllocator as Allocator<T, DimHyper>>::Buffer: Copy,
{
    let fx = super::vector_from_hyperspace(&fx_dual);
    let mut grad = OMatrix::<T, DimOut, DimIn>::zeros();

    for i in 0..DimOut::dim() {
        for j in 0..DimIn::dim() {
            grad[(i, j)] = fx_dual[i][j + 1];
        }
    }
    (fx, grad)
}
