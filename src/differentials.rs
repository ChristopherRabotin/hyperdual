use super::Float;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName};
use na::{MatrixN, VectorN};
use {Dual, DualN, One, Scalar, Zero};

/// Evaluates the function using dual numbers to get the partial derivative at the input point
#[inline]
pub fn differentiate<T: Scalar + One, F>(x: T, f: F) -> T
where
    F: Fn(Dual<T>) -> Dual<T>,
{
    f(Dual::new(x, T::one())).dual()
}

#[inline]
pub fn get_jacobian_and_result<T: Scalar + Zero + Float, DimX: Dim + DimName, DimY: Dim + DimName>(
    fx_dual: &VectorN<DualN<T, DimY>, DimX>,
) -> (VectorN<T, DimX>, MatrixN<T, DimX>)
where
    DefaultAllocator: Allocator<T, DimX> + Allocator<T, DimX, DimX> + Allocator<DualN<T, DimY>, DimX> + Allocator<T, DimY>,
    <DefaultAllocator as Allocator<T, DimY>>::Buffer: Copy,
{
    let fx = super::vector_from_hyperspace(&fx_dual);
    let mut grad = MatrixN::<T, DimX>::zeros();

    for i in 0..DimX::dim() {
        for j in 0..DimX::dim() {
            grad[(i, j)] = fx_dual[i][j + 1];
        }
    }
    (fx, grad)
}
