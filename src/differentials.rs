use super::Float;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName};
use na::{MatrixMN, VectorN};
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
pub fn get_jacobian_and_result<T: Scalar + Zero + Float, 
                               DimInputs: Dim + DimName,
                               DimOutputs: Dim + DimName,
                               DimDual: Dim + DimName>
        (fx_dual: &VectorN<DualN<T, DimDual>, DimOutputs>) 
            -> (VectorN<T, DimOutputs>, MatrixMN<T, DimOutputs, DimInputs>)
where
    DefaultAllocator: Allocator<T, DimInputs> 
            + Allocator<T, DimOutputs> 
            + Allocator<T, DimOutputs, DimInputs> 
            + Allocator<DualN<T, DimDual>, DimOutputs> 
            + Allocator<T, DimDual>,
    <DefaultAllocator as Allocator<T, DimDual>>::Buffer: Copy,
{
    let fx = super::vector_from_hyperspace(&fx_dual);
    let mut grad =  MatrixMN::<T, DimOutputs, DimInputs>::zeros();

    for i in 0..DimOutputs::dim() {
        for j in 0..DimInputs::dim() {
            grad[(i, j)] = fx_dual[i][j + 1];
        }
    }
    (fx, grad)
}
