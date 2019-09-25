// extern crate hyperdual;
// extern crate nalgebra as na;
// extern crate num_traits;
// use std::f64::consts::PI;

fn main() {}
// use hyperdual::{get_jacobian_and_result, hyperspace_from_vector, vector_from_hyperspace, DualN};
// use na::*;

// fn eom(_t: f64, state: &VectorN<DualN<f64, DimSum<U3, U1>>, U3>) -> VectorN<DualN<f64, DimSum<U3, U1>>, U3> {
//     let g: f64 = -9.81;
//     let theta: f64 = 45. * PI / 180.0;
//     let cos_theta = theta.cos();
//     let sin_theta = theta.sin();

//     // let x = state[0].to_owned();
//     // let y = state[1].to_owned();
//     let v = state[2].to_owned();

//     Vector3::new(v * cos_theta, v * sin_theta, DualN::from_real(sin_theta * g))
// }

// fn main() {
//     // find the partial derivatives of a multivariate function
//     let state = Vector3::new(1., 10., 0.);

//     // Create a hyperdual space which allows for first derivatives.
//     let hyperstate = hyperspace_from_vector(&state);

//     fn eom_with_stm<F>(eom: &F, _t: f64, state: &VectorN<DualN<f64, U4>, U3>) -> 
//         (VectorN<f64, U3>, MatrixN<f64, U3>)
//         where F: Fn(f64, &VectorN<DualN<f64, U4>, U3>) -> VectorN<DualN<f64, U4>, U3> 
//     {
//         let f_dual = eom(0.0, &state);
//         // Extract result into Vector6 and Matrix6
//         get_jacobian_and_result(&f_dual)
//     }

//     let (f, g) = eom_with_stm(&eom, 0.0, &hyperstate);
//     println!("{:?}", g);
//     println!("{:?}", f);
// }
