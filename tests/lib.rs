extern crate hyperdual;
extern crate nalgebra as na;

use na::{Const, Matrix2x6, Matrix6, SMatrix, SVector, Vector2, Vector3, Vector6};

use hyperdual::linalg::norm;
use hyperdual::{
    differentiate, extract_jacobian_and_result, hyperspace_from_vector, vector_from_hyperspace, Dual, DualN, Float, FloatConst, Hyperdual,
};

macro_rules! abs_within {
    ($x:expr, $val:expr, $eps:expr, $msg:expr) => {
        assert!(($x - $val).abs() <= $eps, $msg)
    };
}

macro_rules! zero_within {
    ($x:expr, $eps:expr, $msg:expr) => {
        assert!($x.abs() <= $eps, $msg)
    };
}

#[test]
fn default() {
    assert_eq!(Dual::<f64>::default(), Dual::new(0., 0.));
}

#[test]
fn sum_product() {
    let a = [Dual::new(1.0, 1.0), Dual::new(0.5, 0.5)];
    assert_eq!(a.iter().cloned().sum::<Dual<f64>>(), Dual::new(1.5, 1.5));
    assert_eq!(a.iter().cloned().product::<Dual<f64>>(), Dual::new(0.5, 1.0));
}

#[test]
fn derive() {
    abs_within!(
        differentiate(4.0f64, |x| x.sqrt() + Dual::from_real(1.0)),
        1.0 / 4.0,
        std::f64::EPSILON,
        "incorrect norm"
    );

    println!(
        "{:.16}",
        differentiate(1.0f64, |x| {
            let one = Dual::from_real(1.0); // Or use the One trait

            one / (one + Dual::E().powf(-x))
        })
    );

    println!("{:.5}", (Dual::new(0.25f32, 1.0) * Dual::PI()).sin());

    let mut x = Dual::new(2i32, 1);

    x = x * x + x;

    assert_eq!(x.real(), 6i32, "incorrect real");
    assert_eq!(x.dual(), 5i32, "incorrect real");

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).asin();
    abs_within!(c.dual(), std::f64::consts::SQRT_2, std::f64::EPSILON, "incorrect d/dx arcsin");

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).acos();
    abs_within!(c.dual(), -std::f64::consts::SQRT_2, std::f64::EPSILON, "incorrect d/dx arccos");

    let c = Dual::new(1.0 / 2f64.sqrt(), 1.0).atan();
    abs_within!(c.dual(), 2.0f64 / 3.0f64, std::f64::EPSILON, "incorrect d/dx arctan");
}

#[test]
fn type_operations() {
    let mut x = Dual::new(1.0, 2.0);
    let val = 3.0f64;
    abs_within!((x + val).real(), 4.0, std::f64::EPSILON, "add failed on real part");
    abs_within!((x + val).dual(), 2.0, std::f64::EPSILON, "add failed on dual part");
    x += val;
    abs_within!(x.real(), 4.0, std::f64::EPSILON, "add_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "add_assign failed on dual part");
    abs_within!((x - val).real(), 1.0, std::f64::EPSILON, "sub failed on real part");
    abs_within!((x - val).dual(), 2.0, std::f64::EPSILON, "sub failed on dual part");
    x -= val;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "sub_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "sub_assign failed on dual part");
    abs_within!((x * val).real(), 3.0, std::f64::EPSILON, "mul failed on real part");
    abs_within!((x * val).dual(), 6.0, std::f64::EPSILON, "mul failed on dual part");
    x *= val;
    abs_within!(x.real(), 3.0, std::f64::EPSILON, "mul_assign failed on real part");
    abs_within!(x.dual(), 6.0, std::f64::EPSILON, "mul_assign failed on dual part");
    abs_within!((x / val).real(), 1.0, std::f64::EPSILON, "div failed on real part");
    abs_within!((x / val).dual(), 2.0, std::f64::EPSILON, "div failed on dual part");
    x /= val;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "div_assign failed on real part");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "div_assign failed on dual part");
}

#[test]
fn dual_operations() {
    let mut x = Dual::new(1.0, 2.0);
    let y = Dual::new(3.0, 4.0);
    abs_within!((x + y).real(), 4.0, std::f64::EPSILON, "add failed");
    abs_within!((x + y).dual(), 6.0, std::f64::EPSILON, "add failed");
    x += y;
    abs_within!(x.real(), 4.0, std::f64::EPSILON, "add_assign failed");
    abs_within!(x.dual(), 6.0, std::f64::EPSILON, "add_assign failed");
    abs_within!((x - y).real(), 1.0, std::f64::EPSILON, "sub failed");
    abs_within!((x - y).dual(), 2.0, std::f64::EPSILON, "sub failed");
    x -= y;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "sub_assign failed");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "sub_assign failed");
    abs_within!((x * y).real(), 3.0, std::f64::EPSILON, "mul failed");
    abs_within!((x * y).dual(), 10.0, std::f64::EPSILON, "mul failed");
    x *= y;
    abs_within!(x.real(), 3.0, std::f64::EPSILON, "mul_assign failed");
    abs_within!(x.dual(), 10.0, std::f64::EPSILON, "mul_assign failed");
    abs_within!((x / y).real(), 1.0, std::f64::EPSILON, "div failed");
    abs_within!((x / y).dual(), 2.0, std::f64::EPSILON, "div failed");
    x /= y;
    abs_within!(x.real(), 1.0, std::f64::EPSILON, "div_assign failed");
    abs_within!(x.dual(), 2.0, std::f64::EPSILON, "div_assign failed");
}

#[test]
fn linalg() {
    // NOTE: Due to the implementation of std::ops::Mul in nalgebra, the syntax _must_ be vec * x
    // where x is the scalar and vec the vector.
    // Quote from the author, sebcrozet
    // > The thing is that nalgebra cannot define the multiplication of a scalar by a vector
    // > (where the scalar is on the left hand side) because such an implementation would look like
    // > this: `impl<T: Scalar> Mul<Vector<T>> for T` which is forbidden by the compiler. That's
    // > why the only multiplication automatically provided by nalgebra is when the scalar is on
    // > the right-hand-side. When `T` here is `f32` or `f64` both multiplication orders work.
    let vec = Vector2::new(Dual::from(1.0f64), Dual::new(-2.0f64, 3.5f64));
    let x = Dual::new(2.0f64, 0.5f64);
    let computed = vec * x;
    let expected = Vector2::new(Dual::new(2.0f64, 0.5f64), Dual::new(-4.0f64, 6.0f64));

    for i in 0..2 {
        zero_within!(
            (expected - computed)[(i, 0)].real(),
            1e-16,
            format!("Vector2 multiplication incorrect (i={})", i)
        );
        zero_within!(
            (expected - computed)[(i, 0)].dual(),
            1e-16,
            format!("Vector2 multiplication incorrect (i={})", i)
        );
    }

    // Checking the dot product
    // NOTE: The tolerance is relatively high because of some rounding error probably due to the powi call.
    let delta = computed.dot(&expected) - norm(&computed).powi(2);
    zero_within!(delta.real(), 1e-12, "real part of the dot product is incorrect");
    zero_within!(delta.dual(), 1e-12, "dual part of the dot product is incorrect");

    let vec = Vector3::new(Dual::from_real(1.0), Dual::from_real(1.0), Dual::from_real(1.0));
    let this_norm = norm(&vec);
    abs_within!(this_norm.real(), 3.0f64.sqrt(), std::f64::EPSILON, "incorrect real part of the norm");
    zero_within!(this_norm.dual(), std::f64::EPSILON, "incorrect dual part of the norm");
}

#[test]
fn multivariate() {
    // find partial derivative at x=4.0, y=5.0 for f(x,y)=x^2+sin(x*y)+y^3
    let x: Hyperdual<f64, 3> = Hyperdual::from_slice(&[4.0, 1.0, 0.0]);
    // DualN and Hyperdual are interchangeable aliases. Hyperdual is the name from Fike 2012
    // whereas multi-dual is from Revel et al. 2016.
    let y: DualN<f64, 3> = Hyperdual::from_slice(&[5.0, 0.0, 1.0]);

    let res = x * x + (x * y).sin() + y.powi(3);
    zero_within!((res[0] - 141.91294525072763), 1e-13, "f(4, 5) incorrect");
    zero_within!((res[1] - 10.04041030906696), 1e-13, "df/dx(4, 5) incorrect");
    zero_within!((res[2] - 76.63232824725357), 1e-13, "df/dy(4, 5) incorrect");
}

#[test]
fn state_gradient() {
    // This is an example of the equation of motion gradient for a spacecrate in a two body acceleration.

    type StateVectorType = SVector<f64, 6>;
    type JacobianType = SMatrix<f64, 6, 6>;
    type StateVectorDualType = SVector<DualN<f64, 7>, 6>;
    type EomFn<T> = fn(f64, &T) -> T;

    fn eom(_t: f64, state: &StateVectorDualType) -> StateVectorDualType {
        // Extract data from hyperspace
        let radius = state.fixed_rows::<3>(0).into_owned();
        let velocity = state.fixed_rows::<3>(3).into_owned();

        // Code up math as usual
        let rmag = norm(&radius);
        let body_acceleration = radius * (Hyperdual::<f64, 7>::from_real(-398_600.441_5) / rmag.powi(3));

        // Added for inspection only
        println!("velocity = {}", velocity);
        println!("body_acceleration = {}", body_acceleration);

        // Return only the EOMs
        Vector6::new(
            velocity[0],
            velocity[1],
            velocity[2],
            body_acceleration[0],
            body_acceleration[1],
            body_acceleration[2],
        )
    }
    fn eom_with_grad(eom: &EomFn<StateVectorDualType>, _t: f64, state: &StateVectorDualType) -> (StateVectorType, JacobianType) {
        let f_dual = eom(0.0, &state);
        // Extract result into Vector6 and Matrix6
        extract_jacobian_and_result(&f_dual)
    }

    let state = Vector6::new(
        -9042.862233600335,
        18536.333069123244,
        6999.9570694864115,
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
    );

    // Create a hyperdual space which allows for first derivatives.
    let hyperstate = hyperspace_from_vector(&state);

    // Added for inspection
    println!("hyperstate = {}", hyperstate);

    // Extract result into Vector6 and Matrix6
    let (fx, grad) = eom_with_grad(&(eom as EomFn<StateVectorDualType>), 0.0, &hyperstate);

    let expected_fx = Vector6::new(
        -3.28878900377057,
        -2.226285193102822,
        1.6467383807226765,
        0.0003488751720191492,
        -0.0007151349009902908,
        -0.00027005954128877916,
    );

    zero_within!((fx - expected_fx).norm(), 1e-16, "f(x) computation is incorrect");

    let mut expected = Matrix6::zeros();

    expected[(0, 3)] = 1.0;
    expected[(1, 4)] = 1.0;
    expected[(2, 5)] = 1.0;
    expected[(3, 0)] = -0.000000018628398676538285;
    expected[(4, 0)] = -0.00000004089774775108092;
    expected[(5, 0)] = -0.0000000154443965496673;
    expected[(3, 1)] = -0.00000004089774775108092;
    expected[(4, 1)] = 0.000000045253271751873843;
    expected[(5, 1)] = 0.00000003165839212196757;
    expected[(3, 2)] = -0.0000000154443965496673;
    expected[(4, 2)] = 0.00000003165839212196757;
    expected[(5, 2)] = -0.000000026624873075335538;

    zero_within!((grad - expected).norm(), 1e-16, "gradient computation is incorrect");
}

#[test]
fn state_partials() {
    type OutputVectorType = Vector2<f64>;
    type JacobianType = Matrix2x6<f64>;
    type StateVectorDualType = SVector<Hyperdual<f64, 7>, 6>;
    type OutputVectorDualType = SVector<Hyperdual<f64, 7>, 2>;
    type SensitivityFn = fn(&StateVectorDualType) -> OutputVectorDualType;

    // This is an example of the sensitivity matrix (H tilde) of a ranging method.
    fn sensitivity(state: &StateVectorDualType) -> OutputVectorDualType {
        // Extract data from hyperspace
        let range_vec = state.fixed_rows::<3>(0).into_owned();
        let velocity_vec = state.fixed_rows::<3>(3).into_owned();

        // Code up math as usual
        let delta_v_vec = velocity_vec / norm(&range_vec);
        let range = norm(&range_vec);
        let range_rate = range_vec.dot(&delta_v_vec);

        // Extract result into Vector2 and Matrix2x6
        Vector2::new(range, range_rate)
    }

    fn sensitivity_with_partials(eom: &SensitivityFn, state: &StateVectorDualType) -> (OutputVectorType, JacobianType) {
        let f_dual = eom(&state);
        extract_jacobian_and_result(&f_dual)
    }

    let vec = Vector6::new(
        4_354.653_483_456_944,
        18_090.191_366_880_514,
        2_901.658_181_581_637,
        -3.742_815_992_334_434_4,
        0.901_480_766_308_994_1,
        1.644_034_610_527_063_8,
    );

    let hyperstate = hyperspace_from_vector(&vec);

    // Added for inspection
    println!("hyperstate = {}", hyperstate);
    let (fx, dfdx) = sensitivity_with_partials(&(sensitivity as SensitivityFn), &hyperstate);

    let expected_fx = Vector2::new(18831.82547853717, 0.2538107291309079);

    zero_within!(
        (fx - expected_fx).norm(),
        1e-20,
        format!("f(x) computation is incorrect -- here comes the delta: {}", fx - expected_fx)
    );

    let mut expected_dfdx = Matrix2x6::zeros();
    expected_dfdx[(0, 0)] = 0.231_239_052_656_896_62;
    expected_dfdx[(0, 1)] = 0.960_618_044_570_246_1;
    expected_dfdx[(0, 2)] = 0.154_082_682_259_81;
    expected_dfdx[(1, 0)] = -0.00020186608829958833;
    expected_dfdx[(1, 1)] = 0.00003492309339579752;
    expected_dfdx[(1, 2)] = 0.00008522417406774546;
    expected_dfdx[(1, 3)] = 0.231_239_052_656_896_62;
    expected_dfdx[(1, 4)] = 0.960_618_044_570_246_1;
    expected_dfdx[(1, 5)] = 0.154_082_682_259_81;

    zero_within!(
        (dfdx - expected_dfdx).norm(),
        1e-20,
        format!("partial computation is incorrect -- here comes the delta: {}", dfdx - expected_dfdx)
    );
}

#[test]
fn test_hyperspace_from_vector() {
    let vec = Vector6::new(
        4_354.653_483_456_944,
        18_090.191_366_880_514,
        2_901.658_181_581_637,
        -3.742_815_992_334_434_4,
        0.901_480_766_308_994_1,
        1.644_034_610_527_063_8,
    );

    let hyperstate: SVector<DualN<f64, 7>, 6> = hyperspace_from_vector(&vec);

    for i in 0..6 {
        abs_within!(hyperstate[i].real(), vec[i], std::f64::EPSILON, "incorrect real value");
        for j in 1..7 {
            let expected = if j == i + 1 { 1.0 } else { 0.0 };
            abs_within!(hyperstate[i][j], expected, std::f64::EPSILON, "incorrect partial");
        }
    }
}

#[test]
fn test_vector_from_hyperspace() {
    let expected_vec = Vector6::new(
        4_354.653_483_456_944,
        18_090.191_366_880_514,
        2_901.658_181_581_637,
        -3.742_815_992_334_434_4,
        0.901_480_766_308_994_1,
        1.644_034_610_527_063_8,
    );
    let dual_vec = Vector6::new(
        DualN::<f64, 7>::from_real(4_354.653_483_456_944),
        DualN::<f64, 7>::from_real(18_090.191_366_880_514),
        DualN::<f64, 7>::from_real(2_901.658_181_581_637),
        DualN::<f64, 7>::from_real(-3.742_815_992_334_434_4),
        DualN::<f64, 7>::from_real(0.901_480_766_308_994_1),
        DualN::<f64, 7>::from_real(1.644_034_610_527_063_8),
    );

    let vector = vector_from_hyperspace(&dual_vec);

    zero_within!(
        (vector - expected_vec).norm(),
        std::f64::EPSILON,
        format!("vector_from_hyperspace test failed, norm: {}", vector - expected_vec)
    );
}

#[test]
fn test_hypot() {
    let x: Hyperdual<f64, 3> = Hyperdual::from_slice(&[3.0, 1.0, 0.0]);
    let y: Hyperdual<f64, 3> = Hyperdual::from_slice(&[4.0, 0.0, 1.0]);
    let dist1 = (x * x + y * y).sqrt();
    let dist2 = x.hypot(y);

    abs_within!(dist1, dist2, std::f64::EPSILON, "incorrect hypot reals");
    abs_within!(dist1[1], dist2[1], std::f64::EPSILON, "incorrect hypot df/dx");
    abs_within!(dist1[2], dist2[2], std::f64::EPSILON, "incorrect hypot df/dy");
}

#[test]
fn test_div() {
    let x: Hyperdual<f64, 3> = Hyperdual::from_slice(&[3.0, 1.0, 0.0]);
    let y: Hyperdual<f64, 3> = Hyperdual::from_slice(&[4.0, 0.0, 1.0]);
    let rslt = x / y;
    let expt: Hyperdual<f64, 3> = Hyperdual::from_slice(&[3.0 / 4.0, 1.0 / 4.0, -3.0 / 16.0]);

    abs_within!(rslt, expt, std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
    abs_within!(rslt[2], expt[2], std::f64::EPSILON, "incorrect df/dy");
}

#[test]
fn test_powf() {
    let x: Hyperdual<f64, 2> = Hyperdual::from_slice(&[3.0, 1.0]);
    let y: Hyperdual<f64, 2> = Hyperdual::from_slice(&[2.3, 0.0]);
    let rslt = x.powf(y);
    let expt: Hyperdual<f64, 2> = Hyperdual::from_slice(&[12.513502532843182, 9.593685275179771]);

    abs_within!(rslt, expt, std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
}

#[test]
fn test_trig_atan2() {
    /*
    Reproduce in sympy with the following (note that we are doing y.atan2(x))
    The same test was done with atan and atan2, leading to the same result.
    >>> from sympy import *
    >>> x, y = symbols('x y')
    >>> expr = atan(y/x)
    >>> dfdx = expr.diff(x)
    >>> dfdy = expr.diff(y)
    >>> dfdy.evalf(subs={y:2.0, x:3.0})
    0.230769230769231
    >>> dfdx.evalf(subs={y:2.0, x:3.0})
    -0.153846153846154
    >>>

        */
    let x: Hyperdual<f64, 3> = Hyperdual::from_slice(&[3.0, 0.0, 1.0]);
    let y: Hyperdual<f64, 3> = Hyperdual::from_slice(&[2.0, 1.0, 0.0]);
    let rslt = y.atan2(x);
    let rslt_atan = (y / x).atan();
    let expt: Hyperdual<f64, 3> = Hyperdual::from_slice(&[0.5880026035475675, 0.23076923076923075, -0.15384615384615383]);

    abs_within!(dbg!(rslt), dbg!(expt), std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
    abs_within!(rslt[2], expt[2], std::f64::EPSILON, "incorrect df/dy");
    abs_within!(dbg!(rslt[0].tan()), 2.0 / 3.0, std::f64::EPSILON, "incorrect inverse function");

    abs_within!(dbg!(rslt_atan), dbg!(expt), std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt_atan[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
    abs_within!(rslt_atan[2], expt[2], std::f64::EPSILON, "incorrect df/dy");
    abs_within!(dbg!(rslt_atan[0].tan()), 2.0 / 3.0, std::f64::EPSILON, "incorrect inverse function");
}

#[test]
fn test_trig_acos() {
    /*
    Reproduce in sympy with the following (note that we are doing y.atan2(x))
    >>> from sympy import *
    >>> x, y = symbols('x y')
    >>> expr_acos = acos(x)
    >>> dfdx_acos = expr_acos.diff(x)
    >>> dfdx_acos.evalf(subs={x:0.59})
    -1.23853849513327
    >>>
        */
    let x: Hyperdual<f64, 2> = Hyperdual::from_slice(&[0.59, 1.0]);
    let rslt = x.acos();
    let expt: Hyperdual<f64, 2> = Hyperdual::from_slice(&[0.9397374860168752, -1.238538495133269]);

    abs_within!(dbg!(rslt), dbg!(expt), std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
    abs_within!(rslt[0].cos(), 0.59, std::f64::EPSILON, "incorrect inverse function");

    let rslt_deg = rslt.to_degrees();
    abs_within!(rslt_deg[0], rslt[0].to_degrees(), std::f64::EPSILON, "incorrect to_degrees");
    abs_within!(rslt_deg[1], rslt[1].to_degrees(), std::f64::EPSILON, "incorrect to_degrees for dual");
    abs_within!(rslt_deg.to_radians(), rslt, std::f64::EPSILON, "incorrect return conversion");
}

#[test]
fn test_trig_asin() {
    /*
    Reproduce in sympy with the following (note that we are doing y.atan2(x))
    >>> from sympy import *
    >>> x, y = symbols('x y')
    >>> expr_asin = asin(x)
    >>> dfdx_asin = expr_asin.diff(x)
    >>> dfdx_asin.evalf(subs={x:0.59})
    1.23853849513327
    >>>
        */
    let x: Hyperdual<f64, 2> = Hyperdual::from_slice(&[0.59, 1.0]);
    let rslt = x.asin();
    let expt: Hyperdual<f64, 2> = Hyperdual::from_slice(&[0.6310588407780213, 1.238538495133269]);

    abs_within!(dbg!(rslt), dbg!(expt), std::f64::EPSILON, "incorrect reals");
    abs_within!(rslt[1], expt[1], std::f64::EPSILON, "incorrect df/dx");
    abs_within!(rslt[0].sin(), 0.59, std::f64::EPSILON, "incorrect inverse function");
}
