//! Operator overloads for tensor operations
//! テンソル演算の演算子オーバーロード

use super::super::core::Tensor;
use num_traits::Float;
use std::ops;

// Binary tensor operations (&Tensor<T> op &Tensor<T>)
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs).expect("Subtraction failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs).expect("Multiplication failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs).expect("Division failed")
    }
}

// Unary operations
impl<
        T: Float
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive
            + std::ops::Neg<Output = T>,
    > ops::Neg for &Tensor<T>
{
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        // Use the neg method from arithmetic module
        let result_data = self.data.iter().map(|&x| -x).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }
}

// Scalar operations (&Tensor<T> op T)
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<T>
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub<T>
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul<T>
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div<T>
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// In-place operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    ops::AddAssign<&Tensor<T>> for Tensor<T>
{
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        *self = self
            .add(rhs)
            .map_err(|e| e.to_string())
            .expect("AddAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    ops::SubAssign<&Tensor<T>> for Tensor<T>
{
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        *self = self
            .sub(rhs)
            .map_err(|e| e.to_string())
            .expect("SubAssign failed");
    }
}

// Value operations (Tensor<T> op Tensor<T>)
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        (&self)
            .add(&rhs)
            .map_err(|e| e.to_string())
            .expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs).expect("Subtraction failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs).expect("Multiplication failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs).expect("Division failed")
    }
}

// Scalar operations with tensor values (Tensor<T> op T)
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<T>
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub<T>
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul<T>
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div<T>
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// Mixed reference/value operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<&Tensor<T>>
    for Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        (&self).add(rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<Tensor<T>>
    for &Tensor<T>
{
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        self.add(&rhs).expect("Addition failed")
    }
}
