//! Mathematical operations for tensors
//! テンソルの数学演算

use super::core::Tensor;
// Removed unused imports
use num_traits::Float;
use std::ops;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise addition with another tensor.
    /// 別のテンソルとの要素ごとの加算
    pub fn add(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() + other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Element-wise subtraction with another tensor.
    /// 別のテンソルとの要素ごとの減算
    pub fn sub(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() - other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Element-wise multiplication with another tensor.
    /// 別のテンソルとの要素ごとの乗算
    pub fn mul(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() * other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Element-wise division with another tensor.
    /// 別のテンソルとの要素ごとの除算
    pub fn div(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() / other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Scalar addition.
    /// スカラー加算
    pub fn add_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() + scalar;
        Tensor::new(result_data)
    }

    /// Scalar subtraction.
    /// スカラー減算
    pub fn sub_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() - scalar;
        Tensor::new(result_data)
    }

    /// Scalar multiplication.
    /// スカラー乗算
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() * scalar;
        Tensor::new(result_data)
    }

    /// Scalar division.
    /// スカラー除算
    pub fn div_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() / scalar;
        Tensor::new(result_data)
    }

    /// Negation of all elements.
    /// 全要素の符号反転
    pub fn neg(&self) -> Self {
        let result_data = self.as_array().mapv(|x| -x);
        Tensor::new(result_data)
    }

    /// Matrix multiplication.
    /// 行列乗算
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape[1] != other_shape[0] {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} @ {}x{}",
                self_shape[0], self_shape[1], other_shape[0], other_shape[1]
            ));
        }

        let result_shape = vec![self_shape[0], other_shape[1]];
        let mut result_data = vec![T::zero(); result_shape.iter().product()];

        let self_data = self.as_slice().unwrap();
        let other_data = other.as_slice().unwrap();

        for i in 0..result_shape[0] {
            for j in 0..result_shape[1] {
                let mut sum = T::zero();
                for k in 0..self_shape[1] {
                    let self_idx = i * self_shape[1] + k;
                    let other_idx = k * other_shape[1] + j;
                    sum = sum + self_data[self_idx] * other_data[other_idx];
                }
                result_data[i * result_shape[1] + j] = sum;
            }
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    /// Transpose the tensor (swap last two dimensions).
    /// テンソルの転置（最後の2次元を入れ替え）
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() < 2 {
            return Err("Transpose requires at least 2D tensor".to_string());
        }

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let ndim = self.ndim();
        axes.swap(ndim - 2, ndim - 1);

        let transposed = self.as_array().permuted_axes(axes.into_iter().collect::<Vec<_>>());
        Ok(Tensor::new(transposed))
    }

    /// Sum of all elements.
    /// 全要素の合計
    pub fn sum(&self) -> T {
        self.as_array().sum()
    }

    /// Mean of all elements.
    /// 全要素の平均
    pub fn mean(&self) -> T {
        let sum = self.sum();
        let count = T::from(self.numel()).unwrap_or(T::one());
        sum / count
    }

    /// Sum along a specific axis.
    /// 特定の軸に沿った合計
    pub fn sum_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} out of bounds for {}-dimensional tensor", axis, self.ndim()));
        }

        let sum_result = self.as_array().sum_axis(ndarray::Axis(axis));
        Ok(Tensor::new(sum_result))
    }

    /// Mean along a specific axis.
    /// 特定の軸に沿った平均
    pub fn mean_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} out of bounds for {}-dimensional tensor", axis, self.ndim()));
        }

        let mean_result = self.as_array().mean_axis(ndarray::Axis(axis)).unwrap();
        Ok(Tensor::new(mean_result))
    }

    /// Apply a function element-wise.
    /// 要素ごとに関数を適用
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let result_data = self.as_array().mapv(f);
        Tensor::new(result_data)
    }

    /// Apply exponential function element-wise.
    /// 要素ごとに指数関数を適用
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    /// Apply natural logarithm element-wise.
    /// 要素ごとに自然対数を適用
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Apply sine function element-wise.
    /// 要素ごとにサイン関数を適用
    pub fn sin(&self) -> Self {
        self.map(|x| x.sin())
    }

    /// Apply cosine function element-wise.
    /// 要素ごとにコサイン関数を適用
    pub fn cos(&self) -> Self {
        self.map(|x| x.cos())
    }

    /// Apply tangent function element-wise.
    /// 要素ごとにタンジェント関数を適用
    pub fn tan(&self) -> Self {
        self.map(|x| x.tan())
    }

    /// Apply square root element-wise.
    /// 要素ごとに平方根を適用
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    /// Apply absolute value element-wise.
    /// 要素ごとに絶対値を適用
    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    /// Power operation element-wise.
    /// 要素ごとのべき乗演算
    pub fn pow(&self, exponent: T) -> Self {
        self.map(|x| x.powf(exponent))
    }

    /// Maximum value of the tensor.
    /// テンソルの最大値
    pub fn max(&self) -> Option<T> {
        self.as_slice()?.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }

    /// Minimum value of the tensor.
    /// テンソルの最小値
    pub fn min(&self) -> Option<T> {
        self.as_slice()?.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }
}

// Operator overloading for tensors
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs).expect("Subtraction failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs).expect("Multiplication failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs).expect("Division failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::ops::Neg<Output = T>> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let result_data = self.as_array().mapv(|x| -x);
        Tensor::new(result_data)
    }
}

// Scalar operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// In-place operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.add(rhs).expect("AddAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.sub(rhs).expect("SubAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.mul(rhs).expect("MulAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.div(rhs).expect("DivAssign failed");
    }
}