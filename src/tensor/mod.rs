use ndarray::{Array, ArrayD, Ix1, Ix2, IxDyn, ArrayViewD, ArrayViewMutD, Dimension, Dim};
use num_traits::{Float, FromPrimitive, Zero, One};
use serde::{Deserialize, Serialize};
use std::ops;
use std::fmt;
use std::iter::Sum;

/// A multi-dimensional array that supports automatic differentiation.
/// 自動微分をサポートする多次元配列
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Float> {
    data: ArrayD<T>,
}

impl<T: Float + 'static> Tensor<T> {
    /// Creates a new tensor from an array.
    /// 配列から新しいテンソルを作成します。
    pub fn new(data: ArrayD<T>) -> Self {
        Tensor { data }
    }

    /// Creates a tensor filled with zeros.
    /// ゼロで埋められたテンソルを作成します。
    pub fn zeros<D: IntoDimension>(dims: D) -> Self {
        let dims = dims.into_dimension();
        Tensor {
            data: ArrayD::zeros(dims),
        }
    }

    /// Creates a tensor filled with ones.
    /// 1で埋められたテンソルを作成します。
    pub fn ones<D: IntoDimension>(dims: D) -> Self {
        let dims = dims.into_dimension();
        Tensor {
            data: ArrayD::ones(dims),
        }
    }

    /// Returns a reference to the underlying array.
    /// 内部の配列への参照を返します。
    pub fn as_array(&self) -> &ArrayD<T> {
        &self.data
    }

    /// Returns a view of the underlying array.
    /// 内部の配列のビューを返します。
    pub fn view(&self) -> ArrayViewD<T> {
        self.data.view()
    }

    /// Returns a mutable reference to the underlying array.
    /// 内部の配列への可変参照を返します。
    pub fn as_array_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    /// Returns the shape of the tensor.
    /// テンソルの形状を返します。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of elements in the tensor.
    /// テンソルの要素数を返します。
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the tensor contains no elements.
    /// テンソルが空の場合は`true`を返します。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshapes the tensor to the given shape.
    /// テンソルを指定された形状に変形します。
    pub fn reshape(self, shape: &[usize]) -> Self {
        Tensor {
            data: self.data.into_shape(shape).unwrap(),
        }
    }

    /// Transposes the tensor by reversing its dimensions.
    /// テンソルの次元を反転させて転置します。
    pub fn transpose(&self) -> Self {
        let ndim = self.data.ndim();
        let mut axes: Vec<usize> = (0..ndim).rev().collect();
        if ndim < 2 {
            return self.clone();
        }
        Tensor {
            data: self.data.view().permuted_axes(axes.as_slice()).to_owned(),
        }
    }

    /// Performs matrix multiplication with another tensor.
    /// 別のテンソルとの行列乗算を実行します。
    pub fn matmul(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let lhs = &self.data;
        let rhs = &rhs.data;
        
        match (lhs.ndim(), rhs.ndim()) {
            (1, 1) => {
                // Dot product
                let sum = lhs.iter().zip(rhs.iter()).fold(T::zero(), |acc, (&a, &b)| acc + a * b);
                Tensor::new(ArrayD::from_elem(IxDyn(&[]), sum))
            },
            (2, 1) => {
                // Matrix-vector multiplication
                let rhs = rhs.view().into_dimensionality::<Ix1>().unwrap();
                let lhs = lhs.view().into_dimensionality::<Ix2>().unwrap();
                let result = lhs.dot(&rhs);
                Tensor::new(result.into_dyn())
            },
            (2, 2) => {
                // Matrix-matrix multiplication
                let lhs = lhs.view().into_dimensionality::<Ix2>().unwrap();
                let rhs = rhs.view().into_dimensionality::<Ix2>().unwrap();
                let result = lhs.dot(&rhs);
                Tensor::new(result.into_dyn())
            },
            _ => panic!("Unsupported dimensions for matmul: {:?} and {:?}", lhs.shape(), rhs.shape()),
        }
    }

    /// Computes the sum of the tensor along the specified axis.
    /// 指定された軸に沿ってテンソルの和を計算します。
    pub fn sum_axis(&self, axis: usize) -> Tensor<T> {
        let sum = self.data.sum_axis(ndarray::Axis(axis));
        let dim = sum.raw_dim();
        Tensor {
            data: sum.into_shape(dim).unwrap(),
        }
    }
}

impl<T: Float + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<T: Float + 'static> From<ArrayD<T>> for Tensor<T> {
    fn from(data: ArrayD<T>) -> Self {
        Tensor::new(data)
    }
}

impl<T: Float + 'static> From<ndarray::Array1<T>> for Tensor<T> {
    fn from(array: ndarray::Array1<T>) -> Self {
        Tensor::new(array.into_dyn())
    }
}

impl<T: Float + 'static> From<ndarray::Array2<T>> for Tensor<T> {
    fn from(array: ndarray::Array2<T>) -> Self {
        Tensor::new(array.into_dyn())
    }
}

impl<T: Float> ops::Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data + &rhs.data,
        }
    }
}

impl<T: Float> ops::Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data - &rhs.data,
        }
    }
}

impl<T: Float> ops::Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data * &rhs.data,
        }
    }
}

impl<T: Float> ops::Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data / &rhs.data,
        }
    }
}

impl<T: Float> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let mut result = self.data.clone();
        result.mapv_inplace(|x| -x);
        Tensor { data: result }
    }
}

impl<T: Float> ops::AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a + b);
    }
}

impl<T: Float> ops::SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a - b);
    }
}

impl<T: Float> ops::MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a * b);
    }
}

impl<T: Float> ops::DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, rhs: &Tensor<T>) {
        let rhs_data = &rhs.data;
        self.data.zip_mut_with(rhs_data, |a, &b| *a = *a / b);
    }
}
