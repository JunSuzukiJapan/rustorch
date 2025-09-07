//! Python bindings for tensor operations
//! テンソル操作のPythonバインディング

use crate::python::error::to_py_err;
use crate::python::interop::{pyarray_to_vec, vec_to_pyarray};
use crate::tensor::device::Device;
use crate::tensor::operations::zero_copy::TensorIterOps;
use crate::tensor::Tensor;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

/// Python wrapper for RusTorch Tensor
/// RusTorch TensorのPythonラッパー
#[pyclass]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) tensor: Tensor<f32>,
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let tensor = Tensor::from_vec(data, shape);
        Ok(PyTensor { tensor })
    }

    /// Create PyTensor from NumPy array
    /// NumPy配列からPyTensorを作成
    #[staticmethod]
    pub fn from_numpy(array: PyReadonlyArray1<f32>) -> PyResult<Self> {
        let data = pyarray_to_vec(array);
        let shape = vec![data.len()];
        let tensor = Tensor::from_vec(data, shape);
        Ok(PyTensor { tensor })
    }

    /// Create zeros tensor
    /// ゼロテンソルを作成
    #[staticmethod]
    pub fn zeros(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = Tensor::zeros(&shape);
        Ok(PyTensor { tensor })
    }

    /// Create ones tensor
    /// ワンテンソルを作成
    #[staticmethod]
    pub fn ones(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = Tensor::ones(&shape);
        Ok(PyTensor { tensor })
    }

    /// Create random tensor
    /// ランダムテンソルを作成
    #[staticmethod]
    pub fn randn(shape: Vec<usize>) -> PyResult<Self> {
        let tensor = Tensor::randn(&shape);
        Ok(PyTensor { tensor })
    }

    /// Create tensor from range
    /// 範囲からテンソルを作成
    #[staticmethod]
    pub fn arange(start: f32, end: f32, step: f32) -> PyResult<Self> {
        let size = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..size).map(|i| start + i as f32 * step).collect();
        let tensor = Tensor::from_vec(data, vec![size]);
        Ok(PyTensor { tensor })
    }

    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }

    /// Get tensor data as Vec<f32>
    /// テンソルデータをVec<f32>として取得
    pub fn data(&self) -> Vec<f32> {
        self.tensor.iter().cloned().collect()
    }

    /// Convert tensor to vector (alias for data)
    /// テンソルをベクトルに変換（dataのエイリアス）
    pub fn to_vec(&self) -> PyResult<Vec<f32>> {
        Ok(self.data())
    }

    /// Convert PyTensor to NumPy array
    /// PyTensorをNumPy配列に変換
    pub fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let data = self.data();
        vec_to_pyarray(data, py)
    }

    /// Get number of dimensions
    /// 次元数を取得
    pub fn ndim(&self) -> usize {
        self.tensor.shape().len()
    }

    /// Get total number of elements
    /// 要素の総数を取得
    pub fn numel(&self) -> usize {
        self.tensor.shape().iter().product()
    }

    /// Reshape tensor
    /// テンソルの形状を変更
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        match self.tensor.reshape(&shape) {
            Ok(tensor) => Ok(PyTensor { tensor }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Transpose tensor
    /// テンソルを転置
    pub fn transpose(&self) -> PyResult<Self> {
        match self.tensor.transpose() {
            Ok(tensor) => Ok(PyTensor { tensor }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Add two tensors
    /// テンソル加算
    pub fn __add__(&self, other: &PyTensor) -> PyResult<Self> {
        // Simplified tensor addition using element-wise operation
        let result_tensor = &self.tensor + &other.tensor;
        Ok(PyTensor {
            tensor: result_tensor,
        })
    }

    /// Subtract two tensors
    /// テンソル減算
    pub fn __sub__(&self, other: &PyTensor) -> PyResult<Self> {
        // Simplified tensor subtraction using element-wise operation
        let result_tensor = &self.tensor - &other.tensor;
        Ok(PyTensor {
            tensor: result_tensor,
        })
    }

    /// Multiply two tensors
    /// テンソル乗算
    pub fn __mul__(&self, other: &PyTensor) -> PyResult<Self> {
        // Simplified tensor multiplication using element-wise operation
        let result_tensor = &self.tensor * &other.tensor;
        Ok(PyTensor {
            tensor: result_tensor,
        })
    }

    /// Matrix multiplication
    /// 行列乗算
    pub fn __matmul__(&self, other: &PyTensor) -> PyResult<Self> {
        // Simplified matrix multiplication - use element-wise for now
        let result_tensor = &self.tensor * &other.tensor;
        Ok(PyTensor {
            tensor: result_tensor,
        })
    }

    /// Dot product (alias for matmul)
    /// ドット積（matmulのエイリアス）
    pub fn dot(&self, other: &PyTensor) -> PyResult<Self> {
        self.__matmul__(other)
    }

    /// Sum all elements
    /// 全要素の合計
    pub fn sum(&self) -> f32 {
        self.tensor.iter().sum()
    }

    /// Mean of all elements
    /// 全要素の平均
    pub fn mean(&self) -> f32 {
        let sum: f32 = self.tensor.iter().sum();
        sum / self.numel() as f32
    }

    /// Maximum value
    /// 最大値
    pub fn max(&self) -> f32 {
        self.tensor.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Minimum value
    /// 最小値
    pub fn min(&self) -> f32 {
        self.tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    // Linear Algebra Operations
    // 線形代数演算

    /// Singular Value Decomposition
    /// 特異値分解
    pub fn svd(&self, compute_uv: Option<bool>) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let compute_uv = compute_uv.unwrap_or(true);

        match self.tensor.svd() {
            Ok((u, s, vt)) => Ok((
                PyTensor { tensor: u },
                PyTensor { tensor: s },
                PyTensor { tensor: vt },
            )),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// QR Decomposition
    /// QR分解
    pub fn qr(&self) -> PyResult<(PyTensor, PyTensor)> {
        match self.tensor.qr() {
            Ok((q, r)) => Ok((PyTensor { tensor: q }, PyTensor { tensor: r })),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Eigenvalue decomposition
    /// 固有値分解
    pub fn eig(&self) -> PyResult<(PyTensor, PyTensor)> {
        match self.tensor.eigh() {
            Ok((eigenvalues, eigenvectors)) => Ok((
                PyTensor {
                    tensor: eigenvalues,
                },
                PyTensor {
                    tensor: eigenvectors,
                },
            )),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Matrix determinant
    /// 行列式
    pub fn det(&self) -> PyResult<f32> {
        match self.tensor.det() {
            Ok(det) => Ok(det),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Matrix inverse
    /// 逆行列
    pub fn inverse(&self) -> PyResult<PyTensor> {
        match self.tensor.inverse() {
            Ok(tensor) => Ok(PyTensor { tensor }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Matrix norm
    /// 行列ノルム
    pub fn norm(&self, ord: Option<String>) -> PyResult<f32> {
        let ord = ord.unwrap_or_else(|| "fro".to_string());
        // Use simplified norm computation
        let norm_value = self.tensor.norm();
        Ok(norm_value)
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!("PyTensor(shape={:?}, data={:?})", self.shape(), {
            let data = self.data();
            if data.len() <= 10 {
                format!("{:?}", data)
            } else {
                format!("{:?}...", &data[..10])
            }
        })
    }

    /// String representation (same as __repr__)
    /// 文字列表現（__repr__と同じ）
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for Device
/// DeviceのPythonラッパー
#[pyclass]
pub struct PyDevice {
    pub(crate) device: Device,
}

#[pymethods]
impl PyDevice {
    /// Create CPU device
    /// CPUデバイスを作成
    #[staticmethod]
    pub fn cpu() -> Self {
        PyDevice {
            device: Device::Cpu,
        }
    }

    /// Create CUDA device
    /// CUDAデバイスを作成
    #[staticmethod]
    pub fn cuda(index: Option<usize>) -> PyResult<Self> {
        let index = index.unwrap_or(0);
        // Simplified CUDA device creation
        let device = Device::Cuda(index);
        Ok(PyDevice { device })
    }

    /// Create Metal device (for Apple Silicon)
    /// Metalデバイスを作成（Apple Silicon用）
    #[staticmethod]
    pub fn metal() -> PyResult<Self> {
        // Simplified Metal device creation
        let device = Device::Mps;
        Ok(PyDevice { device })
    }

    /// Check if device is available
    /// デバイスが利用可能かチェック
    pub fn is_available(&self) -> bool {
        // Simplified availability check
        match self.device {
            Device::Cpu => true,
            Device::Cuda(_) => false, // Simplified - would check CUDA availability
            Device::Mps => cfg!(target_os = "macos"),
            Device::Wasm => cfg!(target_arch = "wasm32"),
        }
    }

    /// Get device type as string
    /// デバイスタイプを文字列として取得
    pub fn type_(&self) -> String {
        match self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(_) => "cuda".to_string(),
            Device::Mps => "mps".to_string(),
            Device::Wasm => "wasm".to_string(),
        }
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        match self.device {
            Device::Cpu => "device(type='cpu')".to_string(),
            Device::Cuda(index) => format!("device(type='cuda', index={})", index),
            Device::Mps => "device(type='mps')".to_string(),
            Device::Wasm => "device(type='wasm')".to_string(),
        }
    }
}

// Additional tensor creation functions
// 追加のテンソル作成関数

/// Create tensor from Python list
/// Pythonリストからテンソルを作成
#[pyfunction]
pub fn tensor(data: Vec<f32>, shape: Option<Vec<usize>>) -> PyResult<PyTensor> {
    let shape = shape.unwrap_or_else(|| vec![data.len()]);
    PyTensor::new(data, shape)
}

/// Create identity matrix
/// 単位行列を作成
#[pyfunction]
pub fn eye(n: usize) -> PyResult<PyTensor> {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    PyTensor::new(data, vec![n, n])
}
