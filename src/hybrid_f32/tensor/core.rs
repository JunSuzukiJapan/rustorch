//! F32Tensor - コア実装
//! F32Tensor - Core implementation

use crate::error::{RusTorchResult, RusTorchError};
use crate::hybrid_f32_experimental;
use ndarray::{Array, IxDyn};
use std::sync::Arc;
use std::ops::{Index, IndexMut};

/// 2次元インデックス
/// 2D index
#[derive(Debug, Clone, Copy)]
pub struct Index2D(pub usize, pub usize);

/// 3次元インデックス
/// 3D index
#[derive(Debug, Clone, Copy)]
pub struct Index3D(pub usize, pub usize, pub usize);

/// デバイス最適化状態
/// Device optimization state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DeviceState {
    CPU,
    Metal { device_id: usize },
    CoreML { device_id: usize },
    Synchronized, // 全デバイス同期済み
}

/// Metal共有バッファ（プレースホルダー）
/// Metal shared buffer (placeholder)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MetalBuffer {
    _device_id: usize,
    _size: usize,
}

impl MetalBuffer {
    pub fn new(device_id: usize, size: usize) -> Self {
        Self {
            _device_id: device_id,
            _size: size,
        }
    }
}

/// CoreML共有バッファ（プレースホルダー）
/// CoreML shared buffer (placeholder)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoreMLBuffer {
    _device_id: usize,
    _shape: Vec<usize>,
}

impl CoreMLBuffer {
    pub fn new(device_id: usize, shape: Vec<usize>) -> Self {
        Self {
            _device_id: device_id,
            _shape: shape,
        }
    }
}

/// f32専用テンソル（変換コスト最小化）
/// f32-specific tensor (conversion cost minimization)
#[derive(Debug)]
pub struct F32Tensor {
    /// CPU側データ
    /// CPU-side data
    pub data: Array<f32, IxDyn>,

    /// GPU共有バッファ（Metal用）
    /// GPU shared buffer (for Metal)
    pub metal_buffer: Option<Arc<MetalBuffer>>,

    /// Neural Engine共有バッファ（CoreML用）
    /// Neural Engine shared buffer (for CoreML)
    pub coreml_buffer: Option<Arc<CoreMLBuffer>>,

    /// デバイス最適化状態
    /// Device optimization state
    pub device_state: DeviceState,

    /// 勾配追跡
    /// Gradient tracking
    pub requires_grad: bool,

    /// テンソル形状
    /// Tensor shape
    shape: Vec<usize>,
}

impl Clone for F32Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            metal_buffer: self.metal_buffer.clone(),
            coreml_buffer: self.coreml_buffer.clone(),
            device_state: self.device_state.clone(),
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// PyTorchライクな演算子オーバーロード
// PyTorch-like operator overloading

use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};

/// Addition operator: tensor + tensor
impl Add<F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, rhs: F32Tensor) -> Self::Output {
        (&self).add(&rhs)
    }
}

/// Addition operator: tensor + &tensor
impl Add<&F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, rhs: &F32Tensor) -> Self::Output {
        (&self).add(rhs)
    }
}

/// Addition operator: &tensor + &tensor
impl Add for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, rhs: &F32Tensor) -> Self::Output {
        self.add(rhs)
    }
}

/// Addition operator: tensor + scalar
impl Add<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, rhs: f32) -> Self::Output {
        let scalar_tensor = F32Tensor::from_scalar(rhs)?;
        self.add(&scalar_tensor)
    }
}

/// Addition operator: &tensor + scalar
impl Add<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, rhs: f32) -> Self::Output {
        let scalar_tensor = F32Tensor::from_scalar(rhs)?;
        self.add(&scalar_tensor)
    }
}

/// Subtraction operator: tensor - tensor
impl Sub<F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, rhs: F32Tensor) -> Self::Output {
        (&self).sub(&rhs)
    }
}

/// Subtraction operator: tensor - &tensor
impl Sub<&F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, rhs: &F32Tensor) -> Self::Output {
        (&self).sub(rhs)
    }
}

/// Subtraction operator: &tensor - &tensor
impl Sub for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, rhs: &F32Tensor) -> Self::Output {
        self.sub(rhs)
    }
}

/// Subtraction operator: tensor - scalar
impl Sub<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, rhs: f32) -> Self::Output {
        let scalar_tensor = F32Tensor::from_scalar(rhs)?;
        self.sub(&scalar_tensor)
    }
}

/// Subtraction operator: &tensor - scalar
impl Sub<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, rhs: f32) -> Self::Output {
        let scalar_tensor = F32Tensor::from_scalar(rhs)?;
        self.sub(&scalar_tensor)
    }
}

/// Multiplication operator: tensor * tensor
impl Mul<F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, rhs: F32Tensor) -> Self::Output {
        (&self).mul(&rhs)
    }
}

/// Multiplication operator: tensor * &tensor
impl Mul<&F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, rhs: &F32Tensor) -> Self::Output {
        (&self).mul(rhs)
    }
}

/// Multiplication operator: &tensor * &tensor
impl Mul for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, rhs: &F32Tensor) -> Self::Output {
        self.mul(rhs)
    }
}

/// Multiplication operator: tensor * scalar
impl Mul<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

/// Multiplication operator: &tensor * scalar
impl Mul<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

/// Division operator: tensor / tensor
impl Div<F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, rhs: F32Tensor) -> Self::Output {
        (&self).divide(&rhs)
    }
}

/// Division operator: tensor / &tensor
impl Div<&F32Tensor> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, rhs: &F32Tensor) -> Self::Output {
        (&self).divide(rhs)
    }
}

/// Division operator: &tensor / &tensor
impl Div for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, rhs: &F32Tensor) -> Self::Output {
        self.divide(rhs)
    }
}

/// Division operator: tensor / scalar
impl Div<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "div".to_string(),
                message: "Division by zero".to_string(),
            });
        }
        self.mul_scalar(1.0 / rhs)
    }
}

/// Division operator: &tensor / scalar
impl Div<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, rhs: f32) -> Self::Output {
        if rhs == 0.0 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "div".to_string(),
                message: "Division by zero".to_string(),
            });
        }
        self.mul_scalar(1.0 / rhs)
    }
}

/// Negation operator: -tensor
impl Neg for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn neg(self) -> Self::Output {
        self.mul_scalar(-1.0)
    }
}

/// Negation operator: -&tensor
impl Neg for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn neg(self) -> Self::Output {
        self.mul_scalar(-1.0)
    }
}
impl F32Tensor {
    /// テンソルデータへのスライスアクセス
    /// Slice access to tensor data
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice().unwrap_or(&[])
    }

    /// テンソルの次元数を取得
    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// テンソルが空かどうかを取得
    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// テンソルがスカラーかどうかを取得
    /// Check if tensor is scalar
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    /// 勾配計算が有効かどうかを取得
    /// Check if gradient computation is enabled
    pub fn is_grad_enabled(&self) -> bool {
        self.requires_grad
    }

    /// 勾配計算を設定
    /// Set gradient computation
    pub fn requires_grad(&mut self, requires: bool) {
        self.requires_grad = requires;
    }

    /// ゼロテンソル作成
    /// Create zero tensor
    pub fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        hybrid_f32_experimental!();
        
        let data = Array::zeros(IxDyn(shape));
        Ok(Self {
            data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: shape.to_vec(),
        })
    }

    /// 正規分布乱数テンソル作成
    /// Create random normal tensor
    pub fn randn(shape: &[usize]) -> RusTorchResult<Self> {
        hybrid_f32_experimental!();
        
        use rand_distr::StandardNormal;
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        
        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| RusTorchError::InvalidParameters {
                operation: "randn".to_string(),
                message: format!("Shape error: {}", e),
            })?;
        
        Ok(Self {
            data: array,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: shape.to_vec(),
        })
    }

    /// スカラー値からテンソル作成
    /// Create tensor from scalar
    pub fn from_scalar(value: f32) -> RusTorchResult<Self> {
        hybrid_f32_experimental!();
        
        let data = Array::from_elem(IxDyn(&[1]), value);
        Ok(Self {
            data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: vec![1],
        })
    }

    /// テンソル形状取得
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// CPU側へのコピー
    /// Copy to CPU
    pub fn to_cpu(&self) -> RusTorchResult<Self> {
        Ok(self.clone())
    }

    /// Metal GPU転送
    /// Transfer to Metal GPU
    pub fn to_metal(&mut self, device_id: usize) -> RusTorchResult<()> {
        hybrid_f32_experimental!();
        
        self.device_state = DeviceState::Metal { device_id };
        self.metal_buffer = Some(Arc::new(MetalBuffer::new(device_id, self.data.len())));
        Ok(())
    }

    /// CoreML Neural Engine転送
    /// Transfer to CoreML Neural Engine
    pub fn to_coreml(&mut self, device_id: usize) -> RusTorchResult<()> {
        hybrid_f32_experimental!();
        
        self.device_state = DeviceState::CoreML { device_id };
        self.coreml_buffer = Some(Arc::new(CoreMLBuffer::new(device_id, self.shape.clone())));
        Ok(())
    }

    /// デバイス状態取得
    /// Get device state
    pub fn device_state(&self) -> &DeviceState {
        &self.device_state
    }

    /// スカラー値取得（1要素テンソルから）
    /// Get scalar value (from 1-element tensor)
    pub fn unwrap(&self) -> RusTorchResult<f32> {
        if self.data.len() == 1 {
            Ok(self.data.iter().next().copied().unwrap_or(0.0))
        } else {
            Err(RusTorchError::InvalidParameters {
                operation: "unwrap".to_string(),
                message: format!("Tensor has {} elements, expected 1", self.data.len()),
            })
        }
    }

    /// 要素ごと加算
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = &self.data + &other.data;
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 要素ごと乗算
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = &self.data * &other.data;
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 行列乗算
    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> RusTorchResult<Self> {
        // 簡単な2Dケースのみ実装（プレースホルダー）
        if self.shape.len() == 2 && other.shape.len() == 2 {
            let (m, k) = (self.shape[0], self.shape[1]);
            let (k2, n) = (other.shape[0], other.shape[1]);
            
            if k != k2 {
                return Err(RusTorchError::InvalidParameters {
                    operation: "matmul".to_string(),
                    message: format!("Incompatible dimensions: {}x{} and {}x{}", m, k, k2, n),
                });
            }

            let result_shape = vec![m, n];
            let mut result_data = vec![0.0f32; m * n];
            
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += self.data[[i, l]] * other.data[[l, j]];
                    }
                    result_data[i * n + j] = sum;
                }
            }
            
            let array = Array::from_shape_vec(IxDyn(&result_shape), result_data)
                .map_err(|e| RusTorchError::InvalidParameters {
                    operation: "matmul".to_string(),
                    message: format!("Shape error: {}", e),
                })?;
            
            Ok(Self {
                data: array,
                metal_buffer: None,
                coreml_buffer: None,
                device_state: DeviceState::CPU,
                requires_grad: self.requires_grad || other.requires_grad,
                shape: result_shape,
            })
        } else {
            Err(RusTorchError::InvalidParameters {
                operation: "matmul".to_string(),
                message: "Only 2D tensors supported".to_string(),
            })
        }
    }

    /// 転置
    /// Transpose
    pub fn transpose(&self) -> RusTorchResult<Self> {
        if self.shape.len() == 2 {
            let transposed = self.data.view().reversed_axes().to_owned();
            let new_shape = vec![self.shape[1], self.shape[0]];
            
            Ok(Self {
                data: transposed,
                metal_buffer: None,
                coreml_buffer: None,
                device_state: DeviceState::CPU,
                requires_grad: self.requires_grad,
                shape: new_shape,
            })
        } else {
            Err(RusTorchError::InvalidParameters {
                operation: "transpose".to_string(),
                message: "Only 2D tensors supported".to_string(),
            })
        }
    }

    /// 要素ごと減算
    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = &self.data - &other.data;
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 数値要素数取得
    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// より大きい要素マスク
    /// Greater than element mask
    pub fn gt(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| if x > other.data.iter().next().copied().unwrap_or(0.0) { 1.0 } else { 0.0 });
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: self.shape.clone(),
        })
    }

    /// より小さいか等しい要素マスク
    /// Less than or equal element mask
    pub fn le(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| if x <= other.data.iter().next().copied().unwrap_or(0.0) { 1.0 } else { 0.0 });
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: self.shape.clone(),
        })
    }

    /// ReLU活性化関数
    /// ReLU activation function
    pub fn relu(&self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.max(0.0));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// Sigmoid活性化関数
    /// Sigmoid activation function
    pub fn sigmoid(&self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// Tanh活性化関数
    /// Tanh activation function
    pub fn tanh(&self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.tanh());
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 指数関数
    /// Exponential function
    pub fn exp(&self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.exp());
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 対数関数
    /// Logarithm function
    pub fn log(&self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.ln());
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// べき乗
    /// Power function
    pub fn power(&self, exponent: f32) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.powf(exponent));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 最大値（要素ごと）
    /// Element-wise maximum
    pub fn maximum(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.max(other.data.iter().next().copied().unwrap_or(0.0)));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 最小値（要素ごと）
    /// Element-wise minimum
    pub fn minimum(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.min(other.data.iter().next().copied().unwrap_or(0.0)));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 値のクランプ
    /// Clamp values
    pub fn clamp(&self, min: f32, max: f32) -> RusTorchResult<Self> {
        let result_data = self.data.mapv(|x| x.max(min).min(max));
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 最大値のインデックス
    /// Index of maximum value
    pub fn argmax(&self) -> RusTorchResult<Self> {
        let max_idx = self.data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as f32)
            .unwrap_or(0.0);
        
        Self::from_scalar(max_idx)
    }

    /// テンソル形状変更
    /// Reshape tensor
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(RusTorchError::InvalidParameters {
                operation: "reshape".to_string(),
                message: format!("Cannot reshape tensor of size {} to size {}", self.data.len(), new_size),
            });
        }

        let reshaped_data = self.data.clone().into_shape(IxDyn(new_shape))
            .map_err(|e| RusTorchError::InvalidParameters {
                operation: "reshape".to_string(),
                message: format!("Reshape error: {}", e),
            })?;

        Ok(Self {
            data: reshaped_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: new_shape.to_vec(),
        })
    }

    /// テンソルスライス（簡易版）
    /// Tensor slice (simple version)
    pub fn slice(&self, ranges: &[(usize, usize)]) -> RusTorchResult<Self> {
        // Simple slice implementation for compatibility
        let mut result_data = Vec::new();
        let shape = self.shape();

        // For simplicity, just extract the first range for 1D case
        if ranges.len() == 1 && self.ndim() == 1 {
            let (start, end) = ranges[0];
            let slice_data = &self.data.as_slice().unwrap()[start..end];
            result_data.extend_from_slice(slice_data);
            F32Tensor::new(result_data, &[end - start])
        } else {
            // For more complex cases, return a clone for now
            Ok(self.clone())
        }
    }

    /// 型変換
    /// Type conversion
    pub fn to_type(&self, _dtype: &str) -> RusTorchResult<Self> {
        // f32から他の型への変換は今回スキップ
        Ok(self.clone())
    }

    /// 除算
    /// Division
    pub fn divide(&self, other: &Self) -> RusTorchResult<Self> {
        let result_data = &self.data / &other.data;
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        })
    }

    /// 平均（テンソル同士）
    /// Mean (tensor-wise)
    pub fn mean_tensor(&self) -> RusTorchResult<Self> {
        let mean_val = self.data.mean().unwrap_or(0.0);
        Self::from_scalar(mean_val)
    }

    /// 次元に沿った合計
    /// Sum along dimension
    pub fn sum_dim(&self, _dim: usize) -> RusTorchResult<Self> {
        let sum_val = self.data.sum();
        Self::from_scalar(sum_val)
    }

    /// データからテンソル作成
    /// Create tensor from vector data
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> RusTorchResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(RusTorchError::InvalidParameters {
                operation: "from_vec".to_string(),
                message: format!("Data length {} doesn't match shape size {}", data.len(), expected_size),
            });
        }

        let array = Array::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| RusTorchError::InvalidParameters {
                operation: "from_vec".to_string(),
                message: format!("Shape error: {}", e),
            })?;

        Ok(Self {
            data: array,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: shape.to_vec(),
        })
    }

    /// 1のテンソル作成
    /// Create ones tensor
    pub fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        let data = Array::ones(IxDyn(shape));
        Ok(Self {
            data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape: shape.to_vec(),
        })
    }

    /// 汎用newメソッド（from_vecの別名）
    /// Generic new method (alias for from_vec)
    pub fn new(data: Vec<f32>, shape: &[usize]) -> RusTorchResult<Self> {
        Self::from_vec(data, shape)
    }

    /// スライスアクセス（autograd用のOption版）
    /// Slice access (Option version for autograd)
    pub fn as_slice_option(&self) -> Option<&[f32]> {
        self.data.as_slice()
    }

    /// スカラー値取得（unwrapの別名）
    /// Get scalar value (alias for unwrap)
    pub fn scalar_value(&self) -> RusTorchResult<f32> {
        self.unwrap()
    }

    // ========================================
    // try_*メソッド群 - エラー処理改善
    // try_* methods - Improved error handling
    // ========================================

    /// 安全なテンソル加算
    /// Safe tensor addition
    pub fn try_add(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.add(other)
    }

    /// 安全なテンソル減算
    /// Safe tensor subtraction
    pub fn try_sub(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.sub(other)
    }

    /// 安全なテンソル乗算
    /// Safe tensor multiplication
    pub fn try_mul(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.mul(other)
    }

    /// 安全なテンソル除算
    /// Safe tensor division
    pub fn try_div(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.divide(other)
    }

    /// 安全な行列乗算
    /// Safe matrix multiplication
    pub fn try_matmul(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        self.matmul(other)
    }

    /// 安全なスカラー乗算
    /// Safe scalar multiplication
    pub fn try_mul_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        if scalar.is_nan() || scalar.is_infinite() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_mul_scalar".to_string(),
                message: format!("Invalid scalar value: {}", scalar),
            });
        }
        self.mul_scalar(scalar)
    }

    /// 安全な形状変更
    /// Safe reshape
    pub fn try_reshape(&self, new_shape: &[usize]) -> RusTorchResult<F32Tensor> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_reshape".to_string(),
                message: format!(
                    "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                    self.numel(), new_shape, new_numel
                ),
            });
        }
        self.reshape(new_shape)
    }

    /// 安全な転置
    /// Safe transpose
    pub fn try_transpose(&self) -> RusTorchResult<F32Tensor> {
        if self.ndim() != 2 {
            return Err(crate::error::RusTorchError::InvalidOperation(
                format!("transpose requires 2D tensor, got {}D", self.ndim())
            ));
        }
        self.transpose()
    }

    /// 安全なスライス
    /// Safe slice
    pub fn try_slice(&self, ranges: &[(usize, usize)]) -> RusTorchResult<F32Tensor> {
        if ranges.len() != self.ndim() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_slice".to_string(),
                message: format!(
                    "Expected {} slice ranges for {}D tensor, got {}",
                    self.ndim(), self.ndim(), ranges.len()
                ),
            });
        }

        let shape = self.shape();
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if start >= end || end > shape[i] {
                return Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "try_slice".to_string(),
                    message: format!(
                        "Invalid slice range for dimension {}: {}..{} (max: {})",
                        i, start, end, shape[i]
                    ),
                });
            }
        }

        self.slice(ranges)
    }

    /// 安全なCPU転送
    /// Safe CPU transfer
    pub fn try_to_cpu(&self) -> RusTorchResult<F32Tensor> {
        self.to_cpu()
    }

    /// 安全なMetal転送
    /// Safe Metal transfer
    pub fn try_to_metal(&mut self, device_id: usize) -> RusTorchResult<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.to_metal(device_id)
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Err(crate::error::RusTorchError::BackendUnavailable {
                backend: "Metal (macOS + metal feature required)".to_string(),
            })
        }
    }

    /// 安全なCoreML転送
    /// Safe CoreML transfer
    pub fn try_to_coreml(&mut self, device_id: usize) -> RusTorchResult<()> {
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        {
            self.to_coreml(device_id)
        }

        #[cfg(not(all(target_os = "macos", feature = "coreml")))]
        {
            Err(crate::error::RusTorchError::BackendUnavailable {
                backend: "CoreML (macOS + coreml feature required)".to_string(),
            })
        }
    }

    /// 安全な型変換
    /// Safe type conversion
    pub fn try_to_type<T>(&self) -> RusTorchResult<Vec<T>>
    where
        T: From<f32> + Copy,
    {
        if self.numel() == 0 {
            return Ok(Vec::new());
        }

        let data = self.data.as_slice().ok_or_else(|| {
            crate::error::RusTorchError::InvalidOperation(
                "Cannot access tensor data".to_string()
            )
        })?;

        Ok(data.iter().map(|&x| T::from(x)).collect())
    }

    /// 安全な要素アクセス
    /// Safe element access
    pub fn try_get(&self, indices: &[usize]) -> RusTorchResult<f32> {
        if indices.len() != self.ndim() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_get".to_string(),
                message: format!(
                    "Expected {} indices for {}D tensor, got {}",
                    self.ndim(), self.ndim(), indices.len()
                ),
            });
        }

        let shape = self.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(crate::error::RusTorchError::index_out_of_bounds(&[idx], &[shape[i]]));
            }
        }

        // 平坦化インデックス計算
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }

        let data = self.data.as_slice().ok_or_else(|| {
            crate::error::RusTorchError::InvalidOperation(
                "Cannot access tensor data".to_string()
            )
        })?;

        Ok(data[flat_index])
    }

    /// 安全な要素設定
    /// Safe element setting
    pub fn try_set(&mut self, indices: &[usize], value: f32) -> RusTorchResult<()> {
        if value.is_nan() || value.is_infinite() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_set".to_string(),
                message: format!("Invalid value: {}", value),
            });
        }

        if indices.len() != self.ndim() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "try_set".to_string(),
                message: format!(
                    "Expected {} indices for {}D tensor, got {}",
                    self.ndim(), self.ndim(), indices.len()
                ),
            });
        }

        let shape = self.shape();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(crate::error::RusTorchError::index_out_of_bounds(&[idx], &[shape[i]]));
            }
        }

        // 平坦化インデックス計算
        let mut flat_index = 0;
        let mut stride = 1;
        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }

        let data = self.data.as_slice_mut().ok_or_else(|| {
            crate::error::RusTorchError::InvalidOperation(
                "Cannot access tensor data for modification".to_string()
            )
        })?;

        data[flat_index] = value;
        Ok(())
    }

    /// 全要素の合計（スカラー）
    /// Sum of all elements (scalar)
    pub fn sum(&self) -> RusTorchResult<f32> {
        Ok(self.data.sum())
    }

    /// スカラー乗算（修正版）
    /// Scalar multiplication (fixed version)
    pub fn mul_scalar(&self, scalar: f32) -> RusTorchResult<Self> {
        let result_data = &self.data * scalar;
        Ok(Self {
            data: result_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        })
    }
}

// Indexing implementations

/// 1D indexing implementation
impl Index<usize> for F32Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data.as_slice().unwrap()[index]
    }
}

impl IndexMut<usize> for F32Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data.as_slice_mut().unwrap()[index]
    }
}

/// 2D indexing implementation
impl Index<Index2D> for F32Tensor {
    type Output = f32;

    fn index(&self, index: Index2D) -> &Self::Output {
        let flat_index = index.0 * self.shape[1] + index.1;
        &self.data.as_slice().unwrap()[flat_index]
    }
}

impl IndexMut<Index2D> for F32Tensor {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        let flat_index = index.0 * self.shape[1] + index.1;
        &mut self.data.as_slice_mut().unwrap()[flat_index]
    }
}

/// 3D indexing implementation
impl Index<Index3D> for F32Tensor {
    type Output = f32;

    fn index(&self, index: Index3D) -> &Self::Output {
        let flat_index = index.0 * (self.shape[1] * self.shape[2]) +
                        index.1 * self.shape[2] +
                        index.2;
        &self.data.as_slice().unwrap()[flat_index]
    }
}

impl IndexMut<Index3D> for F32Tensor {
    fn index_mut(&mut self, index: Index3D) -> &mut Self::Output {
        let flat_index = index.0 * (self.shape[1] * self.shape[2]) +
                        index.1 * self.shape[2] +
                        index.2;
        &mut self.data.as_slice_mut().unwrap()[flat_index]
    }
}
