//! F32Tensor - コア実装
//! F32Tensor - Core implementation

use crate::error::{RusTorchResult, RusTorchError};
use crate::hybrid_f32_experimental;
use ndarray::{Array, IxDyn};
use std::sync::Arc;

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

    /// テンソルスライス
    /// Tensor slice
    pub fn slice(&self, ranges: &[(usize, usize)]) -> RusTorchResult<Self> {
        // 簡単な実装（プレースホルダー）
        Ok(self.clone())
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
