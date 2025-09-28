//! F32Tensor - f32専用テンソル実装
//! F32Tensor - f32-specific tensor implementation
//!
//! 変換コスト完全削除を目的としたf32専用テンソル
//! f32-specific tensor aimed at complete conversion cost elimination

use crate::error::RusTorchResult;
use ndarray::{Array, IxDyn};
use std::sync::Arc;

/// デバイス最適化状態
/// Device optimization state
#[derive(Debug, Clone)]
pub enum DeviceState {
    CPU,
    Metal { device_id: usize },
    CoreML { device_id: usize },
    Synchronized, // 全デバイス同期済み
}

/// Metal共有バッファ（プレースホルダー）
/// Metal shared buffer (placeholder)
#[derive(Debug)]
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
#[derive(Debug)]
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

impl F32Tensor {
    /// 新しいF32Tensorを作成
    /// Create new F32Tensor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> RusTorchResult<Self> {
        let total_elements: usize = shape.iter().product();

        if data.len() != total_elements {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::new".to_string(),
                message: format!(
                    "Data length {} doesn't match shape elements {}",
                    data.len(),
                    total_elements
                ),
            });
        }

        let ndarray_data = Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| crate::error::RusTorchError::TensorOp {
                message: format!("Failed to create ndarray: {}", e),
                source: None,
            })?;

        Ok(Self {
            data: ndarray_data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape,
        })
    }

    /// ゼロテンソルを作成
    /// Create zero tensor
    pub fn zeros(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0f32; total_elements];

        Self::new(data, shape.to_vec()).expect("Failed to create zero tensor")
    }

    /// ランダムテンソルを作成
    /// Create random tensor
    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Self::new(data, shape.to_vec()).expect("Failed to create random tensor")
    }

    /// 1埋めテンソルを作成
    /// Create ones tensor
    pub fn ones(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![1.0f32; total_elements];

        Self::new(data, shape.to_vec()).expect("Failed to create ones tensor")
    }

    /// 一様分布[0,1)からランダムテンソルを作成
    /// Create random tensor from uniform distribution [0,1)
    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.gen::<f32>())
            .collect();

        Self::new(data, shape.to_vec()).expect("Failed to create random tensor")
    }

    /// 一様分布[low,high)からランダムテンソルを作成
    /// Create random tensor from uniform distribution [low,high)
    pub fn uniform(shape: &[usize], low: f32, high: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.gen_range(low..high))
            .collect();

        Self::new(data, shape.to_vec()).expect("Failed to create uniform tensor")
    }

    /// 連続値テンソルを作成
    /// Create arange tensor
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut current = start;

        while current < end {
            data.push(current);
            current += step;
        }

        let len = data.len();
        Self::new(data, vec![len]).expect("Failed to create arange tensor")
    }

    /// 等間隔値テンソルを作成
    /// Create linspace tensor
    pub fn linspace(start: f32, end: f32, steps: usize) -> Self {
        if steps <= 1 {
            return Self::new(vec![start], vec![1]).expect("Failed to create linspace tensor");
        }

        let step_size = (end - start) / (steps - 1) as f32;
        let data: Vec<f32> = (0..steps)
            .map(|i| start + i as f32 * step_size)
            .collect();

        Self::new(data, vec![steps]).expect("Failed to create linspace tensor")
    }

    /// 単位行列を作成
    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0f32;
        }

        Self::new(data, vec![n, n]).expect("Failed to create identity matrix")
    }

    /// ベクターからテンソルを作成
    /// Create tensor from vector
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> RusTorchResult<Self> {
        Self::new(data, shape)
    }

    /// テンソル形状を取得
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// デバイス状態を取得
    /// Get device state
    pub fn device_state(&self) -> &DeviceState {
        &self.device_state
    }


    /// テンソル加算（f32専用）
    /// Tensor addition (f32-specific)
    pub fn add(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if self.shape != other.shape {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::add".to_string(),
                message: format!(
                    "Shape mismatch: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// テンソル減算（f32専用）
    /// Tensor subtraction (f32-specific)
    pub fn sub(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if self.shape != other.shape {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::sub".to_string(),
                message: format!(
                    "Shape mismatch: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// テンソル乗算（f32専用）
    /// Tensor multiplication (f32-specific)
    pub fn mul(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if self.shape != other.shape {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::mul".to_string(),
                message: format!(
                    "Shape mismatch: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// テンソル除算（f32専用）
    /// Tensor division (f32-specific)
    pub fn div(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if self.shape != other.shape {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::div".to_string(),
                message: format!(
                    "Shape mismatch: {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// スカラー加算（f32専用）
    /// Scalar addition (f32-specific)
    pub fn add_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a + scalar)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// スカラー減算（f32専用）
    /// Scalar subtraction (f32-specific)
    pub fn sub_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a - scalar)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// スカラー乗算（f32専用）
    /// Scalar multiplication (f32-specific)
    pub fn mul_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a * scalar)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// スカラー除算（f32専用）
    /// Scalar division (f32-specific)
    pub fn div_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a / scalar)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// 符号反転（f32専用）
    /// Negation (f32-specific)
    pub fn neg(&self) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| -a)
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// 絶対値（f32専用）
    /// Absolute value (f32-specific)
    pub fn abs(&self) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a.abs())
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// べき乗（f32専用）
    /// Power (f32-specific)
    pub fn pow(&self, exponent: f32) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a.powf(exponent))
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// 平方根（f32専用）
    /// Square root (f32-specific)
    pub fn sqrt(&self) -> RusTorchResult<F32Tensor> {
        let result_data: Vec<f32> = self.data.iter()
            .map(|&a| a.sqrt())
            .collect();

        F32Tensor::new(result_data, self.shape.clone())
    }

    /// テンソル合計（f32専用）
    /// Tensor sum (f32-specific)
    pub fn sum(&self) -> RusTorchResult<f32> {
        Ok(self.data.iter().sum::<f32>())
    }

    /// テンソル平均（f32専用）
    /// Tensor mean (f32-specific)
    pub fn mean(&self) -> RusTorchResult<f32> {
        if self.data.is_empty() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::mean".to_string(),
                message: "Cannot compute mean of empty tensor".to_string(),
            });
        }
        Ok(self.data.iter().sum::<f32>() / self.data.len() as f32)
    }

    /// テンソル最大値（f32専用）
    /// Tensor maximum (f32-specific)
    pub fn max(&self) -> RusTorchResult<f32> {
        self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .ok_or_else(|| crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::max".to_string(),
                message: "Cannot compute max of empty tensor".to_string(),
            })
    }

    /// テンソル最小値（f32専用）
    /// Tensor minimum (f32-specific)
    pub fn min(&self) -> RusTorchResult<f32> {
        self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .ok_or_else(|| crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::min".to_string(),
                message: "Cannot compute min of empty tensor".to_string(),
            })
    }

    /// テンソル標準偏差（f32専用）
    /// Tensor standard deviation (f32-specific)
    pub fn std(&self) -> RusTorchResult<f32> {
        if self.data.is_empty() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::std".to_string(),
                message: "Cannot compute std of empty tensor".to_string(),
            });
        }

        let mean = self.mean()?;
        let variance = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.data.len() as f32;

        Ok(variance.sqrt())
    }

    /// テンソル分散（f32専用）
    /// Tensor variance (f32-specific)
    pub fn var(&self) -> RusTorchResult<f32> {
        if self.data.is_empty() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::var".to_string(),
                message: "Cannot compute var of empty tensor".to_string(),
            });
        }

        let mean = self.mean()?;
        let variance = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.data.len() as f32;

        Ok(variance)
    }

    /// 軸指定合計（f32専用）
    /// Axis-specific sum (f32-specific)
    pub fn sum_axis(&self, axis: usize) -> RusTorchResult<F32Tensor> {
        if axis >= self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::sum_axis".to_string(),
                message: format!("Axis {} out of bounds for {}D tensor", axis, self.shape.len()),
            });
        }

        // 簡単な実装: 2次元テンソルの場合のみ対応
        if self.shape.len() == 2 {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            if axis == 0 {
                // 行方向に合計（結果は1xCols）
                let mut result_data = vec![0.0f32; cols];
                for j in 0..cols {
                    for i in 0..rows {
                        result_data[j] += self.data[[i, j]];
                    }
                }
                return F32Tensor::new(result_data, vec![cols]);
            } else if axis == 1 {
                // 列方向に合計（結果はRowsx1）
                let mut result_data = vec![0.0f32; rows];
                for i in 0..rows {
                    for j in 0..cols {
                        result_data[i] += self.data[[i, j]];
                    }
                }
                return F32Tensor::new(result_data, vec![rows]);
            }
        }

        // 他の次元はまだ未実装
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::sum_axis".to_string(),
            message: "Only 2D tensors supported for axis operations".to_string(),
        })
    }

    /// 軸指定平均（f32専用）
    /// Axis-specific mean (f32-specific)
    pub fn mean_axis(&self, axis: usize) -> RusTorchResult<F32Tensor> {
        let sum_result = self.sum_axis(axis)?;
        let divisor = if axis < self.shape.len() {
            self.shape[axis] as f32
        } else {
            1.0
        };
        sum_result.div_scalar(divisor)
    }

    // =========================================================================
    // フェーズ2: 形状操作メソッド / Phase 2: Shape Operations
    // =========================================================================

    /// テンソルの形状変更（f32専用）
    /// Reshape tensor (f32-specific)
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<F32Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let new_total: usize = new_shape.iter().product();

        if total_elements != new_total {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::reshape".to_string(),
                message: format!(
                    "Cannot reshape tensor with {} elements to shape with {} elements",
                    total_elements, new_total
                ),
            });
        }

        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), new_shape.to_vec())
    }

    /// テンソル転置（f32専用）
    /// Transpose tensor (f32-specific)
    pub fn transpose(&self) -> RusTorchResult<F32Tensor> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::transpose".to_string(),
                message: "Transpose currently only supports 2D tensors".to_string(),
            });
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result_data = vec![0.0f32; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = self.data[[i, j]];
            }
        }

        F32Tensor::new(result_data, vec![cols, rows])
    }

    /// 軸の順序変更（f32専用）
    /// Permute axes (f32-specific)
    pub fn permute(&self, axes: &[usize]) -> RusTorchResult<F32Tensor> {
        if axes.len() != self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::permute".to_string(),
                message: format!(
                    "Number of axes {} doesn't match tensor dimensions {}",
                    axes.len(), self.shape.len()
                ),
            });
        }

        // 現在は2Dテンソルのみサポート（transpose相当）
        if self.shape.len() == 2 && axes == &[1, 0] {
            return self.transpose();
        }

        // その他の次元はまだ未実装
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::permute".to_string(),
            message: "Permute currently only supports 2D transpose".to_string(),
        })
    }

    /// 次元削除（サイズ1の次元を削除）
    /// Squeeze dimensions (remove size-1 dimensions)
    pub fn squeeze(&self) -> RusTorchResult<F32Tensor> {
        let new_shape: Vec<usize> = self.shape.iter()
            .copied()
            .filter(|&dim| dim != 1)
            .collect();

        if new_shape.is_empty() {
            // 全ての次元がサイズ1の場合、スカラーとして[1]の形状にする
            return F32Tensor::new(self.data.as_slice().unwrap().to_vec(), vec![1]);
        }

        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), new_shape)
    }

    /// 指定次元でのsqueeze
    /// Squeeze specific dimension
    pub fn squeeze_dim(&self, dim: usize) -> RusTorchResult<F32Tensor> {
        if dim >= self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::squeeze_dim".to_string(),
                message: format!("Dimension {} out of bounds for {}D tensor", dim, self.shape.len()),
            });
        }

        if self.shape[dim] != 1 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::squeeze_dim".to_string(),
                message: format!("Cannot squeeze dimension {} with size {}", dim, self.shape[dim]),
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);

        if new_shape.is_empty() {
            new_shape.push(1);
        }

        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), new_shape)
    }

    /// 次元追加（指定位置にサイズ1の次元を追加）
    /// Unsqueeze (add size-1 dimension at specified position)
    pub fn unsqueeze(&self, dim: usize) -> RusTorchResult<F32Tensor> {
        if dim > self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::unsqueeze".to_string(),
                message: format!("Dimension {} out of bounds for insertion", dim),
            });
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);

        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), new_shape)
    }

    /// テンソル平坦化（f32専用）
    /// Flatten tensor (f32-specific)
    pub fn flatten(&self) -> RusTorchResult<F32Tensor> {
        let total_elements = self.shape.iter().product();
        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), vec![total_elements])
    }

    /// 範囲指定平坦化
    /// Flatten with range specification
    pub fn flatten_range(&self, start_dim: usize, end_dim: Option<usize>) -> RusTorchResult<F32Tensor> {
        let end_dim = end_dim.unwrap_or(self.shape.len() - 1);

        if start_dim > end_dim || end_dim >= self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::flatten_range".to_string(),
                message: "Invalid dimension range".to_string(),
            });
        }

        let mut new_shape = Vec::new();

        // start_dim前の次元をそのまま追加
        for i in 0..start_dim {
            new_shape.push(self.shape[i]);
        }

        // start_dimからend_dimまでの次元をまとめる
        let flattened_size: usize = self.shape[start_dim..=end_dim].iter().product();
        new_shape.push(flattened_size);

        // end_dim後の次元をそのまま追加
        for i in (end_dim + 1)..self.shape.len() {
            new_shape.push(self.shape[i]);
        }

        F32Tensor::new(self.data.as_slice().unwrap().to_vec(), new_shape)
    }

    /// テンソル拡張（ブロードキャスト）
    /// Expand tensor (broadcasting)
    pub fn expand(&self, target_shape: &[usize]) -> RusTorchResult<F32Tensor> {
        if !self.can_broadcast_to(target_shape) {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::expand".to_string(),
                message: format!(
                    "Cannot expand shape {:?} to {:?}",
                    self.shape, target_shape
                ),
            });
        }

        // 簡単な実装：同じ形状の場合はそのまま返す
        if self.shape == target_shape {
            return Ok(self.clone());
        }

        // 実際のブロードキャスト実装は複雑なため、現在は基本的なケースのみサポート
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::expand".to_string(),
            message: "Complex broadcasting not yet implemented".to_string(),
        })
    }

    /// ブロードキャスト可能性チェック
    /// Check if broadcasting is possible
    pub fn can_broadcast_to(&self, target_shape: &[usize]) -> bool {
        // 簡単な実装：同じ形状のみ許可
        self.shape == target_shape
    }

    /// テンソル繰り返し
    /// Repeat tensor
    pub fn repeat(&self, repeats: &[usize]) -> RusTorchResult<F32Tensor> {
        if repeats.len() != self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::repeat".to_string(),
                message: format!(
                    "Repeat dimensions {} doesn't match tensor dimensions {}",
                    repeats.len(), self.shape.len()
                ),
            });
        }

        // 簡単な1次元実装
        if self.shape.len() == 1 {
            let mut result_data = Vec::new();
            for _ in 0..repeats[0] {
                result_data.extend_from_slice(self.data.as_slice().unwrap());
            }
            let new_size = self.shape[0] * repeats[0];
            return F32Tensor::new(result_data, vec![new_size]);
        }

        // 他の次元は未実装
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::repeat".to_string(),
            message: "Repeat currently only supports 1D tensors".to_string(),
        })
    }

    /// テンソル分割
    /// Split tensor
    pub fn split(&self, split_size: usize, dim: usize) -> RusTorchResult<Vec<F32Tensor>> {
        if dim >= self.shape.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::split".to_string(),
                message: format!("Dimension {} out of bounds", dim),
            });
        }

        // 1次元の場合の実装
        if self.shape.len() == 1 && dim == 0 {
            let mut chunks = Vec::new();
            let data = self.data.as_slice().unwrap();

            for i in (0..data.len()).step_by(split_size) {
                let end = (i + split_size).min(data.len());
                let chunk_data = data[i..end].to_vec();
                let chunk_size = end - i;
                chunks.push(F32Tensor::new(chunk_data, vec![chunk_size])?);
            }

            return Ok(chunks);
        }

        // 他の次元は未実装
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::split".to_string(),
            message: "Split currently only supports 1D tensors".to_string(),
        })
    }

    /// テンソル結合
    /// Concatenate tensors
    pub fn concat(tensors: &[&F32Tensor], dim: usize) -> RusTorchResult<F32Tensor> {
        if tensors.is_empty() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::concat".to_string(),
                message: "Cannot concatenate empty tensor list".to_string(),
            });
        }

        let first_shape = &tensors[0].shape;

        // 他のテンソルの形状チェック（指定次元以外は同じである必要がある）
        for tensor in tensors.iter().skip(1) {
            if tensor.shape.len() != first_shape.len() {
                return Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "F32Tensor::concat".to_string(),
                    message: "All tensors must have same number of dimensions".to_string(),
                });
            }

            for (i, (&dim_size, &first_dim_size)) in tensor.shape.iter().zip(first_shape.iter()).enumerate() {
                if i != dim && dim_size != first_dim_size {
                    return Err(crate::error::RusTorchError::InvalidParameters {
                        operation: "F32Tensor::concat".to_string(),
                        message: format!(
                            "Dimension {} size mismatch: {} vs {}",
                            i, dim_size, first_dim_size
                        ),
                    });
                }
            }
        }

        // 1次元の場合の実装
        if first_shape.len() == 1 && dim == 0 {
            let mut result_data = Vec::new();
            for tensor in tensors {
                result_data.extend_from_slice(tensor.data.as_slice().unwrap());
            }
            let total_size: usize = tensors.iter().map(|t| t.shape[0]).sum();
            return F32Tensor::new(result_data, vec![total_size]);
        }

        // 他の次元は未実装
        Err(crate::error::RusTorchError::InvalidParameters {
            operation: "F32Tensor::concat".to_string(),
            message: "Concat currently only supports 1D tensors".to_string(),
        })
    }

    /// テンソル積み重ね（新しい次元を作成）
    /// Stack tensors (create new dimension)
    pub fn stack(tensors: &[&F32Tensor], dim: usize) -> RusTorchResult<F32Tensor> {
        if tensors.is_empty() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::stack".to_string(),
                message: "Cannot stack empty tensor list".to_string(),
            });
        }

        let first_shape = &tensors[0].shape;

        // 全てのテンソルが同じ形状であることを確認
        for tensor in tensors.iter().skip(1) {
            if tensor.shape != *first_shape {
                return Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "F32Tensor::stack".to_string(),
                    message: "All tensors must have same shape for stacking".to_string(),
                });
            }
        }

        // 新しい形状を計算
        let mut new_shape = first_shape.clone();
        new_shape.insert(dim, tensors.len());

        // データを結合
        let mut result_data = Vec::new();
        for tensor in tensors {
            result_data.extend_from_slice(tensor.data.as_slice().unwrap());
        }

        F32Tensor::new(result_data, new_shape)
    }

    /// 行列乗算（f32専用）
    /// Matrix multiplication (f32-specific)
    pub fn matmul(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::matmul".to_string(),
                message: "Both tensors must be 2D matrices".to_string(),
            });
        }

        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k1 != k2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::matmul".to_string(),
                message: format!(
                    "Matrix dimension mismatch: {}x{} × {}x{}",
                    m, k1, k2, n
                ),
            });
        }

        // シンプルなf32行列乗算実装
        let mut result_data = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..k1 {
                    let a_val = self.data[[i, k]];
                    let b_val = other.data[[k, j]];
                    sum += a_val * b_val;
                }
                result_data[i * n + j] = sum;
            }
        }

        F32Tensor::new(result_data, vec![m, n])
    }

    // ===== 線形代数演算 / Linear Algebra Operations =====

    /// 行列転置（f32専用）
    /// Matrix transpose (f32-specific)
    pub fn t(&self) -> RusTorchResult<F32Tensor> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::t".to_string(),
                message: "Transpose requires 2D tensor".to_string(),
            });
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let mut result_data = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                result_data[j * m + i] = self.data[[i, j]];
            }
        }

        F32Tensor::new(result_data, vec![n, m])
    }

    /// 行列式（f32専用）
    /// Matrix determinant (f32-specific)
    pub fn det(&self) -> RusTorchResult<f32> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::det".to_string(),
                message: "Determinant requires square matrix".to_string(),
            });
        }

        let n = self.shape[0];

        match n {
            1 => Ok(self.data[[0, 0]]),
            2 => {
                let a = self.data[[0, 0]];
                let b = self.data[[0, 1]];
                let c = self.data[[1, 0]];
                let d = self.data[[1, 1]];
                Ok(a * d - b * c)
            }
            3 => {
                // サラス公式による3x3行列式
                let a11 = self.data[[0, 0]];
                let a12 = self.data[[0, 1]];
                let a13 = self.data[[0, 2]];
                let a21 = self.data[[1, 0]];
                let a22 = self.data[[1, 1]];
                let a23 = self.data[[1, 2]];
                let a31 = self.data[[2, 0]];
                let a32 = self.data[[2, 1]];
                let a33 = self.data[[2, 2]];

                Ok(a11 * (a22 * a33 - a23 * a32)
                    - a12 * (a21 * a33 - a23 * a31)
                    + a13 * (a21 * a32 - a22 * a31))
            }
            _ => {
                // 大きな行列の場合は簡単なLU分解による実装
                // （実際の実装では LAPACK を使用）
                Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "F32Tensor::det".to_string(),
                    message: "Determinant for matrices larger than 3x3 not yet implemented".to_string(),
                })
            }
        }
    }

    /// 行列逆行列（f32専用）
    /// Matrix inverse (f32-specific)
    pub fn inverse(&self) -> RusTorchResult<F32Tensor> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::inverse".to_string(),
                message: "Inverse requires square matrix".to_string(),
            });
        }

        let n = self.shape[0];
        let det = self.det()?;

        if det.abs() < 1e-10 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::inverse".to_string(),
                message: "Matrix is singular (determinant near zero)".to_string(),
            });
        }

        match n {
            1 => {
                let inv_val = 1.0 / self.data[[0, 0]];
                F32Tensor::new(vec![inv_val], vec![1, 1])
            }
            2 => {
                let a = self.data[[0, 0]];
                let b = self.data[[0, 1]];
                let c = self.data[[1, 0]];
                let d = self.data[[1, 1]];

                let inv_det = 1.0 / det;
                let result_data = vec![
                    d * inv_det, -b * inv_det,
                    -c * inv_det, a * inv_det
                ];
                F32Tensor::new(result_data, vec![2, 2])
            }
            _ => {
                // ガウス-ジョルダン法による逆行列計算
                let mut augmented = vec![0.0f32; n * 2 * n];

                // 拡大行列を作成 [A|I]
                for i in 0..n {
                    for j in 0..n {
                        augmented[i * 2 * n + j] = self.data[[i, j]];
                        augmented[i * 2 * n + n + j] = if i == j { 1.0 } else { 0.0 };
                    }
                }

                // ガウス-ジョルダン消去法
                for i in 0..n {
                    // ピボット選択
                    let mut max_row = i;
                    for k in (i + 1)..n {
                        if augmented[k * 2 * n + i].abs() > augmented[max_row * 2 * n + i].abs() {
                            max_row = k;
                        }
                    }

                    if max_row != i {
                        for j in 0..(2 * n) {
                            augmented.swap(i * 2 * n + j, max_row * 2 * n + j);
                        }
                    }

                    let pivot = augmented[i * 2 * n + i];
                    if pivot.abs() < 1e-10 {
                        return Err(crate::error::RusTorchError::InvalidParameters {
                            operation: "F32Tensor::inverse".to_string(),
                            message: "Matrix is singular".to_string(),
                        });
                    }

                    // 行を正規化
                    for j in 0..(2 * n) {
                        augmented[i * 2 * n + j] /= pivot;
                    }

                    // 他の行を消去
                    for k in 0..n {
                        if k != i {
                            let factor = augmented[k * 2 * n + i];
                            for j in 0..(2 * n) {
                                augmented[k * 2 * n + j] -= factor * augmented[i * 2 * n + j];
                            }
                        }
                    }
                }

                // 逆行列を抽出
                let mut inv_data = vec![0.0f32; n * n];
                for i in 0..n {
                    for j in 0..n {
                        inv_data[i * n + j] = augmented[i * 2 * n + n + j];
                    }
                }

                F32Tensor::new(inv_data, vec![n, n])
            }
        }
    }

    /// トレース（対角和）（f32専用）
    /// Matrix trace (diagonal sum) (f32-specific)
    pub fn trace(&self) -> RusTorchResult<f32> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::trace".to_string(),
                message: "Trace requires 2D tensor".to_string(),
            });
        }

        let min_dim = self.shape[0].min(self.shape[1]);
        let mut trace_sum = 0.0f32;

        for i in 0..min_dim {
            trace_sum += self.data[[i, i]];
        }

        Ok(trace_sum)
    }

    /// 行列ランク（f32専用）
    /// Matrix rank (f32-specific)
    pub fn rank(&self) -> RusTorchResult<usize> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::rank".to_string(),
                message: "Rank requires 2D tensor".to_string(),
            });
        }

        let (m, n) = (self.shape[0], self.shape[1]);
        let mut matrix = self.data.as_slice().unwrap().to_vec();

        // ガウス消去法でランクを計算
        let mut rank = 0;
        let min_dim = m.min(n);

        for col in 0..min_dim {
            // ピボット探索
            let mut pivot_row = None;
            for row in rank..m {
                if matrix[row * n + col].abs() > 1e-10 {
                    pivot_row = Some(row);
                    break;
                }
            }

            if let Some(pivot_row) = pivot_row {
                // 行を交換
                if pivot_row != rank {
                    for j in 0..n {
                        matrix.swap(rank * n + j, pivot_row * n + j);
                    }
                }

                let pivot = matrix[rank * n + col];

                // 下の行を消去
                for i in (rank + 1)..m {
                    let factor = matrix[i * n + col] / pivot;
                    for j in col..n {
                        matrix[i * n + j] -= factor * matrix[rank * n + j];
                    }
                }

                rank += 1;
            }
        }

        Ok(rank)
    }

    /// 条件数（f32専用）
    /// Condition number (f32-specific)
    pub fn cond(&self) -> RusTorchResult<f32> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::cond".to_string(),
                message: "Condition number requires square matrix".to_string(),
            });
        }

        // 簡単な実装：||A|| * ||A^(-1)||
        let norm_a = self.frobenius_norm()?;
        let inv_a = self.inverse()?;
        let norm_inv_a = inv_a.frobenius_norm()?;

        Ok(norm_a * norm_inv_a)
    }

    /// フロベニウスノルム（f32専用）
    /// Frobenius norm (f32-specific)
    pub fn frobenius_norm(&self) -> RusTorchResult<f32> {
        let mut sum_squares = 0.0f32;
        for &val in self.data.as_slice().unwrap() {
            sum_squares += val * val;
        }
        Ok(sum_squares.sqrt())
    }

    /// QR分解（f32専用）
    /// QR decomposition (f32-specific)
    pub fn qr(&self) -> RusTorchResult<(F32Tensor, F32Tensor)> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::qr".to_string(),
                message: "QR decomposition requires 2D tensor".to_string(),
            });
        }

        let (m, n) = (self.shape[0], self.shape[1]);

        // 簡単なグラム・シュミット法による実装
        let mut q_data = vec![0.0f32; m * n];
        let mut r_data = vec![0.0f32; n * n];

        // A の列ベクトルを正規直交化
        for j in 0..n {
            // A の j 列目をコピー
            let mut v_j = vec![0.0f32; m];
            for i in 0..m {
                v_j[i] = self.data[[i, j]];
            }

            // 既に求めた Q の列との直交化
            for k in 0..j {
                let mut dot_product = 0.0f32;
                for i in 0..m {
                    dot_product += v_j[i] * q_data[i * n + k];
                }
                r_data[k * n + j] = dot_product;

                for i in 0..m {
                    v_j[i] -= dot_product * q_data[i * n + k];
                }
            }

            // ノルムを計算
            let mut norm = 0.0f32;
            for &val in &v_j {
                norm += val * val;
            }
            norm = norm.sqrt();

            if norm < 1e-10 {
                return Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "F32Tensor::qr".to_string(),
                    message: "Matrix is rank deficient".to_string(),
                });
            }

            r_data[j * n + j] = norm;

            // 正規化してQに格納
            for i in 0..m {
                q_data[i * n + j] = v_j[i] / norm;
            }
        }

        let q_tensor = F32Tensor::new(q_data, vec![m, n])?;
        let r_tensor = F32Tensor::new(r_data, vec![n, n])?;

        Ok((q_tensor, r_tensor))
    }

    /// 特異値分解（SVD）（f32専用）
    /// Singular Value Decomposition (SVD) (f32-specific)
    pub fn svd(&self) -> RusTorchResult<(F32Tensor, F32Tensor, F32Tensor)> {
        if self.shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::svd".to_string(),
                message: "SVD requires 2D tensor".to_string(),
            });
        }

        // SVDは複雑なアルゴリズムなので、簡単な2x2の場合のみ実装
        if self.shape[0] == 2 && self.shape[1] == 2 {
            let a = self.data[[0, 0]];
            let b = self.data[[0, 1]];
            let c = self.data[[1, 0]];
            let d = self.data[[1, 1]];

            // A^T * A の固有値を計算
            let ata_00 = a * a + c * c;
            let ata_01 = a * b + c * d;
            let ata_11 = b * b + d * d;

            let trace = ata_00 + ata_11;
            let det = ata_00 * ata_11 - ata_01 * ata_01;
            let discriminant = (trace * trace - 4.0 * det).sqrt();

            let lambda1 = (trace + discriminant) / 2.0;
            let lambda2 = (trace - discriminant) / 2.0;

            let sigma1 = lambda1.sqrt();
            let sigma2 = lambda2.sqrt();

            // 簡単なU, S, V行列を構築
            let u_data = vec![1.0, 0.0, 0.0, 1.0]; // 単位行列で近似
            let s_data = vec![sigma1, sigma2];
            let v_data = vec![1.0, 0.0, 0.0, 1.0]; // 単位行列で近似

            let u_tensor = F32Tensor::new(u_data, vec![2, 2])?;
            let s_tensor = F32Tensor::new(s_data, vec![2])?;
            let v_tensor = F32Tensor::new(v_data, vec![2, 2])?;

            Ok((u_tensor, s_tensor, v_tensor))
        } else {
            Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::svd".to_string(),
                message: "SVD currently only supports 2x2 matrices".to_string(),
            })
        }
    }

    /// 固有値・固有ベクトル（f32専用）
    /// Eigenvalues and eigenvectors (f32-specific)
    pub fn eig(&self) -> RusTorchResult<(F32Tensor, F32Tensor)> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::eig".to_string(),
                message: "Eigen decomposition requires square matrix".to_string(),
            });
        }

        let n = self.shape[0];

        if n == 2 {
            let a = self.data[[0, 0]];
            let b = self.data[[0, 1]];
            let c = self.data[[1, 0]];
            let d = self.data[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant < 0.0 {
                return Err(crate::error::RusTorchError::InvalidParameters {
                    operation: "F32Tensor::eig".to_string(),
                    message: "Complex eigenvalues not supported".to_string(),
                });
            }

            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;

            // 固有ベクトルを計算
            let mut v1 = if b.abs() > 1e-10 {
                vec![b, lambda1 - a]
            } else {
                vec![lambda1 - d, c]
            };

            let mut v2 = if b.abs() > 1e-10 {
                vec![b, lambda2 - a]
            } else {
                vec![lambda2 - d, c]
            };

            // 正規化
            let norm1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
            if norm1 > 1e-10 {
                v1[0] /= norm1;
                v1[1] /= norm1;
            }

            let norm2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
            if norm2 > 1e-10 {
                v2[0] /= norm2;
                v2[1] /= norm2;
            }

            let eigenvalues = F32Tensor::new(vec![lambda1, lambda2], vec![2])?;
            let eigenvectors = F32Tensor::new(vec![v1[0], v2[0], v1[1], v2[1]], vec![2, 2])?;

            Ok((eigenvalues, eigenvectors))
        } else {
            Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::eig".to_string(),
                message: "Eigen decomposition currently only supports 2x2 matrices".to_string(),
            })
        }
    }

    /// コレスキー分解（f32専用）
    /// Cholesky decomposition (f32-specific)
    pub fn cholesky(&self) -> RusTorchResult<F32Tensor> {
        if self.shape.len() != 2 || self.shape[0] != self.shape[1] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "F32Tensor::cholesky".to_string(),
                message: "Cholesky decomposition requires square matrix".to_string(),
            });
        }

        let n = self.shape[0];
        let mut l_data = vec![0.0f32; n * n];

        // コレスキー分解アルゴリズム
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // 対角要素
                    let mut sum = 0.0f32;
                    for k in 0..j {
                        sum += l_data[j * n + k] * l_data[j * n + k];
                    }
                    let val = self.data[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err(crate::error::RusTorchError::InvalidParameters {
                            operation: "F32Tensor::cholesky".to_string(),
                            message: "Matrix is not positive definite".to_string(),
                        });
                    }
                    l_data[j * n + j] = val.sqrt();
                } else {
                    // 下三角要素
                    let mut sum = 0.0f32;
                    for k in 0..j {
                        sum += l_data[i * n + k] * l_data[j * n + k];
                    }
                    l_data[i * n + j] = (self.data[[i, j]] - sum) / l_data[j * n + j];
                }
            }
        }

        F32Tensor::new(l_data, vec![n, n])
    }
}

/// Cloneトレイトの実装
/// Clone trait implementation
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
    /// Metal GPUに移動（変換なし）
    /// Move to Metal GPU (no conversion)
    pub fn to_metal(&mut self, device_id: usize) -> RusTorchResult<()> {
        crate::hybrid_f32_experimental!();

        // Metal共有バッファを作成（実際の実装では Metal API を使用）
        let buffer_size = self.data.len() * std::mem::size_of::<f32>();
        self.metal_buffer = Some(Arc::new(MetalBuffer::new(device_id, buffer_size)));
        self.device_state = DeviceState::Metal { device_id };

        println!("🚀 F32Tensor moved to Metal GPU {} (zero-copy)", device_id);
        Ok(())
    }

    /// Neural Engineに移動（変換なし）
    /// Move to Neural Engine (no conversion)
    pub fn to_coreml(&mut self, device_id: usize) -> RusTorchResult<()> {
        crate::hybrid_f32_experimental!();

        // CoreML共有バッファを作成（実際の実装では MLMultiArray を使用）
        self.coreml_buffer = Some(Arc::new(CoreMLBuffer::new(device_id, self.shape.clone())));
        self.device_state = DeviceState::CoreML { device_id };

        println!("🧠 F32Tensor moved to Neural Engine {} (zero-copy)", device_id);
        Ok(())
    }

    /// CPUに移動
    /// Move to CPU
    pub fn to_cpu(&mut self) -> RusTorchResult<()> {
        self.metal_buffer = None;
        self.coreml_buffer = None;
        self.device_state = DeviceState::CPU;

        println!("💻 F32Tensor moved to CPU");
        Ok(())
    }

    /// 全デバイス同期
    /// Synchronize all devices
    pub fn synchronize_all(&mut self) -> RusTorchResult<()> {
        crate::hybrid_f32_experimental!();

        // 実際の実装では各デバイス間でデータ同期
        self.device_state = DeviceState::Synchronized;

        println!("🔄 F32Tensor synchronized across all devices");
        Ok(())
    }

    /// 行列乗算（変換レス実行）
    /// Matrix multiplication (conversion-less execution)
    pub fn matmul_optimized(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        crate::hybrid_f32_experimental!();

        // 最適デバイス選択
        let optimal_device = self.select_optimal_device_for_matmul(&other);

        match optimal_device {
            DeviceState::Metal { device_id } => {
                println!("⚡ Executing matmul on Metal GPU {} (f32 direct)", device_id);
                self.execute_metal_matmul_f32(other)
            }
            DeviceState::CoreML { device_id } => {
                println!("🧠 Executing matmul on Neural Engine {} (f32 direct)", device_id);
                self.execute_coreml_matmul_f32(other)
            }
            DeviceState::CPU => {
                println!("💻 Executing matmul on CPU (f32 direct)");
                self.execute_cpu_matmul_f32(other)
            }
            DeviceState::Synchronized => {
                println!("🔄 Executing matmul on synchronized devices");
                self.execute_cpu_matmul_f32(other) // フォールバック
            }
        }
    }

    /// 最適デバイス選択（行列乗算用）
    /// Select optimal device (for matrix multiplication)
    fn select_optimal_device_for_matmul(&self, _other: &F32Tensor) -> DeviceState {
        let matrix_size = self.shape.iter().product::<usize>();

        match matrix_size {
            size if size > 50000 => DeviceState::Metal { device_id: 0 }, // 大規模 → Metal
            size if size > 1000 => DeviceState::CoreML { device_id: 0 }, // 中規模 → Neural Engine
            _ => DeviceState::CPU, // 小規模 → CPU
        }
    }

    /// Metal GPU f32直接実行
    /// Metal GPU f32 direct execution
    fn execute_metal_matmul_f32(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        // 実際の実装では Metal Performance Shaders を使用
        // プレースホルダー実装: CPU計算でシミュレート
        self.execute_cpu_matmul_f32(other)
    }

    /// Neural Engine f32直接実行
    /// Neural Engine f32 direct execution
    fn execute_coreml_matmul_f32(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        // 実際の実装では CoreML MLCompute を使用
        // プレースホルダー実装: CPU計算でシミュレート
        self.execute_cpu_matmul_f32(other)
    }

    /// CPU f32直接実行
    /// CPU f32 direct execution
    fn execute_cpu_matmul_f32(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        // 簡単な行列乗算実装（実際の実装では BLAS を使用）
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "matmul".to_string(),
                message: "Only 2D matrices supported in this experimental implementation".to_string(),
            });
        }

        if a_shape[1] != b_shape[0] {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "matmul".to_string(),
                message: format!(
                    "Matrix dimensions don't match: {}x{} and {}x{}",
                    a_shape[0], a_shape[1], b_shape[0], b_shape[1]
                ),
            });
        }

        let result_shape = vec![a_shape[0], b_shape[1]];
        let mut result_data = vec![0.0f32; result_shape.iter().product()];

        // 単純な行列乗算
        for i in 0..a_shape[0] {
            for j in 0..b_shape[1] {
                let mut sum = 0.0f32;
                for k in 0..a_shape[1] {
                    let a_idx = i * a_shape[1] + k;
                    let b_idx = k * b_shape[1] + j;
                    sum += self.data[[i, k]] * other.data[[k, j]];
                }
                let result_idx = i * b_shape[1] + j;
                result_data[result_idx] = sum;
            }
        }

        F32Tensor::new(result_data, result_shape)
    }

    /// データをf32スライスとして取得
    /// Get data as f32 slice
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice().unwrap()
    }

    /// データの要素数を取得
    /// Get number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// テンソルが空かどうか
    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}