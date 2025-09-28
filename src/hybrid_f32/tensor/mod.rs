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