//! Custom GPU kernels for specialized tensor operations
//! 特殊テンソル演算用のカスタムGPUカーネル

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::DeviceType;
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Custom kernel types for specialized operations
/// 特殊演算用のカスタムカーネルタイプ
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CustomKernelType {
    /// Optimized convolution kernel
    /// 最適化畳み込みカーネル
    OptimizedConvolution,
    /// Fast Fourier Transform kernel
    /// 高速フーリエ変換カーネル
    FastFourierTransform,
    /// Attention mechanism kernel
    /// アテンション機構カーネル
    AttentionKernel,
    /// Batch normalization kernel
    /// バッチ正規化カーネル
    BatchNormalization,
    /// Custom activation functions
    /// カスタム活性化関数
    CustomActivation(String),
    /// Memory-optimized reduction
    /// メモリ最適化リダクション
    OptimizedReduction,
    /// Sparse matrix operations
    /// スパース行列演算
    SparseOperations,
    /// Tensor fusion operations
    /// テンソル融合演算
    TensorFusion,
}

/// Custom kernel configuration
/// カスタムカーネル設定#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Type of custom kernel to execute
    /// 実行するカスタムカーネルのタイプ
    pub kernel_type: CustomKernelType,
    /// Block size for kernel execution (x, y, z)
    /// カーネル実行のブロックサイズ（x, y, z）
    pub block_size: (u32, u32, u32),
    /// Grid size for kernel execution (x, y, z)
    /// カーネル実行のグリッドサイズ（x, y, z）
    pub grid_size: (u32, u32, u32),
    /// Shared memory size in bytes
    /// 共有メモリサイズ（バイト）
    pub shared_memory_size: usize,
    /// Kernel parameters
    /// カーネルパラメータ
    pub parameters: HashMap<String, KernelParameter>,
}

/// Kernel parameter types
/// カーネルパラメータタイプ#[derive(Debug, Clone)]
pub enum KernelParameter {
    /// Integer parameter
    /// 整数パラメータ
    Int(i32),
    /// Float parameter
    /// 浮動小数点パラメータ
    Float(f32),
    /// Boolean parameter
    /// 真偽値パラメータ
    Bool(bool),
    /// String parameter
    /// 文字列パラメータ
    String(String),
    /// Integer array parameter
    /// 整数配列パラメータ
    IntArray(Vec<i32>),
    /// Float array parameter
    /// 浮動小数点配列パラメータ
    FloatArray(Vec<f32>),
}

/// Custom kernel manager
/// カスタムカーネルマネージャー
pub struct CustomKernelManager {
    compiled_kernels: Arc<Mutex<HashMap<CustomKernelType, CompiledKernel>>>,
    device_type: DeviceType,
}

/// Compiled kernel representation
/// コンパイル済みカーネル表現
#[derive(Debug)]
pub struct CompiledKernel {
    /// Type of the compiled kernel
    /// コンパイル済みカーネルのタイプ
    pub kernel_type: CustomKernelType,
    /// Source code of the kernel
    /// カーネルのソースコード
    pub source_code: String,
    /// Compiled binary data
    /// コンパイル済みバイナリデータ
    pub binary_data: Vec<u8>,
    /// Kernel entry point function name
    /// カーネルエントリポイント関数名
    pub entry_point: String,
    /// Time when the kernel was compiled
    /// カーネルがコンパイルされた時刻
    pub compilation_time: std::time::Instant,
}

impl CustomKernelManager {
    /// Create new custom kernel manager
    /// 新しいカスタムカーネルマネージャーを作成
    pub fn new(device_type: DeviceType) -> Self {
        Self {
            compiled_kernels: Arc::new(Mutex::new(HashMap::new())),
            device_type,
        }
    }

    /// Compile and cache custom kernel
    /// カスタムカーネルをコンパイルしてキャッシュ
    pub fn compile_kernel(&self, config: &KernelConfig) -> RusTorchResult<()> {
        let source_code = self.generate_kernel_source(config)?;
        let binary_data = self.compile_source(&source_code, config)?;

        let compiled_kernel = CompiledKernel {
            kernel_type: config.kernel_type.clone(),
            source_code,
            binary_data,
            entry_point: self.get_entry_point(&config.kernel_type),
            compilation_time: std::time::Instant::now(),
        };

        let mut kernels = self
            .compiled_kernels
            .lock()
            .map_err(|_| RusTorchError::KernelError("Failed to lock kernel cache".to_string()))?;

        kernels.insert(config.kernel_type.clone(), compiled_kernel);
        Ok(())
    }

    /// Execute custom kernel
    /// カスタムカーネルを実行
    pub fn execute_kernel<T: Float + 'static>(
        &self,
        kernel_type: &CustomKernelType,
        inputs: &[&Tensor<T>],
        outputs: &mut [&mut Tensor<T>],
        config: &KernelConfig,
    ) -> RusTorchResult<()> {
        let kernels = self
            .compiled_kernels
            .lock()
            .map_err(|_| RusTorchError::KernelError("Failed to lock kernel cache".to_string()))?;

        let kernel = kernels.get(kernel_type).ok_or_else(|| {
            RusTorchError::KernelError(format!("Kernel {:?} not found", kernel_type))
        })?;

        match self.device_type {
            DeviceType::Cuda(_) => self.execute_cuda_kernel(kernel, inputs, outputs, config),
            DeviceType::Metal(_) => self.execute_metal_kernel(kernel, inputs, outputs, config),
            DeviceType::OpenCL(_) => self.execute_opencl_kernel(kernel, inputs, outputs, config),
            DeviceType::Cpu => Err(RusTorchError::UnsupportedOperation(
                "Custom kernels not supported on CPU".to_string(),
            )),
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(_) => Err(RusTorchError::UnsupportedOperation(
                "Custom kernels not supported on CoreML".to_string(),
            )),
            DeviceType::Auto => Err(RusTorchError::UnsupportedOperation(
                "Custom kernels not supported on Auto device".to_string(),
            )),
            #[cfg(feature = "mac-hybrid")]
            DeviceType::MacHybrid => Err(RusTorchError::UnsupportedOperation(
                "Custom kernels not supported on MacHybrid - use Metal or CoreML directly"
                    .to_string(),
            )),
        }
    }

    /// Generate kernel source code
    /// カーネルソースコードを生成
    fn generate_kernel_source(&self, config: &KernelConfig) -> RusTorchResult<String> {
        match &config.kernel_type {
            CustomKernelType::OptimizedConvolution => self.generate_convolution_kernel(config),
            CustomKernelType::FastFourierTransform => self.generate_fft_kernel(config),
            CustomKernelType::AttentionKernel => self.generate_attention_kernel(config),
            CustomKernelType::BatchNormalization => self.generate_batchnorm_kernel(config),
            CustomKernelType::CustomActivation(name) => {
                self.generate_activation_kernel(name, config)
            }
            CustomKernelType::OptimizedReduction => self.generate_reduction_kernel(config),
            CustomKernelType::SparseOperations => self.generate_sparse_kernel(config),
            CustomKernelType::TensorFusion => self.generate_fusion_kernel(config),
        }
    }

    /// Generate optimized convolution kernel
    /// 最適化畳み込みカーネルを生成
    fn generate_convolution_kernel(&self, config: &KernelConfig) -> RusTorchResult<String> {
        let kernel_size = config
            .parameters
            .get("kernel_size")
            .and_then(|p| {
                if let KernelParameter::IntArray(arr) = p {
                    Some(arr)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                RusTorchError::KernelError("Missing kernel_size parameter".to_string())
            })?;

        let default_stride = vec![1, 1];
        let stride = config
            .parameters
            .get("stride")
            .and_then(|p| {
                if let KernelParameter::IntArray(arr) = p {
                    Some(arr)
                } else {
                    None
                }
            })
            .unwrap_or(&default_stride);

        let default_padding = vec![0, 0];
        let padding = config
            .parameters
            .get("padding")
            .and_then(|p| {
                if let KernelParameter::IntArray(arr) = p {
                    Some(arr)
                } else {
                    None
                }
            })
            .unwrap_or(&default_padding);

        match self.device_type {
            DeviceType::Cuda(_) => Ok(format!(
                r#"
extern "C" __global__ void optimized_convolution(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {{
    // Optimized convolution with shared memory and register blocking
    // 共有メモリとレジスタブロッキングを使用した最適化畳み込み
    
    __shared__ float shared_input[{}];
    __shared__ float shared_kernel[{}];
    
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Load input tile to shared memory
    // 入力タイルを共有メモリにロード
    for (int i = thread_id; i < {}; i += blockDim.x * blockDim.y) {{
        // Coalesced memory access pattern
        // 連続メモリアクセスパターン
        shared_input[i] = input[/* calculated index */];
    }}
    
    // Load kernel weights to shared memory
    // カーネル重みを共有メモリにロード
    for (int i = thread_id; i < {}; i += blockDim.x * blockDim.y) {{
        shared_kernel[i] = kernel[out_channel * in_channels * kernel_h * kernel_w + i];
    }}
    
    __syncthreads();
    
    // Compute convolution with register accumulation
    // レジスタ累積を使用した畳み込み計算
    int out_y = threadIdx.y;
    int out_x = threadIdx.x;
    
    if (out_y < out_height && out_x < out_width) {{
        float result = 0.0f;
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {{
            for (int ky = 0; ky < kernel_h; ++ky) {{
                for (int kx = 0; kx < kernel_w; ++kx) {{
                    int in_y = out_y * {} - {} + ky;
                    int in_x = out_x * {} - {} + kx;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {{
                        int input_idx = in_c * in_height * in_width + in_y * in_width + in_x;
                        int kernel_idx = in_c * kernel_h * kernel_w + ky * kernel_w + kx;
                        
                        result += shared_input[input_idx] * shared_kernel[kernel_idx];
                    }}
                }}
            }}
        }}
        
        int output_idx = batch_idx * out_channels * out_height * out_width +
                        out_channel * out_height * out_width +
                        out_y * out_width + out_x;
        output[output_idx] = result;
    }}
}}
"#,
                config.shared_memory_size / 4,   // input tile size
                kernel_size[0] * kernel_size[1], // kernel size
                config.shared_memory_size / 8,   // input elements
                kernel_size[0] * kernel_size[1], // kernel elements
                stride[0],
                padding[0],
                stride[1],
                padding[1]
            )),

            DeviceType::Metal(_) => Ok(format!(
                r"
#include <metal_stdlib>
using namespace metal;

kernel void optimized_convolution(
    device const float* input [[buffer(0)]],
    device const float* kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& in_channels [[buffer(4)]],
    constant int& in_height [[buffer(5)]],
    constant int& in_width [[buffer(6)]],
    constant int& out_channels [[buffer(7)]],
    constant int& out_height [[buffer(8)]],
    constant int& out_width [[buffer(9)]],
    constant int& kernel_h [[buffer(10)]],
    constant int& kernel_w [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {{
    // Metal optimized convolution implementation
    // Metal最適化畳み込み実装
    
    threadgroup float shared_input[{}];
    threadgroup float shared_kernel[{}];
    
    uint batch_idx = gid.z;
    uint out_channel = tgid.y;
    uint out_y = tid.y;
    uint out_x = tid.x;
    
    // Optimized memory access and computation
    // 最適化されたメモリアクセスと計算
    if (out_y < out_height && out_x < out_width) {{
        float result = 0.0f;
        
        for (uint in_c = 0; in_c < in_channels; ++in_c) {{
            for (uint ky = 0; ky < kernel_h; ++ky) {{
                for (uint kx = 0; kx < kernel_w; ++kx) {{
                    int in_y = out_y * {} - {} + ky;
                    int in_x = out_x * {} - {} + kx;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {{
                        uint input_idx = batch_idx * in_channels * in_height * in_width +
                                        in_c * in_height * in_width + in_y * in_width + in_x;
                        uint kernel_idx = out_channel * in_channels * kernel_h * kernel_w +
                                         in_c * kernel_h * kernel_w + ky * kernel_w + kx;
                        
                        result += input[input_idx] * kernel[kernel_idx];
                    }}
                }}
            }}
        }}
        
        uint output_idx = batch_idx * out_channels * out_height * out_width +
                         out_channel * out_height * out_width +
                         out_y * out_width + out_x;
        output[output_idx] = result;
    }}
}}
",
                config.shared_memory_size / 4,
                kernel_size[0] * kernel_size[1],
                stride[0],
                padding[0],
                stride[1],
                padding[1]
            )),

            _ => Err(RusTorchError::UnsupportedOperation(format!(
                "Convolution kernel not supported for {:?}",
                self.device_type
            ))),
        }
    }

    /// Generate attention mechanism kernel
    /// アテンション機構カーネルを生成
    fn generate_attention_kernel(&self, config: &KernelConfig) -> RusTorchResult<String> {
        let head_dim = config
            .parameters
            .get("head_dim")
            .and_then(|p| {
                if let KernelParameter::Int(val) = p {
                    Some(*val)
                } else {
                    None
                }
            })
            .unwrap_or(64);

        let _num_heads = config
            .parameters
            .get("num_heads")
            .and_then(|p| {
                if let KernelParameter::Int(val) = p {
                    Some(*val)
                } else {
                    None
                }
            })
            .unwrap_or(8);

        match self.device_type {
            DeviceType::Cuda(_) => Ok(format!(
                r#"
extern "C" __global__ void fused_attention(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    float* attention_weights,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float scale
) {{
    // Fused multi-head attention with memory optimization
    // メモリ最適化を伴う融合マルチヘッドアテンション
    
    __shared__ float shared_query[{}];
    __shared__ float shared_key[{}];
    __shared__ float shared_value[{}];
    
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = threadIdx.x;
    
    // Load query, key, value to shared memory
    // クエリ、キー、バリューを共有メモリにロード
    if (seq_idx < seq_len) {{
        for (int d = 0; d < head_dim; ++d) {{
            int qkv_offset = batch_idx * num_heads * seq_len * head_dim +
                           head_idx * seq_len * head_dim +
                           seq_idx * head_dim + d;
            
            shared_query[seq_idx * head_dim + d] = query[qkv_offset];
            shared_key[seq_idx * head_dim + d] = key[qkv_offset];
            shared_value[seq_idx * head_dim + d] = value[qkv_offset];
        }}
    }}
    
    __syncthreads();
    
    // Compute attention scores
    // アテンションスコアを計算
    if (seq_idx < seq_len) {{
        float max_score = -INFINITY;
        
        // Find maximum for numerical stability
        // 数値安定性のための最大値を見つける
        for (int k = 0; k < seq_len; ++k) {{
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {{
                score += shared_query[seq_idx * head_dim + d] * shared_key[k * head_dim + d];
            }}
            score *= scale;
            max_score = fmaxf(max_score, score);
        }}
        
        // Compute softmax
        // ソフトマックスを計算
        float sum_exp = 0.0f;
        for (int k = 0; k < seq_len; ++k) {{
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {{
                score += shared_query[seq_idx * head_dim + d] * shared_key[k * head_dim + d];
            }}
            score = expf((score * scale) - max_score);
            
            int attn_idx = batch_idx * num_heads * seq_len * seq_len +
                          head_idx * seq_len * seq_len +
                          seq_idx * seq_len + k;
            attention_weights[attn_idx] = score;
            sum_exp += score;
        }}
        
        // Normalize and compute output
        // 正規化して出力を計算
        for (int d = 0; d < head_dim; ++d) {{
            float output_val = 0.0f;
            for (int k = 0; k < seq_len; ++k) {{
                int attn_idx = batch_idx * num_heads * seq_len * seq_len +
                              head_idx * seq_len * seq_len +
                              seq_idx * seq_len + k;
                float normalized_weight = attention_weights[attn_idx] / sum_exp;
                output_val += normalized_weight * shared_value[k * head_dim + d];
            }}
            
            int output_idx = batch_idx * num_heads * seq_len * head_dim +
                           head_idx * seq_len * head_dim +
                           seq_idx * head_dim + d;
            output[output_idx] = output_val;
        }}
    }}
}}
"#,
                head_dim * 32,
                head_dim * 32,
                head_dim * 32
            )), // shared memory sizes

            _ => Err(RusTorchError::UnsupportedOperation(format!(
                "Attention kernel not supported for {:?}",
                self.device_type
            ))),
        }
    }

    /// Generate other kernel types (simplified implementations)
    /// 他のカーネルタイプを生成（簡略化実装）
    fn generate_fft_kernel(&self, _config: &KernelConfig) -> RusTorchResult<String> {
        Ok("// FFT kernel implementation placeholder".to_string())
    }

    fn generate_batchnorm_kernel(&self, _config: &KernelConfig) -> RusTorchResult<String> {
        Ok("// Batch normalization kernel implementation placeholder".to_string())
    }

    fn generate_activation_kernel(
        &self,
        _name: &str,
        _config: &KernelConfig,
    ) -> RusTorchResult<String> {
        Ok("// Custom activation kernel implementation placeholder".to_string())
    }

    fn generate_reduction_kernel(&self, _config: &KernelConfig) -> RusTorchResult<String> {
        Ok("// Optimized reduction kernel implementation placeholder".to_string())
    }

    fn generate_sparse_kernel(&self, _config: &KernelConfig) -> RusTorchResult<String> {
        Ok("// Sparse operations kernel implementation placeholder".to_string())
    }

    fn generate_fusion_kernel(&self, _config: &KernelConfig) -> RusTorchResult<String> {
        Ok("// Tensor fusion kernel implementation placeholder".to_string())
    }

    /// Compile kernel source code
    /// カーネルソースコードをコンパイル
    fn compile_source(&self, source: &str, _config: &KernelConfig) -> RusTorchResult<Vec<u8>> {
        // Simplified compilation - in practice would use actual GPU compiler
        // 簡略化されたコンパイル - 実際にはGPUコンパイラを使用
        Ok(source.as_bytes().to_vec())
    }

    /// Get entry point name for kernel type
    /// カーネルタイプのエントリポイント名を取得
    fn get_entry_point(&self, kernel_type: &CustomKernelType) -> String {
        match kernel_type {
            CustomKernelType::OptimizedConvolution => "optimized_convolution".to_string(),
            CustomKernelType::AttentionKernel => "fused_attention".to_string(),
            CustomKernelType::FastFourierTransform => "fft_kernel".to_string(),
            CustomKernelType::BatchNormalization => "batch_normalization".to_string(),
            CustomKernelType::CustomActivation(name) => format!("custom_activation_{}", name),
            CustomKernelType::OptimizedReduction => "optimized_reduction".to_string(),
            CustomKernelType::SparseOperations => "sparse_operations".to_string(),
            CustomKernelType::TensorFusion => "tensor_fusion".to_string(),
        }
    }

    /// Execute CUDA kernel (simplified)
    /// CUDAカーネルを実行（簡略化）
    fn execute_cuda_kernel<T: Float + 'static>(
        &self,
        _kernel: &CompiledKernel,
        _inputs: &[&Tensor<T>],
        _outputs: &mut [&mut Tensor<T>],
        _config: &KernelConfig,
    ) -> RusTorchResult<()> {
        // Simplified execution - would use actual CUDA runtime
        // 簡略化された実行 - 実際にはCUDAランタイムを使用
        Ok(())
    }

    /// Execute Metal kernel (simplified)
    /// Metalカーネルを実行（簡略化）
    fn execute_metal_kernel<T: Float + 'static>(
        &self,
        _kernel: &CompiledKernel,
        _inputs: &[&Tensor<T>],
        _outputs: &mut [&mut Tensor<T>],
        _config: &KernelConfig,
    ) -> RusTorchResult<()> {
        // Simplified execution - would use actual Metal runtime
        // 簡略化された実行 - 実際にはMetalランタイムを使用
        Ok(())
    }

    /// Execute OpenCL kernel (simplified)
    /// OpenCLカーネルを実行（簡略化）
    fn execute_opencl_kernel<T: Float + 'static>(
        &self,
        _kernel: &CompiledKernel,
        _inputs: &[&Tensor<T>],
        _outputs: &mut [&mut Tensor<T>],
        _config: &KernelConfig,
    ) -> RusTorchResult<()> {
        // Simplified execution - would use actual OpenCL runtime
        // 簡略化された実行 - 実際にはOpenCLランタイムを使用
        Ok(())
    }

    /// Get kernel performance statistics
    /// カーネルパフォーマンス統計を取得
    pub fn get_kernel_stats(&self, kernel_type: &CustomKernelType) -> RusTorchResult<KernelStats> {
        let kernels = self
            .compiled_kernels
            .lock()
            .map_err(|_| RusTorchError::KernelError("Failed to lock kernel cache".to_string()))?;

        let kernel = kernels.get(kernel_type).ok_or_else(|| {
            RusTorchError::KernelError(format!("Kernel {:?} not found", kernel_type))
        })?;

        Ok(KernelStats {
            kernel_type: kernel_type.clone(),
            compilation_time: kernel.compilation_time.elapsed(),
            binary_size: kernel.binary_data.len(),
            execution_count: 0, // Would track in practice
            total_execution_time: std::time::Duration::from_secs(0),
        })
    }
}

/// Kernel performance statistics
/// カーネルパフォーマンス統計#[derive(Debug)]
pub struct KernelStats {
    /// Type of the kernel
    /// カーネルのタイプ
    pub kernel_type: CustomKernelType,
    /// Time taken to compile the kernel
    /// カーネルのコンパイル時間
    pub compilation_time: std::time::Duration,
    /// Size of the compiled binary in bytes
    /// コンパイル済みバイナリのサイズ（バイト）
    pub binary_size: usize,
    /// Number of times the kernel has been executed
    /// カーネルが実行された回数
    pub execution_count: u64,
    /// Total time spent executing the kernel
    /// カーネルの実行に費やされた総時間
    pub total_execution_time: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_manager_creation() {
        let manager = CustomKernelManager::new(DeviceType::Cuda(0));
        assert_eq!(manager.device_type, DeviceType::Cuda(0));
    }

    #[test]
    fn test_convolution_kernel_compilation() {
        let manager = CustomKernelManager::new(DeviceType::Cuda(0));

        let mut config = KernelConfig {
            kernel_type: CustomKernelType::OptimizedConvolution,
            block_size: (16, 16, 1),
            grid_size: (1, 1, 1),
            shared_memory_size: 4096,
            parameters: HashMap::new(),
        };

        config.parameters.insert(
            "kernel_size".to_string(),
            KernelParameter::IntArray(vec![3, 3]),
        );

        assert!(manager.compile_kernel(&config).is_ok());
    }

    #[test]
    fn test_attention_kernel_compilation() {
        let manager = CustomKernelManager::new(DeviceType::Cuda(0));

        let mut config = KernelConfig {
            kernel_type: CustomKernelType::AttentionKernel,
            block_size: (32, 1, 1),
            grid_size: (1, 8, 1),
            shared_memory_size: 8192,
            parameters: HashMap::new(),
        };

        config
            .parameters
            .insert("head_dim".to_string(), KernelParameter::Int(64));
        config
            .parameters
            .insert("num_heads".to_string(), KernelParameter::Int(8));

        assert!(manager.compile_kernel(&config).is_ok());
    }
}
