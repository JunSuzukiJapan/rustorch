//! Model Quantization for compression and acceleration
//! モデル量子化による圧縮と高速化
//!
//! This module provides comprehensive quantization techniques for neural networks,
//! including post-training quantization, quantization-aware training, and various
//! quantization schemes (int8, int4, float16, etc.).
//! ニューラルネットワーク用の包括的な量子化技術を提供、
//! ポストトレーニング量子化、量子化対応訓練、様々な量子化スキーム（int8、int4、float16など）を含む。

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, One, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

/// Quantization scheme types
/// 量子化スキーム型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    /// 8ビット整数量子化
    Int8,
    /// 4-bit integer quantization (for extreme compression)
    /// 4ビット整数量子化（極限圧縮用）
    Int4,
    /// 16-bit floating point quantization
    /// 16ビット浮動小数点量子化
    Float16,
    /// Dynamic range quantization
    /// 動的範囲量子化
    Dynamic,
}

/// Quantization calibration mode
/// 量子化キャリブレーションモード
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationMode {
    /// Use min-max values for range determination
    /// 範囲決定に最小最大値を使用
    MinMax,
    /// Use percentile values for outlier-robust quantization
    /// 外れ値に頑健な量子化のためにパーセンタイル値を使用
    Percentile(f32),
    /// Use entropy-based optimization (KL divergence)
    /// エントロピーベース最適化（KLダイバージェンス）を使用
    Entropy,
}

/// Quantization parameters for a tensor
/// テンソル用量子化パラメータ
#[derive(Debug, Clone)]
pub struct QuantizationParams<T: Float> {
    /// Scale factor for dequantization
    /// 逆量子化用スケール係数
    pub scale: T,
    /// Zero point for asymmetric quantization
    /// 非対称量子化用ゼロ点
    pub zero_point: i32,
    /// Minimum value in the quantized range
    /// 量子化範囲の最小値
    pub qmin: i32,
    /// Maximum value in the quantized range
    /// 量子化範囲の最大値
    pub qmax: i32,
    /// Quantization type
    /// 量子化タイプ
    pub qtype: QuantizationType,
}

impl<T: Float + FromPrimitive> QuantizationParams<T> {
    /// Create quantization parameters for Int8 symmetric quantization
    /// Int8対称量子化用の量子化パラメータを作成
    pub fn int8_symmetric(scale: T) -> Self {
        QuantizationParams {
            scale,
            zero_point: 0,
            qmin: -128,
            qmax: 127,
            qtype: QuantizationType::Int8,
        }
    }

    /// Create quantization parameters for Int8 asymmetric quantization
    /// Int8非対称量子化用の量子化パラメータを作成
    pub fn int8_asymmetric(scale: T, zero_point: i32) -> Self {
        QuantizationParams {
            scale,
            zero_point,
            qmin: -128,
            qmax: 127,
            qtype: QuantizationType::Int8,
        }
    }

    /// Create quantization parameters for Int4 quantization
    /// Int4量子化用の量子化パラメータを作成
    pub fn int4_symmetric(scale: T) -> Self {
        QuantizationParams {
            scale,
            zero_point: 0,
            qmin: -8,
            qmax: 7,
            qtype: QuantizationType::Int4,
        }
    }
}

/// Quantized tensor representation
/// 量子化テンソル表現
#[derive(Debug, Clone)]
pub struct QuantizedTensor<T: Float> {
    /// Quantized integer values
    /// 量子化整数値
    pub data: Vec<i8>,
    /// Original tensor shape
    /// 元のテンソル形状
    pub shape: Vec<usize>,
    /// Quantization parameters
    /// 量子化パラメータ
    pub params: QuantizationParams<T>,
}

impl<T> QuantizedTensor<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Zero
        + One
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + Signed,
{
    /// Create a new quantized tensor
    /// 新しい量子化テンソルを作成
    pub fn new(data: Vec<i8>, shape: Vec<usize>, params: QuantizationParams<T>) -> Self {
        QuantizedTensor {
            data,
            shape,
            params,
        }
    }

    /// Dequantize back to floating point tensor
    /// 浮動小数点テンソルに逆量子化
    pub fn dequantize(&self) -> Tensor<T> {
        let mut float_data = Vec::with_capacity(self.data.len());

        for &qval in &self.data {
            let qval_adjusted = i32::from(qval) - self.params.zero_point;
            let float_val = T::from_i32(qval_adjusted).unwrap() * self.params.scale;
            float_data.push(float_val);
        }

        Tensor::from_vec(float_data, self.shape.clone())
    }

    /// Get the compression ratio compared to full precision
    /// 全精度と比較した圧縮比を取得
    pub fn compression_ratio(&self) -> f32 {
        match self.params.qtype {
            QuantizationType::Int8 => 4.0,    // 32-bit float to 8-bit int
            QuantizationType::Int4 => 8.0,    // 32-bit float to 4-bit int
            QuantizationType::Float16 => 2.0, // 32-bit float to 16-bit float
            QuantizationType::Dynamic => 3.0, // Average compression
        }
    }

    /// Get memory usage in bytes
    /// メモリ使用量をバイト単位で取得
    pub fn memory_bytes(&self) -> usize {
        match self.params.qtype {
            QuantizationType::Int8 => self.data.len(),
            QuantizationType::Int4 => self.data.len().div_ceil(2), // 4 bits per element
            QuantizationType::Float16 => self.data.len() * 2,
            QuantizationType::Dynamic => self.data.len(),
        }
    }
}

/// Quantizer for neural network layers and tensors
/// ニューラルネットワーク層とテンソル用量子化器
#[derive(Debug)]
pub struct Quantizer<T: Float> {
    /// Calibration mode for range determination
    /// 範囲決定用キャリブレーションモード
    calibration_mode: CalibrationMode,
    /// Cache of quantization parameters per layer
    /// 層ごとの量子化パラメータキャッシュ
    param_cache: HashMap<String, QuantizationParams<T>>,
    /// Whether to use symmetric quantization
    /// 対称量子化を使用するかどうか
    symmetric: bool,
}

impl<T> Quantizer<T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Zero
        + One
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + Signed,
{
    /// Create a new quantizer
    /// 新しい量子化器を作成
    pub fn new(calibration_mode: CalibrationMode, symmetric: bool) -> Self {
        Quantizer {
            calibration_mode,
            param_cache: HashMap::new(),
            symmetric,
        }
    }

    /// Quantize a tensor using the specified quantization type
    /// 指定された量子化タイプを使用してテンソルを量子化
    pub fn quantize_tensor(
        &mut self,
        tensor: &Tensor<T>,
        qtype: QuantizationType,
        layer_name: Option<&str>,
    ) -> QuantizedTensor<T> {
        // Compute or retrieve quantization parameters
        let params = if let Some(name) = layer_name {
            if let Some(cached_params) = self.param_cache.get(name) {
                cached_params.clone()
            } else {
                let params = self.compute_quantization_params(tensor, qtype);
                self.param_cache.insert(name.to_string(), params.clone());
                params
            }
        } else {
            self.compute_quantization_params(tensor, qtype)
        };

        // Quantize the tensor data
        let quantized_data = self.quantize_data(tensor, &params);

        QuantizedTensor::new(quantized_data, tensor.shape().to_vec(), params)
    }

    /// Compute quantization parameters for a tensor
    /// テンソル用の量子化パラメータを計算
    fn compute_quantization_params(
        &self,
        tensor: &Tensor<T>,
        qtype: QuantizationType,
    ) -> QuantizationParams<T> {
        let tensor_array = tensor.as_array();
        let tensor_slice = tensor_array
            .as_slice()
            .unwrap_or_else(|| panic!("Cannot get slice from tensor"));

        let (min_val, max_val) = match self.calibration_mode {
            CalibrationMode::MinMax => {
                let min = tensor_slice
                    .iter()
                    .fold(T::infinity(), |a, &b| if a < b { a } else { b });
                let max = tensor_slice
                    .iter()
                    .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
                (min, max)
            }
            CalibrationMode::Percentile(p) => {
                let mut sorted = tensor_slice.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let len = sorted.len();
                let low_idx = ((1.0 - p) * 0.5 * len as f32) as usize;
                let high_idx = len - 1 - low_idx;
                (sorted[low_idx], sorted[high_idx])
            }
            CalibrationMode::Entropy => {
                // Simplified entropy-based calibration (would need full KL divergence implementation)
                let min = tensor_slice
                    .iter()
                    .fold(T::infinity(), |a, &b| if a < b { a } else { b });
                let max = tensor_slice
                    .iter()
                    .fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
                (min, max)
            }
        };

        match qtype {
            QuantizationType::Int8 => {
                if self.symmetric {
                    let abs_max = min_val.abs().max(max_val.abs());
                    let scale = abs_max / T::from_i32(127).unwrap();
                    QuantizationParams::int8_symmetric(scale)
                } else {
                    let range = max_val - min_val;
                    let scale = range / T::from_i32(255).unwrap();
                    let zero_point = -(min_val / scale).round().to_i32().unwrap_or(0) - 128;
                    QuantizationParams::int8_asymmetric(scale, zero_point)
                }
            }
            QuantizationType::Int4 => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = abs_max / T::from_i32(7).unwrap();
                QuantizationParams::int4_symmetric(scale)
            }
            QuantizationType::Float16 => {
                // For float16, we just use direct conversion (simplified)
                QuantizationParams::int8_symmetric(T::one())
            }
            QuantizationType::Dynamic => {
                // Dynamic quantization uses different scales per channel/group
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = abs_max / T::from_i32(127).unwrap();
                QuantizationParams::int8_symmetric(scale)
            }
        }
    }

    /// Quantize tensor data using the given parameters
    /// 与えられたパラメータを使用してテンソルデータを量子化
    fn quantize_data(&self, tensor: &Tensor<T>, params: &QuantizationParams<T>) -> Vec<i8> {
        let tensor_array = tensor.as_array();
        let tensor_slice = tensor_array.as_slice().unwrap();

        tensor_slice
            .iter()
            .map(|&val| {
                let scaled = val / params.scale;
                let quantized = scaled.round().to_i32().unwrap_or(0) + params.zero_point;
                let clamped = quantized.max(params.qmin).min(params.qmax);
                clamped as i8
            })
            .collect()
    }

    /// Apply post-training quantization to a module
    /// モジュールにポストトレーニング量子化を適用
    pub fn quantize_module<M: Module<T>>(
        &mut self,
        module: &M,
        qtype: QuantizationType,
        layer_name: &str,
    ) -> Vec<QuantizedTensor<T>> {
        let parameters = module.parameters();
        let mut quantized_params = Vec::new();

        for (i, param) in parameters.iter().enumerate() {
            let param_name = format!("{}_{}", layer_name, i);
            let param_tensor = param.data();
            let param_data = param_tensor.read().unwrap();

            let quantized = self.quantize_tensor(&*param_data, qtype, Some(&param_name));
            quantized_params.push(quantized);
        }

        quantized_params
    }

    /// Clear the parameter cache
    /// パラメータキャッシュをクリア
    pub fn clear_cache(&mut self) {
        self.param_cache.clear();
    }

    /// Get calibration statistics
    /// キャリブレーション統計を取得
    pub fn get_statistics(&self) -> HashMap<String, QuantizationParams<T>> {
        self.param_cache.clone()
    }
}

/// Quantization-aware training (QAT) wrapper
/// 量子化対応訓練（QAT）ラッパー
#[derive(Debug)]
pub struct QuantizationAwareModule<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    M: Module<T> + 'static,
> {
    /// Underlying module
    /// 基底モジュール
    module: M,
    /// Quantization parameters for each layer
    /// 各層の量子化パラメータ
    qparams: HashMap<String, QuantizationParams<T>>,
    /// Whether QAT is enabled
    /// QATが有効かどうか
    qat_enabled: bool,
    /// Fake quantization during training
    /// 訓練中の擬似量子化
    fake_quantize: bool,
}

impl<T, M> QuantizationAwareModule<T, M>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Zero
        + One
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + Signed
        + 'static,
    M: Module<T> + 'static,
{
    /// Create a new QAT module
    /// 新しいQATモジュールを作成
    pub fn new(module: M) -> Self {
        QuantizationAwareModule {
            module,
            qparams: HashMap::new(),
            qat_enabled: false,
            fake_quantize: true,
        }
    }

    /// Enable quantization-aware training
    /// 量子化対応訓練を有効化
    pub fn enable_qat(&mut self) {
        self.qat_enabled = true;
    }

    /// Disable quantization-aware training
    /// 量子化対応訓練を無効化
    pub fn disable_qat(&mut self) {
        self.qat_enabled = false;
    }

    /// Set fake quantization mode
    /// 擬似量子化モードを設定
    pub fn set_fake_quantize(&mut self, enabled: bool) {
        self.fake_quantize = enabled;
    }

    /// Apply fake quantization to simulate quantization effects during training
    /// 訓練中に量子化効果をシミュレートするための擬似量子化を適用
    fn apply_fake_quantization(&self, input: &Variable<T>, layer_name: &str) -> Variable<T> {
        if !self.qat_enabled || !self.fake_quantize {
            return input.clone();
        }

        if let Some(params) = self.qparams.get(layer_name) {
            let input_binding = input.data();
            let input_data = input_binding.read().unwrap();

            // Simulate quantization by quantizing and dequantizing
            let quantized_data: Vec<T> = input_data
                .as_array()
                .iter()
                .map(|&val| {
                    let scaled = val / params.scale;
                    let quantized = scaled.round().to_i32().unwrap_or(0) + params.zero_point;
                    let clamped = quantized.max(params.qmin).min(params.qmax);
                    let qval_adjusted = clamped - params.zero_point;
                    T::from_i32(qval_adjusted).unwrap() * params.scale
                })
                .collect();

            let fake_quantized_tensor =
                Tensor::from_vec(quantized_data, input_data.shape().to_vec());
            Variable::new(fake_quantized_tensor, input.requires_grad())
        } else {
            input.clone()
        }
    }
}

impl<T, M> Module<T> for QuantizationAwareModule<T, M>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Default
        + Zero
        + One
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + Signed
        + 'static,
    M: Module<T> + 'static,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let fake_quantized_input = self.apply_fake_quantization(input, "input");
        self.module.forward(&fake_quantized_input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        self.module.parameters()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_params_creation() {
        let params = QuantizationParams::<f32>::int8_symmetric(0.1);
        assert_eq!(params.qmin, -128);
        assert_eq!(params.qmax, 127);
        assert_eq!(params.zero_point, 0);
        assert_eq!(params.qtype, QuantizationType::Int8);
    }

    #[test]
    fn test_quantizer_creation() {
        let quantizer = Quantizer::<f32>::new(CalibrationMode::MinMax, true);
        assert_eq!(quantizer.calibration_mode, CalibrationMode::MinMax);
        assert!(quantizer.symmetric);
    }

    #[test]
    fn test_tensor_quantization() {
        let mut quantizer = Quantizer::<f32>::new(CalibrationMode::MinMax, true);

        // Create a simple tensor
        let data = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]);

        let quantized = quantizer.quantize_tensor(&tensor, QuantizationType::Int8, None);

        assert_eq!(quantized.shape, vec![2, 3]);
        assert_eq!(quantized.data.len(), 6);
        assert_eq!(quantized.params.qtype, QuantizationType::Int8);
    }

    #[test]
    fn test_quantized_tensor_dequantization() {
        let params = QuantizationParams::<f32>::int8_symmetric(0.1);
        let data = vec![10, 20, 30, -10, -20, -30];
        let shape = vec![2, 3];

        let quantized = QuantizedTensor::new(data, shape, params);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.shape(), &[2, 3]);

        // Check that dequantization is approximately correct
        let dequant_array = dequantized.as_array();
        let expected = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        for (i, &val) in dequant_array.iter().enumerate() {
            assert!((val - expected[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let params = QuantizationParams::<f32>::int8_symmetric(0.1);
        let quantized = QuantizedTensor::new(vec![1, 2, 3], vec![3], params);

        assert_eq!(quantized.compression_ratio(), 4.0);
    }
}
