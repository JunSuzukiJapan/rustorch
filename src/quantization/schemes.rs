//! Quantization schemes and algorithms
//! 量子化スキームとアルゴリズム

use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, ArrayViewD};
use num_traits::Float;
use std::marker::PhantomData;

/// Quantization scheme enumeration
/// 量子化スキーム列挙型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    /// Symmetric quantization (zero_point = 0)
    /// 対称量子化（ゼロポイント = 0）
    Symmetric,
    /// Asymmetric quantization (zero_point != 0)  
    /// 非対称量子化（ゼロポイント != 0）
    Asymmetric,
    /// Per-channel symmetric quantization
    /// チャンネル別対称量子化
    PerChannelSymmetric,
    /// Per-channel asymmetric quantization
    /// チャンネル別非対称量子化
    PerChannelAsymmetric,
}

impl QuantizationScheme {
    /// Compute quantization parameters for the given data
    /// 指定されたデータの量子化パラメータを計算
    pub fn compute_params<T: Float>(&self, data: &ArrayD<T>) -> RusTorchResult<(f32, i32)> {
        match self {
            QuantizationScheme::Symmetric => SymmetricQuantization::compute_params(data),
            QuantizationScheme::Asymmetric => AsymmetricQuantization::compute_params(data),
            QuantizationScheme::PerChannelSymmetric => {
                // For simplicity, use global symmetric for now
                // 簡略化のため、現在はグローバル対称を使用
                SymmetricQuantization::compute_params(data)
            }
            QuantizationScheme::PerChannelAsymmetric => {
                // For simplicity, use global asymmetric for now
                // 簡略化のため、現在はグローバル非対称を使用
                AsymmetricQuantization::compute_params(data)
            }
        }
    }

    /// Compute per-channel quantization parameters
    /// チャンネル別量子化パラメータを計算
    pub fn compute_per_channel_params<T: Float>(
        &self,
        data: &ArrayD<T>,
        channel_axis: usize,
    ) -> RusTorchResult<(Vec<f32>, Vec<i32>)> {
        if channel_axis >= data.ndim() {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Channel axis {} exceeds tensor dimensions {}",
                    channel_axis,
                    data.ndim()
                ),
                source: None,
            });
        }

        let num_channels = data.shape()[channel_axis];
        let mut scales = Vec::with_capacity(num_channels);
        let mut zero_points = Vec::with_capacity(num_channels);

        for channel in 0..num_channels {
            // Extract channel data
            let channel_data = data.index_axis(ndarray::Axis(channel_axis), channel);
            let channel_array = channel_data.into_owned();

            let (scale, zero_point) = match self {
                QuantizationScheme::PerChannelSymmetric => {
                    SymmetricQuantization::compute_params(&channel_array)?
                }
                QuantizationScheme::PerChannelAsymmetric => {
                    AsymmetricQuantization::compute_params(&channel_array)?
                }
                _ => {
                    return Err(RusTorchError::TensorOp {
                        message: "Not a per-channel quantization scheme".to_string(),
                        source: None,
                    })
                }
            };

            scales.push(scale);
            zero_points.push(zero_point);
        }

        Ok((scales, zero_points))
    }
}

/// Trait for quantization parameter computation
/// 量子化パラメータ計算のトレイト
pub trait QuantizationParams<T: Float> {
    /// Compute scale and zero_point for quantization
    /// 量子化のためのスケールとゼロポイントを計算
    fn compute_params(data: &ArrayD<T>) -> RusTorchResult<(f32, i32)>;

    /// Compute quantization parameters from min/max values
    /// 最小/最大値から量子化パラメータを計算
    fn compute_from_range(
        min_val: T,
        max_val: T,
        qmin: i32,
        qmax: i32,
    ) -> RusTorchResult<(f32, i32)>;
}

/// Symmetric quantization implementation
/// 対称量子化実装
pub struct SymmetricQuantization<T> {
    _phantom: PhantomData<T>,
}

impl<T: Float> QuantizationParams<T> for SymmetricQuantization<T> {
    fn compute_params(data: &ArrayD<T>) -> RusTorchResult<(f32, i32)> {
        if data.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Cannot quantize empty tensor".to_string(),
                source: None,
            });
        }

        // Find absolute maximum value for symmetric quantization
        // 対称量子化のための絶対最大値を検索
        let abs_max = data
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |acc, x| if acc > x { acc } else { x });

        if abs_max.is_zero() {
            return Ok((1.0, 0)); // All zeros case
        }

        // For INT8 quantization: range is [-127, 127] (not -128 to keep symmetry)
        // INT8量子化の場合：範囲は[-127, 127]（対称性を保つため-128ではない）
        let qmin = -127i32;
        let qmax = 127i32;

        Self::compute_from_range(-abs_max, abs_max, qmin, qmax)
    }

    fn compute_from_range(
        min_val: T,
        max_val: T,
        qmin: i32,
        qmax: i32,
    ) -> RusTorchResult<(f32, i32)> {
        if min_val > max_val {
            return Err(RusTorchError::TensorOp {
                message: "min_val cannot be greater than max_val".to_string(),
                source: None,
            });
        }

        // For symmetric quantization, zero_point is always 0
        // 対称量子化の場合、ゼロポイントは常に0
        let zero_point = 0i32;

        // Scale is the maximum absolute value divided by the quantization range
        // スケールは絶対最大値を量子化範囲で割った値
        let abs_max = if min_val.abs() > max_val.abs() {
            min_val.abs()
        } else {
            max_val.abs()
        };
        let scale = if abs_max.is_zero() {
            1.0f32
        } else {
            abs_max.to_f32().unwrap_or(1.0) / (qmax as f32)
        };

        Ok((scale, zero_point))
    }
}

/// Asymmetric quantization implementation
/// 非対称量子化実装
pub struct AsymmetricQuantization<T> {
    _phantom: PhantomData<T>,
}

impl<T: Float> QuantizationParams<T> for AsymmetricQuantization<T> {
    fn compute_params(data: &ArrayD<T>) -> RusTorchResult<(f32, i32)> {
        if data.is_empty() {
            return Err(RusTorchError::TensorOp {
                message: "Cannot quantize empty tensor".to_string(),
                source: None,
            });
        }

        // Find min and max values
        // 最小値と最大値を検索
        let min_val = data
            .iter()
            .fold(T::infinity(), |acc, &x| if acc < x { acc } else { x });
        let max_val = data
            .iter()
            .fold(T::neg_infinity(), |acc, &x| if acc > x { acc } else { x });

        // For INT8 quantization: full range [-128, 127]
        // INT8量子化の場合：フル範囲[-128, 127]
        let qmin = -128i32;
        let qmax = 127i32;

        Self::compute_from_range(min_val, max_val, qmin, qmax)
    }

    fn compute_from_range(
        min_val: T,
        max_val: T,
        qmin: i32,
        qmax: i32,
    ) -> RusTorchResult<(f32, i32)> {
        if min_val > max_val {
            return Err(RusTorchError::TensorOp {
                message: "min_val cannot be greater than max_val".to_string(),
                source: None,
            });
        }

        let range = max_val - min_val;

        // Handle case where all values are the same
        // 全ての値が同じ場合を処理
        if range.is_zero() {
            let zero_point = if min_val.is_zero() { 0 } else { qmin };
            return Ok((1.0, zero_point));
        }

        // Compute scale: (max - min) / (qmax - qmin)
        // スケールを計算：(max - min) / (qmax - qmin)
        let scale = range.to_f32().unwrap_or(1.0) / ((qmax - qmin) as f32);

        // Compute zero_point: round(qmin - min_val / scale)
        // ゼロポイントを計算：round(qmin - min_val / scale)
        let zero_point_f = qmin as f32 - min_val.to_f32().unwrap_or(0.0) / scale;
        let zero_point = zero_point_f.round() as i32;

        // Clamp zero_point to quantization range
        // ゼロポイントを量子化範囲にクランプ
        let zero_point = zero_point.clamp(qmin, qmax);

        Ok((scale, zero_point))
    }
}

/// Per-channel quantization utilities
/// チャンネル別量子化ユーティリティ
pub struct PerChannelQuantization;

impl PerChannelQuantization {
    /// Compute per-channel quantization parameters for weight tensors
    /// 重みテンソルのチャンネル別量子化パラメータを計算
    pub fn compute_weight_params<T: Float>(
        weights: &ArrayD<T>,
        scheme: QuantizationScheme,
    ) -> RusTorchResult<(Vec<f32>, Vec<i32>)> {
        // Assume weights are in format [out_channels, in_channels, ...]
        // 重みが[out_channels, in_channels, ...]形式であると仮定
        let output_channel_axis = 0;
        scheme.compute_per_channel_params(weights, output_channel_axis)
    }

    /// Apply per-channel quantization to weights
    /// 重みにチャンネル別量子化を適用
    pub fn quantize_weights<T: Float>(
        weights: &ArrayD<T>,
        scales: &[f32],
        zero_points: &[i32],
    ) -> RusTorchResult<ArrayD<i8>> {
        let num_channels = weights.shape()[0];
        if scales.len() != num_channels || zero_points.len() != num_channels {
            return Err(RusTorchError::TensorOp {
                message: "Scales and zero_points length must match number of output channels"
                    .to_string(),
                source: None,
            });
        }

        let mut quantized = ArrayD::<i8>::zeros(weights.raw_dim());

        for (channel_idx, ((mut q_channel, w_channel), (&scale, &zero_point))) in quantized
            .axis_iter_mut(ndarray::Axis(0))
            .zip(weights.axis_iter(ndarray::Axis(0)))
            .zip(scales.iter().zip(zero_points.iter()))
            .enumerate()
        {
            for (q_val, &w_val) in q_channel.iter_mut().zip(w_channel.iter()) {
                let quantized_val =
                    (w_val.to_f32().unwrap_or(0.0) / scale).round() as i32 + zero_point;
                *q_val = quantized_val.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            }
        }

        Ok(quantized)
    }
}

/// Quantization calibration using different statistics
/// 異なる統計を使用した量子化キャリブレーション
#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    /// Use min/max values
    /// 最小/最大値を使用
    MinMax,
    /// Use percentiles to handle outliers
    /// 外れ値を処理するためにパーセンタイルを使用
    Percentile { lower: f32, upper: f32 },
    /// Use entropy-based calibration
    /// エントロピーベースのキャリブレーション
    Entropy { num_bins: usize },
}

impl CalibrationMethod {
    /// Apply calibration method to compute quantization range
    /// キャリブレーション手法を適用して量子化範囲を計算
    pub fn compute_range<T: Float>(&self, data: &ArrayD<T>) -> RusTorchResult<(T, T)> {
        match self {
            CalibrationMethod::MinMax => {
                let min_val = data
                    .iter()
                    .fold(T::infinity(), |acc, &x| if acc < x { acc } else { x });
                let max_val = data
                    .iter()
                    .fold(T::neg_infinity(), |acc, &x| if acc > x { acc } else { x });
                Ok((min_val, max_val))
            }
            CalibrationMethod::Percentile { lower, upper } => {
                let mut values: Vec<T> = data.iter().copied().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = values.len();
                if n == 0 {
                    return Err(RusTorchError::TensorOp {
                        message: "Cannot compute percentiles on empty data".to_string(),
                        source: None,
                    });
                }

                let lower_idx = ((lower / 100.0) * (n - 1) as f32).round() as usize;
                let upper_idx = ((upper / 100.0) * (n - 1) as f32).round() as usize;

                let min_val = values[lower_idx.min(n - 1)];
                let max_val = values[upper_idx.min(n - 1)];

                Ok((min_val, max_val))
            }
            CalibrationMethod::Entropy { num_bins: _ } => {
                // Simplified entropy method - use min/max for now
                // 簡略化されたエントロピー手法 - 現在は最小/最大を使用
                // TODO: Implement full entropy-based calibration
                self.compute_range(data)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_symmetric_quantization() {
        let data = Array1::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let (scale, zero_point) = SymmetricQuantization::compute_params(&data).unwrap();

        assert_eq!(zero_point, 0);
        assert!((scale - (2.0 / 127.0)).abs() < 1e-6);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let data = Array1::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0]).into_dyn();
        let (scale, zero_point) = AsymmetricQuantization::compute_params(&data).unwrap();

        // Should have non-zero zero_point for asymmetric case
        assert_ne!(zero_point, 0);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_quantization_scheme_api() {
        let data = Array1::from_vec(vec![-1.0f32, 0.0, 1.0]).into_dyn();

        let (scale_sym, zp_sym) = QuantizationScheme::Symmetric.compute_params(&data).unwrap();
        let (scale_asym, zp_asym) = QuantizationScheme::Asymmetric
            .compute_params(&data)
            .unwrap();

        assert_eq!(zp_sym, 0);
        assert_ne!(zp_asym, 0);
        assert!(scale_sym > 0.0 && scale_asym > 0.0);
    }

    #[test]
    fn test_per_channel_params() {
        let data = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap()
        .into_dyn();

        let (scales, zero_points) = QuantizationScheme::PerChannelSymmetric
            .compute_per_channel_params(&data, 0)
            .unwrap();

        assert_eq!(scales.len(), 2); // 2 channels
        assert_eq!(zero_points.len(), 2);
        assert!(scales.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_calibration_methods() {
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
            .into_dyn(); // Has outlier

        let (min_max_min, min_max_max) = CalibrationMethod::MinMax.compute_range(&data).unwrap();
        assert_eq!(min_max_min, 1.0);
        assert_eq!(min_max_max, 100.0);

        let (perc_min, perc_max) = CalibrationMethod::Percentile {
            lower: 10.0,
            upper: 90.0,
        }
        .compute_range(&data)
        .unwrap();
        // Percentile method should reduce impact of outlier
        assert!(perc_max < 100.0);
    }

    #[test]
    fn test_edge_cases() {
        // Empty tensor
        let empty_data = ArrayD::<f32>::zeros(ndarray::IxDyn(&[0]));
        assert!(SymmetricQuantization::compute_params(&empty_data).is_err());

        // All zeros
        let zero_data = Array1::<f32>::zeros(5).into_dyn();
        let (scale, zero_point) = SymmetricQuantization::compute_params(&zero_data).unwrap();
        assert_eq!(scale, 1.0);
        assert_eq!(zero_point, 0);

        // Single value
        let single_data = Array1::from_vec(vec![3.14f32]).into_dyn();
        let result = AsymmetricQuantization::compute_params(&single_data);
        assert!(result.is_ok());
    }
}
