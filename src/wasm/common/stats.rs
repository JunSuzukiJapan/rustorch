//! Shared statistical computation utilities
//! 共通統計計算ユーティリティ

use crate::wasm::common::error::{WasmError, WasmResult};
use wasm_bindgen::prelude::*;

/// High-performance statistical operations
pub struct WasmStats;

impl WasmStats {
    /// Calculate mean of data slice
    #[inline]
    pub fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Calculate variance with optional precomputed mean
    #[inline]
    pub fn variance(data: &[f32], mean: Option<f32>) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let m = mean.unwrap_or_else(|| Self::mean(data));
        data.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / data.len() as f32
    }

    /// Calculate standard deviation with optional precomputed mean
    #[inline]
    pub fn std_dev(data: &[f32], mean: Option<f32>) -> f32 {
        Self::variance(data, mean).sqrt()
    }

    /// Calculate minimum value
    #[inline]
    pub fn min(data: &[f32]) -> f32 {
        data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    /// Calculate maximum value
    #[inline]
    pub fn max(data: &[f32]) -> f32 {
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Calculate Z-score for a value
    #[inline]
    pub fn z_score(value: f32, mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 {
            return 0.0;
        }
        (value - mean) / std_dev
    }

    /// Calculate percentile (modifies input data by sorting)
    pub fn percentile(data: &mut [f32], p: f32) -> WasmResult<f32> {
        if data.is_empty() {
            return Err(WasmError::empty_tensor());
        }

        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((p / 100.0) * (data.len() - 1) as f32).round() as usize;
        Ok(data[index.min(data.len() - 1)])
    }

    /// Calculate percentile without modifying input (creates copy)
    pub fn percentile_immutable(data: &[f32], p: f32) -> WasmResult<f32> {
        let mut sorted_data = data.to_vec();
        Self::percentile(&mut sorted_data, p)
    }

    /// Comprehensive statistics in one pass
    pub fn comprehensive_stats(data: &[f32]) -> WasmResult<BasicStats> {
        if data.is_empty() {
            return Err(WasmError::empty_tensor());
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut finite_count = 0;
        let mut nan_count = 0;

        for &value in data {
            if value.is_finite() {
                finite_count += 1;
                min = min.min(value);
                max = max.max(value);
                sum += value;
                sum_sq += value * value;
            } else if value.is_nan() {
                nan_count += 1;
            }
        }

        let mean = sum / finite_count as f32;
        let variance = (sum_sq / finite_count as f32) - (mean * mean);
        let std_dev = variance.sqrt();

        Ok(BasicStats {
            mean,
            std_dev,
            min,
            max,
            count: data.len(),
            finite_count,
            nan_count,
        })
    }

    /// Calculate correlation coefficient
    pub fn correlation(x: &[f32], y: &[f32]) -> WasmResult<f32> {
        if x.len() != y.len() {
            return Err(WasmError::size_mismatch(x.len(), y.len()));
        }
        if x.is_empty() {
            return Err(WasmError::empty_tensor());
        }

        let n = x.len() as f32;
        let x_mean = Self::mean(x);
        let y_mean = Self::mean(y);

        let numerator: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_var: f32 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let y_var: f32 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let denominator = (x_var * y_var).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate covariance
    pub fn covariance(x: &[f32], y: &[f32]) -> WasmResult<f32> {
        if x.len() != y.len() {
            return Err(WasmError::size_mismatch(x.len(), y.len()));
        }
        if x.len() < 2 {
            return Err(WasmError::insufficient_data("covariance", 2, x.len()));
        }

        let x_mean = Self::mean(x);
        let y_mean = Self::mean(y);

        let covariance: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f32>()
            / (x.len() - 1) as f32;

        Ok(covariance)
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(data: &[f32], factor: f32) -> WasmResult<Vec<(usize, f32)>> {
        if data.len() < 4 {
            return Ok(Vec::new());
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = data.len() / 4;
        let q3_idx = (3 * data.len()) / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;

        Ok(data
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some((i, value))
                } else {
                    None
                }
            })
            .collect())
    }
}

/// Basic statistics structure
#[derive(Debug, Clone)]
pub struct BasicStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub count: usize,
    pub finite_count: usize,
    pub nan_count: usize,
}

impl BasicStats {
    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        format!(
            "{{\"mean\":{:.6},\"std\":{:.6},\"min\":{:.6},\"max\":{:.6},\"count\":{},\"finite_count\":{},\"nan_count\":{}}}",
            self.mean, self.std_dev, self.min, self.max, self.count, self.finite_count, self.nan_count
        )
    }
}
