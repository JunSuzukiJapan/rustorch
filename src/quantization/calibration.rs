//! Calibration and statistical observation for quantization
//! 量子化のためのキャリブレーションと統計観測

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use super::{TensorQuantization};
use super::schemes::{QuantizationScheme, CalibrationMethod};
use super::types::{QuantizedTensor, QuantizableInteger};
use ndarray::ArrayD;
use num_traits::Float;
use std::collections::HashMap;

/// Trait for statistical observers used in quantization calibration
/// 量子化キャリブレーションで使用される統計観測器のトレイト
pub trait Observer<T: Float> {
    /// Observe a batch of data
    /// データのバッチを観測
    fn observe(&mut self, data: &ArrayD<T>);
    
    /// Get the observed minimum value
    /// 観測された最小値を取得
    fn min_val(&self) -> Option<T>;
    
    /// Get the observed maximum value
    /// 観測された最大値を取得  
    fn max_val(&self) -> Option<T>;
    
    /// Reset observer state
    /// 観測器の状態をリセット
    fn reset(&mut self);
    
    /// Get quantization parameters based on observations
    /// 観測に基づいて量子化パラメータを取得
    fn get_quantization_params(&self, scheme: QuantizationScheme) -> RusTorchResult<(f32, i32)>;
}

/// Simple min-max observer for calibration
/// キャリブレーション用のシンプルな最小-最大観測器
#[derive(Debug, Clone)]
pub struct MinMaxObserver<T: Float> {
    min_val: Option<T>,
    max_val: Option<T>,
    num_observations: usize,
}

impl<T: Float> MinMaxObserver<T> {
    /// Create a new min-max observer
    /// 新しい最小-最大観測器を作成
    pub fn new() -> Self {
        Self {
            min_val: None,
            max_val: None,
            num_observations: 0,
        }
    }
}

impl<T: Float> Default for MinMaxObserver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Observer<T> for MinMaxObserver<T> {
    fn observe(&mut self, data: &ArrayD<T>) {
        if data.is_empty() {
            return;
        }

        let data_min = data.iter().fold(T::infinity(), |acc, &x| if acc < x { acc } else { x });
        let data_max = data.iter().fold(T::neg_infinity(), |acc, &x| if acc > x { acc } else { x });

        self.min_val = Some(match self.min_val {
            None => data_min,
            Some(current_min) => if current_min < data_min { current_min } else { data_min },
        });

        self.max_val = Some(match self.max_val {
            None => data_max,  
            Some(current_max) => if current_max > data_max { current_max } else { data_max },
        });

        self.num_observations += 1;
    }

    fn min_val(&self) -> Option<T> {
        self.min_val
    }

    fn max_val(&self) -> Option<T> {
        self.max_val
    }

    fn reset(&mut self) {
        self.min_val = None;
        self.max_val = None;
        self.num_observations = 0;
    }

    fn get_quantization_params(&self, scheme: QuantizationScheme) -> RusTorchResult<(f32, i32)> {
        let min_val = self.min_val.ok_or_else(|| RusTorchError::TensorOp {
            message: "No observations recorded".to_string(),
            source: None,
        })?;

        let max_val = self.max_val.ok_or_else(|| RusTorchError::TensorOp {
            message: "No observations recorded".to_string(),
            source: None,
        })?;

        match scheme {
            QuantizationScheme::Symmetric => {
                let abs_max = if min_val.abs() > max_val.abs() { min_val.abs() } else { max_val.abs() };
                let scale = if abs_max.is_zero() { 1.0f32 } else { abs_max.to_f32().unwrap_or(1.0) / 127.0 };
                Ok((scale, 0))
            },
            QuantizationScheme::Asymmetric => {
                let range = max_val - min_val;
                if range.is_zero() {
                    Ok((1.0, 0))
                } else {
                    let scale = range.to_f32().unwrap_or(1.0) / 255.0; // INT8 range: 256 levels
                    let zero_point = (-128.0 - min_val.to_f32().unwrap_or(0.0) / scale).round() as i32;
                    let zero_point = zero_point.clamp(-128, 127);
                    Ok((scale, zero_point))
                }
            },
            _ => {
                // For per-channel schemes, fall back to asymmetric
                // チャンネル別スキームの場合、非対称にフォールバック
                self.get_quantization_params(QuantizationScheme::Asymmetric)
            }
        }
    }
}

/// Histogram-based observer for more robust calibration
/// より堅牢なキャリブレーションのためのヒストグラムベース観測器
#[derive(Debug, Clone)]
pub struct HistogramObserver<T: Float> {
    histogram: Vec<usize>,
    bin_edges: Vec<T>,
    num_bins: usize,
    min_val: Option<T>,
    max_val: Option<T>,
    total_count: usize,
}

impl<T: Float> HistogramObserver<T> {
    /// Create a new histogram observer
    /// 新しいヒストグラム観測器を作成
    pub fn new(num_bins: usize) -> Self {
        Self {
            histogram: vec![0; num_bins],
            bin_edges: Vec::new(),
            num_bins,
            min_val: None,
            max_val: None,
            total_count: 0,
        }
    }

    /// Initialize histogram bins based on observed data range
    /// 観測データ範囲に基づいてヒストグラムビンを初期化
    fn initialize_bins(&mut self, min_val: T, max_val: T) {
        if self.bin_edges.is_empty() {
            let range = max_val - min_val;
            let bin_width = range / T::from(self.num_bins).unwrap_or_else(|| T::one());
            
            self.bin_edges = (0..=self.num_bins)
                .map(|i| min_val + bin_width * T::from(i).unwrap_or_else(T::zero))
                .collect();
        }
    }

    /// Find bin index for a value
    /// 値のビンインデックスを検索
    fn find_bin(&self, value: T) -> Option<usize> {
        if self.bin_edges.len() < 2 {
            return None;
        }

        for (i, &edge) in self.bin_edges.iter().enumerate().skip(1) {
            if value <= edge {
                return Some((i - 1).min(self.num_bins - 1));
            }
        }
        Some(self.num_bins - 1)
    }

    /// Compute optimal quantization range using entropy method
    /// エントロピー法を使用して最適量子化範囲を計算
    pub fn compute_optimal_range(&self, target_quantiles: (f32, f32)) -> RusTorchResult<(T, T)> {
        if self.total_count == 0 {
            return Err(RusTorchError::TensorOp {
                message: "No data observed for entropy calculation".to_string(),
                source: None,
            });
        }

        // Find quantiles based on histogram
        // ヒストグラムに基づいて分位数を検索
        let lower_count = (target_quantiles.0 * self.total_count as f32) as usize;
        let upper_count = (target_quantiles.1 * self.total_count as f32) as usize;

        let mut cumulative_count = 0;
        let mut lower_bin = 0;
        let mut upper_bin = self.num_bins - 1;

        for (i, &count) in self.histogram.iter().enumerate() {
            cumulative_count += count;
            if cumulative_count >= lower_count && lower_bin == 0 {
                lower_bin = i;
            }
            if cumulative_count >= upper_count {
                upper_bin = i;
                break;
            }
        }

        let min_range = self.bin_edges.get(lower_bin).copied()
            .unwrap_or_else(|| self.min_val.unwrap_or_else(T::zero));
        let max_range = self.bin_edges.get(upper_bin + 1).copied()
            .unwrap_or_else(|| self.max_val.unwrap_or_else(T::one));

        Ok((min_range, max_range))
    }
}

impl<T: Float> Observer<T> for HistogramObserver<T> {
    fn observe(&mut self, data: &ArrayD<T>) {
        if data.is_empty() {
            return;
        }

        let data_min = data.iter().fold(T::infinity(), |acc, &x| if acc < x { acc } else { x });
        let data_max = data.iter().fold(T::neg_infinity(), |acc, &x| if acc > x { acc } else { x });

        // Update global min/max
        self.min_val = Some(match self.min_val {
            None => data_min,
            Some(current_min) => if current_min < data_min { current_min } else { data_min },
        });

        self.max_val = Some(match self.max_val {
            None => data_max,
            Some(current_max) => if current_max > data_max { current_max } else { data_max },
        });

        // Initialize bins if needed
        if let (Some(min_val), Some(max_val)) = (self.min_val, self.max_val) {
            self.initialize_bins(min_val, max_val);
        }

        // Update histogram
        for &value in data.iter() {
            if let Some(bin_idx) = self.find_bin(value) {
                self.histogram[bin_idx] += 1;
                self.total_count += 1;
            }
        }
    }

    fn min_val(&self) -> Option<T> {
        self.min_val
    }

    fn max_val(&self) -> Option<T> {
        self.max_val
    }

    fn reset(&mut self) {
        self.histogram.fill(0);
        self.bin_edges.clear();
        self.min_val = None;
        self.max_val = None;
        self.total_count = 0;
    }

    fn get_quantization_params(&self, scheme: QuantizationScheme) -> RusTorchResult<(f32, i32)> {
        // Use 1st and 99th percentiles to handle outliers
        // 外れ値を処理するために第1および第99パーセンタイルを使用
        let (range_min, range_max) = self.compute_optimal_range((0.01, 0.99))?;

        match scheme {
            QuantizationScheme::Symmetric => {
                let abs_max = if range_min.abs() > range_max.abs() { range_min.abs() } else { range_max.abs() };
                let scale = if abs_max.is_zero() { 1.0f32 } else { abs_max.to_f32().unwrap_or(1.0) / 127.0 };
                Ok((scale, 0))
            },
            QuantizationScheme::Asymmetric => {
                let range = range_max - range_min;
                if range.is_zero() {
                    Ok((1.0, 0))
                } else {
                    let scale = range.to_f32().unwrap_or(1.0) / 255.0;
                    let zero_point = (-128.0 - range_min.to_f32().unwrap_or(0.0) / scale).round() as i32;
                    let zero_point = zero_point.clamp(-128, 127);
                    Ok((scale, zero_point))
                }
            },
            _ => self.get_quantization_params(QuantizationScheme::Asymmetric)
        }
    }
}

/// Static quantizer for pre-calibrated quantization
/// 事前キャリブレーション済み量子化のための静的量子化器
pub struct StaticQuantizer<T: Float> {
    observers: HashMap<String, Box<dyn Observer<T>>>,
    quantization_params: HashMap<String, (f32, i32)>,
    calibration_method: CalibrationMethod,
    calibrated: bool,
}

impl<T: Float> StaticQuantizer<T> {
    /// Create a new static quantizer
    /// 新しい静的量子化器を作成
    pub fn new() -> Self {
        Self {
            observers: HashMap::new(),
            quantization_params: HashMap::new(),
            calibration_method: CalibrationMethod::MinMax,
            calibrated: false,
        }
    }

    /// Set calibration method
    /// キャリブレーション手法を設定
    pub fn with_calibration_method(mut self, method: CalibrationMethod) -> Self {
        self.calibration_method = method;
        self
    }

    /// Add an observer for a specific layer/tensor
    /// 特定のレイヤー/テンソルの観測器を追加
    pub fn add_observer(&mut self, name: String, observer: Box<dyn Observer<T>>) {
        self.observers.insert(name, observer);
    }

    /// Observe data for calibration
    /// キャリブレーションのためのデータ観測
    pub fn observe(&mut self, name: &str, data: &ArrayD<T>) -> RusTorchResult<()> {
        let observer = self.observers.get_mut(name)
            .ok_or_else(|| RusTorchError::TensorOp {
                message: format!("Observer for '{}' not found", name),
                source: None,
            })?;
        
        observer.observe(data);
        Ok(())
    }

    /// Calibrate quantization parameters
    /// 量子化パラメータをキャリブレート
    pub fn calibrate(&mut self, scheme: QuantizationScheme) -> RusTorchResult<()> {
        self.quantization_params.clear();

        for (name, observer) in &self.observers {
            let params = observer.get_quantization_params(scheme)?;
            self.quantization_params.insert(name.clone(), params);
        }

        self.calibrated = true;
        Ok(())
    }

    /// Get quantization parameters for a layer
    /// レイヤーの量子化パラメータを取得
    pub fn get_params(&self, name: &str) -> Option<(f32, i32)> {
        self.quantization_params.get(name).copied()
    }

    /// Quantize a tensor using calibrated parameters
    /// キャリブレーション済みパラメータを使用してテンソルを量子化
    pub fn quantize<Q: QuantizableInteger>(
        &self, 
        name: &str, 
        tensor: &Tensor<T>
    ) -> RusTorchResult<QuantizedTensor<Q>>
    where
        T: super::Quantizable<QuantizedType = Q>,
    {
        if !self.calibrated {
            return Err(RusTorchError::TensorOp {
                message: "Quantizer not calibrated. Call calibrate() first.".to_string(),
                source: None,
            });
        }

        let (scale, zero_point) = self.get_params(name)
            .ok_or_else(|| RusTorchError::TensorOp {
                message: format!("No calibration parameters for '{}'", name),
                source: None,
            })?;

        // Manual quantization since trait bounds are too restrictive
        let quantized_data = tensor.data.mapv(|val| {
            let q_val = val.quantize(scale, zero_point);
            q_val
        });
        
        Ok(QuantizedTensor::new(
            quantized_data,
            scale,
            zero_point,
            tensor.device.clone(),
        ))
    }

    /// Check if quantizer is calibrated
    /// 量子化器がキャリブレーション済みかチェック
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Reset all observers
    /// 全ての観測器をリセット
    pub fn reset(&mut self) {
        for observer in self.observers.values_mut() {
            observer.reset();
        }
        self.quantization_params.clear();
        self.calibrated = false;
    }
}

impl<T: Float> Default for StaticQuantizer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for batch calibration
/// バッチキャリブレーション用の便利関数
pub fn calibrate_batch<T: Float>(
    data_batches: &[ArrayD<T>], 
    scheme: QuantizationScheme
) -> RusTorchResult<(f32, i32)> {
    let mut observer = MinMaxObserver::new();
    
    for batch in data_batches {
        observer.observe(batch);
    }
    
    observer.get_quantization_params(scheme)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_min_max_observer() {
        let mut observer = MinMaxObserver::new();
        
        let batch1 = Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        let batch2 = Array1::from_vec(vec![0.0f32, 4.0, 5.0]).into_dyn();
        
        observer.observe(&batch1);
        observer.observe(&batch2);
        
        assert_eq!(observer.min_val(), Some(0.0));
        assert_eq!(observer.max_val(), Some(5.0));
        
        let (scale, zero_point) = observer
            .get_quantization_params(QuantizationScheme::Symmetric)
            .unwrap();
        assert_eq!(zero_point, 0);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_histogram_observer() {
        let mut observer = HistogramObserver::new(10);
        
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        observer.observe(&data);
        
        assert_eq!(observer.min_val(), Some(1.0));
        assert_eq!(observer.max_val(), Some(5.0));
        
        let (range_min, range_max) = observer.compute_optimal_range((0.1, 0.9)).unwrap();
        assert!(range_min >= 1.0);
        assert!(range_max <= 5.0);
    }

    #[test]
    fn test_static_quantizer() {
        let mut quantizer = StaticQuantizer::<f32>::new();
        
        let observer = Box::new(MinMaxObserver::new());
        quantizer.add_observer("layer1".to_string(), observer);
        
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        quantizer.observe("layer1", &data).unwrap();
        
        assert!(!quantizer.is_calibrated());
        quantizer.calibrate(QuantizationScheme::Symmetric).unwrap();
        assert!(quantizer.is_calibrated());
        
        let params = quantizer.get_params("layer1");
        assert!(params.is_some());
        let (scale, zero_point) = params.unwrap();
        assert_eq!(zero_point, 0); // Symmetric quantization
        assert!(scale > 0.0);
    }

    #[test]
    fn test_batch_calibration() {
        let batch1 = Array1::from_vec(vec![1.0f32, 2.0]).into_dyn();
        let batch2 = Array1::from_vec(vec![3.0f32, 4.0]).into_dyn();
        let batches = vec![batch1, batch2];
        
        let (scale, zero_point) = calibrate_batch(&batches, QuantizationScheme::Asymmetric).unwrap();
        
        assert!(scale > 0.0);
        // For asymmetric quantization of positive values, zero_point should be negative
        assert!(zero_point <= 0);
    }

    #[test]
    fn test_observer_reset() {
        let mut observer = MinMaxObserver::new();
        
        let data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn();
        observer.observe(&data);
        
        assert!(observer.min_val().is_some());
        assert!(observer.max_val().is_some());
        
        observer.reset();
        
        assert!(observer.min_val().is_none());
        assert!(observer.max_val().is_none());
    }
}