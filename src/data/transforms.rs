//! Advanced data transformations
//! 高度なデータ変換
//!
//! This module provides comprehensive data transformation capabilities
//! for various data types including images, text, and numerical data.
//! このモジュールは画像、テキスト、数値データを含む様々なデータタイプの
//! 包括的なデータ変換機能を提供します。

use crate::tensor::Tensor;
use num_traits::Float;
use rand::Rng;
use std::collections::HashMap;

/// Base trait for all data transformations
/// すべてのデータ変換の基底トレイト
pub trait Transform<T: Float + 'static> {
    /// Apply transformation to input data
    /// 入力データに変換を適用
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String>;

    /// Get transformation name
    /// 変換名を取得
    fn name(&self) -> &str;

    /// Get transformation parameters
    /// 変換パラメータを取得
    fn parameters(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Compose multiple transformations in sequence
/// 複数の変換を順次組み合わせ
pub struct Compose<T: Float + 'static> {
    transforms: Vec<Box<dyn Transform<T> + Send + Sync>>,
    name: String,
}

impl<T: Float + 'static> Compose<T> {
    /// Create a new composition of transforms
    /// 新しい変換の組み合わせを作成
    pub fn new(transforms: Vec<Box<dyn Transform<T> + Send + Sync>>) -> Self {
        let names: Vec<String> = transforms.iter().map(|t| t.name().to_string()).collect();
        let name = format!("Compose[{}]", names.join(", "));

        Self { transforms, name }
    }
}

impl<T: Float + 'static> Transform<T> for Compose<T> {
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        self.transforms
            .iter()
            .try_fold(data.clone(), |acc, transform| transform.apply(&acc))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Random choice between multiple transforms
/// 複数の変換からランダム選択
pub struct RandomChoice<T: Float + 'static> {
    transforms: Vec<Box<dyn Transform<T> + Send + Sync>>,
    probabilities: Vec<f64>,
    name: String,
}

impl<T: Float + 'static> RandomChoice<T> {
    /// Create a new random choice transform
    /// 新しいランダム選択変換を作成
    pub fn new(
        transforms: Vec<Box<dyn Transform<T> + Send + Sync>>,
        probabilities: Option<Vec<f64>>,
    ) -> Result<Self, String> {
        if transforms.is_empty() {
            return Err("At least one transform must be provided".to_string());
        }

        let probs = if let Some(p) = probabilities {
            if p.len() != transforms.len() {
                return Err("Probabilities length must match transforms length".to_string());
            }
            let sum: f64 = p.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err("Probabilities must sum to 1.0".to_string());
            }
            p
        } else {
            let uniform_prob = 1.0 / transforms.len() as f64;
            vec![uniform_prob; transforms.len()]
        };

        let names: Vec<String> = transforms.iter().map(|t| t.name().to_string()).collect();
        let name = format!("RandomChoice[{}]", names.join(", "));

        Ok(Self {
            transforms,
            probabilities: probs,
            name,
        })
    }
}

impl<T: Float + 'static> Transform<T> for RandomChoice<T> {
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        let random_val: f64 = rand::thread_rng().gen();
        let mut cumulative = 0.0;

        for (i, &prob) in self.probabilities.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return self.transforms[i].apply(data);
            }
        }

        // Fallback to last transform
        self.transforms.last().unwrap().apply(data)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ===== NORMALIZATION TRANSFORMS =====

/// Standard normalization: (x - mean) / std
/// 標準正規化: (x - 平均) / 標準偏差
pub struct Normalize<T: Float + 'static> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Normalize<T> {
    /// Create a new normalize transform
    /// 新しい正規化変換を作成
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Result<Self, String> {
        if mean.len() != std.len() {
            return Err("Mean and std must have the same length".to_string());
        }
        if std.iter().any(|&s| s <= T::zero()) {
            return Err("All std values must be positive".to_string());
        }
        Ok(Self { mean, std })
    }

    /// Create normalization from dataset statistics
    /// データセット統計から正規化を作成
    pub fn from_dataset<D: crate::data::Dataset<T>>(dataset: &D) -> Result<Self, String> {
        if dataset.len() == 0 {
            return Err("Dataset is empty".to_string());
        }

        // Calculate mean and std from first few samples
        let mut sum = Vec::<T>::new();
        let mut sum_sq = Vec::<T>::new();
        let mut count = 0;

        for i in 0..std::cmp::min(1000, dataset.len()) {
            if let Some((features, _)) = dataset.get(i) {
                if let Some(data) = features.as_slice() {
                    if sum.is_empty() {
                        sum = vec![T::zero(); data.len()];
                        sum_sq = vec![T::zero(); data.len()];
                    }

                    for (j, &val) in data.iter().enumerate() {
                        if j < sum.len() {
                            sum[j] = sum[j] + val;
                            sum_sq[j] = sum_sq[j] + val * val;
                        }
                    }
                    count += 1;
                }
            }
        }

        if count == 0 || sum.is_empty() {
            return Err("Could not calculate statistics from dataset".to_string());
        }

        let count_t = T::from_usize(count).unwrap_or(T::one());
        let mean: Vec<T> = sum.into_iter().map(|s| s / count_t).collect();
        let variance: Vec<T> = sum_sq
            .into_iter()
            .zip(&mean)
            .map(|(sq, &m)| sq / count_t - m * m)
            .collect();
        let std: Vec<T> = variance.into_iter().map(|v| v.sqrt()).collect();

        Self::new(mean, std)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for Normalize<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        if self.mean.is_empty() || self.std.is_empty() {
            return Ok(data.clone());
        }

        // Apply channel-wise normalization if tensor has multiple channels
        let shape = data.shape();
        if let Some(data_slice) = data.as_slice() {
            let mut normalized_data = Vec::with_capacity(data_slice.len());

            match shape.len() {
                1 => {
                    // 1D tensor - apply normalization element-wise
                    for (i, &val) in data_slice.iter().enumerate() {
                        let mean_idx = i.min(self.mean.len() - 1);
                        let std_idx = i.min(self.std.len() - 1);
                        let normalized = (val - self.mean[mean_idx]) / self.std[std_idx];
                        normalized_data.push(normalized);
                    }
                }
                3 => {
                    // 3D tensor (C, H, W) - apply per-channel normalization
                    let channels = shape[0];
                    let height = shape[1];
                    let width = shape[2];

                    for c in 0..channels {
                        let mean_idx = c.min(self.mean.len() - 1);
                        let std_idx = c.min(self.std.len() - 1);
                        let mean_val = self.mean[mean_idx];
                        let std_val = self.std[std_idx];

                        for h in 0..height {
                            for w in 0..width {
                                let idx = c * height * width + h * width + w;
                                if idx < data_slice.len() {
                                    let normalized = (data_slice[idx] - mean_val) / std_val;
                                    normalized_data.push(normalized);
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Generic case - use first mean/std values
                    let mean_val = self.mean[0];
                    let std_val = self.std[0];
                    for &val in data_slice {
                        let normalized = (val - mean_val) / std_val;
                        normalized_data.push(normalized);
                    }
                }
            }

            Ok(Tensor::from_vec(normalized_data, shape.to_vec()))
        } else {
            Err("Cannot access tensor data for normalization".to_string())
        }
    }

    fn name(&self) -> &str {
        "Normalize"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("channels".to_string(), self.mean.len().to_string());
        params
    }
}

/// Min-Max normalization: (x - min) / (max - min)
/// Min-Max正規化: (x - 最小値) / (最大値 - 最小値)
pub struct MinMaxNormalize<T: Float + 'static> {
    min_val: T,
    max_val: T,
    target_min: T,
    target_max: T,
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> MinMaxNormalize<T> {
    /// Create a new min-max normalizer
    /// 新しいmin-max正規化器を作成
    pub fn new(min_val: T, max_val: T, target_min: T, target_max: T) -> Result<Self, String> {
        if min_val >= max_val {
            return Err("min_val must be less than max_val".to_string());
        }
        if target_min >= target_max {
            return Err("target_min must be less than target_max".to_string());
        }
        Ok(Self {
            min_val,
            max_val,
            target_min,
            target_max,
        })
    }

    /// Create standard [0, 1] normalizer
    /// 標準[0, 1]正規化器を作成
    pub fn zero_one(min_val: T, max_val: T) -> Result<Self, String> {
        Self::new(min_val, max_val, T::zero(), T::one())
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for MinMaxNormalize<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        let range = self.max_val - self.min_val;
        let target_range = self.target_max - self.target_min;

        if let Some(data_slice) = data.as_slice() {
            let normalized_data: Vec<T> = data_slice
                .iter()
                .map(|&val| {
                    let normalized = (val - self.min_val) / range;
                    let scaled = normalized * target_range + self.target_min;
                    scaled.max(self.target_min).min(self.target_max) // Clamp to target range
                })
                .collect();

            Ok(Tensor::from_vec(normalized_data, data.shape().to_vec()))
        } else {
            Err("Cannot access tensor data for min-max normalization".to_string())
        }
    }

    fn name(&self) -> &str {
        "MinMaxNormalize"
    }
}

// ===== IMAGE TRANSFORMS =====

/// Random horizontal flip
/// ランダム水平反転
pub struct RandomHorizontalFlip<T: Float + 'static> {
    probability: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> RandomHorizontalFlip<T> {
    /// Create a new random horizontal flip transform
    /// 新しいランダム水平反転変換を作成
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.max(0.0).min(1.0),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for RandomHorizontalFlip<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        if rand::thread_rng().gen::<f64>() > self.probability {
            return Ok(data.clone());
        }

        let shape = data.shape();
        if shape.len() != 3 {
            return Err("RandomHorizontalFlip requires 3D tensor (C, H, W)".to_string());
        }

        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        if let Some(data_slice) = data.as_slice() {
            let mut flipped_data = Vec::with_capacity(data_slice.len());

            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let original_idx = c * height * width + h * width + w;
                        let flipped_w = width - 1 - w;
                        let flipped_idx = c * height * width + h * width + flipped_w;

                        if original_idx < data_slice.len() {
                            flipped_data.push(data_slice[original_idx]);
                        } else {
                            flipped_data.push(T::zero());
                        }
                    }
                }
            }

            Ok(Tensor::from_vec(flipped_data, shape.to_vec()))
        } else {
            Err("Cannot access tensor data for horizontal flip".to_string())
        }
    }

    fn name(&self) -> &str {
        "RandomHorizontalFlip"
    }
}

/// Random rotation
/// ランダム回転
pub struct RandomRotation<T: Float + 'static> {
    max_angle: f64, // in degrees
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> RandomRotation<T> {
    /// Create a new random rotation transform
    /// 新しいランダム回転変換を作成
    pub fn new(max_angle: f64) -> Self {
        Self {
            max_angle: max_angle.abs(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for RandomRotation<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        let angle = rand::thread_rng().gen_range(-self.max_angle..=self.max_angle);
        let shape = data.shape();

        if shape.len() != 3 {
            return Err("RandomRotation requires 3D tensor (C, H, W)".to_string());
        }

        // For simplicity, if rotation angle is small, return original
        // In a full implementation, this would perform actual rotation
        if angle.abs() < 1.0 {
            Ok(data.clone())
        } else {
            // Placeholder: return original data
            // Real implementation would apply rotation matrix
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "RandomRotation"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("max_angle".to_string(), self.max_angle.to_string());
        params
    }
}

/// Center crop
/// 中央クロップ
pub struct CenterCrop<T: Float + 'static> {
    output_size: (usize, usize), // (height, width)
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> CenterCrop<T> {
    /// Create a new center crop transform
    /// 新しい中央クロップ変換を作成
    pub fn new(output_size: (usize, usize)) -> Self {
        Self {
            output_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for CenterCrop<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        let shape = data.shape();
        if shape.len() != 3 {
            return Err("CenterCrop requires 3D tensor (C, H, W)".to_string());
        }

        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        let (target_height, target_width) = self.output_size;

        if target_height > height || target_width > width {
            return Err("Target size cannot be larger than input size".to_string());
        }

        let start_h = (height - target_height) / 2;
        let start_w = (width - target_width) / 2;

        if let Some(data_slice) = data.as_slice() {
            let mut cropped_data = Vec::new();

            for c in 0..channels {
                for h in start_h..(start_h + target_height) {
                    for w in start_w..(start_w + target_width) {
                        let idx = c * height * width + h * width + w;
                        if idx < data_slice.len() {
                            cropped_data.push(data_slice[idx]);
                        } else {
                            cropped_data.push(T::zero());
                        }
                    }
                }
            }

            Ok(Tensor::from_vec(
                cropped_data,
                vec![channels, target_height, target_width],
            ))
        } else {
            Err("Cannot access tensor data for center crop".to_string())
        }
    }

    fn name(&self) -> &str {
        "CenterCrop"
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("height".to_string(), self.output_size.0.to_string());
        params.insert("width".to_string(), self.output_size.1.to_string());
        params
    }
}

// ===== NOISE AND AUGMENTATION TRANSFORMS =====

/// Add Gaussian noise
/// ガウシアンノイズ追加
pub struct AddGaussianNoise<T: Float + 'static> {
    mean: T,
    std: T,
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> AddGaussianNoise<T> {
    /// Create a new Gaussian noise transform
    /// 新しいガウシアンノイズ変換を作成
    pub fn new(mean: T, std: T) -> Self {
        Self { mean, std }
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for AddGaussianNoise<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        if let Some(data_slice) = data.as_slice() {
            let mut rng = rand::thread_rng();
            let noisy_data: Vec<T> = data_slice
                .iter()
                .map(|&val| {
                    // Simple Box-Muller approximation for Gaussian noise
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                    if let (Some(noise_t), Some(mean_f), Some(std_f)) =
                        (T::from_f64(noise), self.mean.to_f64(), self.std.to_f64())
                    {
                        let scaled_noise = T::from_f64(noise * std_f + mean_f).unwrap_or(T::zero());
                        val + scaled_noise
                    } else {
                        val
                    }
                })
                .collect();

            Ok(Tensor::from_vec(noisy_data, data.shape().to_vec()))
        } else {
            Err("Cannot access tensor data for adding noise".to_string())
        }
    }

    fn name(&self) -> &str {
        "AddGaussianNoise"
    }
}

/// Random brightness adjustment
/// ランダム明度調整
pub struct RandomBrightness<T: Float + 'static> {
    factor_range: (T, T),
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> RandomBrightness<T> {
    /// Create a new random brightness transform
    /// 新しいランダム明度変換を作成
    pub fn new(factor_range: (T, T)) -> Result<Self, String> {
        if factor_range.0 > factor_range.1 {
            return Err("factor_range.0 must be <= factor_range.1".to_string());
        }
        Ok(Self { factor_range })
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for RandomBrightness<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        let factor = {
            let mut rng = rand::thread_rng();
            if let (Some(min), Some(max)) =
                (self.factor_range.0.to_f64(), self.factor_range.1.to_f64())
            {
                let random_factor = rng.gen_range(min..=max);
                T::from_f64(random_factor).unwrap_or(T::one())
            } else {
                T::one()
            }
        };

        if let Some(data_slice) = data.as_slice() {
            let brightened_data: Vec<T> = data_slice.iter().map(|&val| val * factor).collect();

            Ok(Tensor::from_vec(brightened_data, data.shape().to_vec()))
        } else {
            Err("Cannot access tensor data for brightness adjustment".to_string())
        }
    }

    fn name(&self) -> &str {
        "RandomBrightness"
    }
}

// ===== TEXT TRANSFORMS =====

/// Token dropout for text data
/// テキストデータ用トークンドロップアウト
pub struct TokenDropout<T: Float + 'static> {
    dropout_rate: f64,
    replacement_token: T,
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> TokenDropout<T> {
    /// Create a new token dropout transform
    /// 新しいトークンドロップアウト変換を作成
    pub fn new(dropout_rate: f64, replacement_token: T) -> Self {
        Self {
            dropout_rate: dropout_rate.max(0.0).min(1.0),
            replacement_token,
        }
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Transform<T>
    for TokenDropout<T>
{
    fn apply(&self, data: &Tensor<T>) -> Result<Tensor<T>, String> {
        if let Some(data_slice) = data.as_slice() {
            let mut rng = rand::thread_rng();
            let dropped_data: Vec<T> = data_slice
                .iter()
                .map(|&val| {
                    if rng.gen::<f64>() < self.dropout_rate {
                        self.replacement_token
                    } else {
                        val
                    }
                })
                .collect();

            Ok(Tensor::from_vec(dropped_data, data.shape().to_vec()))
        } else {
            Err("Cannot access tensor data for token dropout".to_string())
        }
    }

    fn name(&self) -> &str {
        "TokenDropout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let normalize = Normalize::new(vec![2.5], vec![1.5]).unwrap();

        let result = normalize.apply(&data).unwrap();
        let result_data = result.as_slice().unwrap();

        // Expected: [(1-2.5)/1.5, (2-2.5)/1.5, (3-2.5)/1.5, (4-2.5)/1.5]
        //         = [-1.0, -1/3, 1/3, 1.0]
        assert!((result_data[0] - (-1.0)).abs() < 1e-6);
        assert!((result_data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_normalize() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let normalize = MinMaxNormalize::zero_one(1.0, 4.0).unwrap();

        let result = normalize.apply(&data).unwrap();
        let result_data = result.as_slice().unwrap();

        // Expected: [0.0, 1/3, 2/3, 1.0]
        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compose() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let transforms: Vec<Box<dyn Transform<f32> + Send + Sync>> = vec![
            Box::new(MinMaxNormalize::zero_one(1.0, 4.0).unwrap()),
            Box::new(AddGaussianNoise::new(0.0, 0.01)),
        ];

        let compose = Compose::new(transforms);
        let result = compose.apply(&data).unwrap();

        assert_eq!(result.shape(), data.shape());
        assert_eq!(compose.name(), "Compose[MinMaxNormalize, AddGaussianNoise]");
    }

    #[test]
    fn test_random_choice() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let transforms: Vec<Box<dyn Transform<f32> + Send + Sync>> = vec![
            Box::new(MinMaxNormalize::zero_one(1.0, 4.0).unwrap()),
            Box::new(AddGaussianNoise::new(0.0, 0.1)),
        ];

        let random_choice = RandomChoice::new(transforms, Some(vec![0.5, 0.5])).unwrap();
        let result = random_choice.apply(&data).unwrap();

        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_center_crop() {
        let data = Tensor::from_vec(
            vec![1.0; 3 * 4 * 4], // 3 channels, 4x4 image
            vec![3, 4, 4],
        );

        let crop = CenterCrop::new((2, 2));
        let result = crop.apply(&data).unwrap();

        assert_eq!(result.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_random_horizontal_flip() {
        let data = Tensor::from_vec(
            (0..12).map(|i| i as f32).collect(), // 1 channel, 3x4 image
            vec![1, 3, 4],
        );

        let flip = RandomHorizontalFlip::new(1.0); // Always flip
        let result = flip.apply(&data).unwrap();

        assert_eq!(result.shape(), data.shape());
    }

    #[test]
    fn test_token_dropout() {
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let dropout = TokenDropout::new(0.5, 0.0); // 50% dropout, replace with 0

        let result = dropout.apply(&data).unwrap();
        let result_data = result.as_slice().unwrap();

        // Some tokens should be dropped (replaced with 0)
        let zero_count = result_data.iter().filter(|&&x| x == 0.0).count();
        assert!(zero_count > 0); // At least some should be dropped
    }
}
