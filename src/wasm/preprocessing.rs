//! Data preprocessing utilities for WASM
//! WASM用のデータ前処理ユーティリティ

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Data preprocessing utilities for neural networks
/// ニューラルネットワーク用のデータ前処理ユーティリティ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmPreprocessor;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmPreprocessor {
    /// Normalize data using min-max normalization: (x - min) / (max - min)
    /// min-max正規化を使用してデータを正規化: (x - min) / (max - min)
    #[wasm_bindgen]
    pub fn min_max_normalize(data: Vec<f32>, min_val: f32, max_val: f32) -> Vec<f32> {
        if (max_val - min_val).abs() < f32::EPSILON {
            return vec![0.0; data.len()];
        }
        
        let range = max_val - min_val;
        data.into_iter()
            .map(|x| (x - min_val) / range)
            .collect()
    }

    /// Standardize data using z-score normalization: (x - mean) / std
    /// z-score正規化を使用してデータを標準化: (x - mean) / std
    #[wasm_bindgen]
    pub fn z_score_normalize(data: Vec<f32>, mean: f32, std: f32) -> Vec<f32> {
        if std < f32::EPSILON {
            return vec![0.0; data.len()];
        }
        
        data.into_iter()
            .map(|x| (x - mean) / std)
            .collect()
    }

    /// Compute statistics (mean, std, min, max) for normalization
    /// 正規化用の統計値（平均、標準偏差、最小値、最大値）を計算
    #[wasm_bindgen]
    pub fn compute_stats(data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return vec![0.0, 1.0, 0.0, 0.0]; // mean, std, min, max
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        vec![mean, std, min, max]
    }

    /// One-hot encoding for categorical data
    /// カテゴリカルデータのワンホットエンコーディング
    #[wasm_bindgen]
    pub fn one_hot_encode(labels: Vec<u32>, num_classes: u32) -> Vec<f32> {
        let mut result = vec![0.0; labels.len() * num_classes as usize];
        
        for (i, &label) in labels.iter().enumerate() {
            if label < num_classes {
                let index = i * num_classes as usize + label as usize;
                result[index] = 1.0;
            }
        }
        
        result
    }

    /// Convert one-hot encoding back to labels
    /// ワンホットエンコーディングをラベルに戻す
    #[wasm_bindgen]
    pub fn one_hot_decode(one_hot: Vec<f32>, num_classes: u32) -> Vec<u32> {
        if num_classes == 0 {
            return Vec::new();
        }
        
        let batch_size = one_hot.len() / num_classes as usize;
        let mut result = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let start_idx = i * num_classes as usize;
            let end_idx = start_idx + num_classes as usize;
            
            let mut max_val = -f32::INFINITY;
            let mut max_idx = 0;
            
            for (j, &val) in one_hot[start_idx..end_idx].iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            
            result.push(max_idx as u32);
        }
        
        result
    }

    /// Data augmentation: add Gaussian noise
    /// データ拡張: ガウシアンノイズの追加
    #[wasm_bindgen]
    pub fn add_gaussian_noise(data: Vec<f32>, mean: f32, std: f32, seed: u32) -> Vec<f32> {
        // Simple linear congruential generator for deterministic noise
        let mut rng_state = seed as u64;
        
        data.into_iter()
            .map(|x| {
                // Generate pseudo-random number using LCG
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let uniform = (rng_state % 2147483647) as f32 / 2147483647.0;
                
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let uniform2 = (rng_state % 2147483647) as f32 / 2147483647.0;
                
                // Box-Muller transform for Gaussian noise
                let noise = if uniform > 0.0 && uniform2 > 0.0 {
                    (-2.0 * uniform.ln()).sqrt() * (2.0 * std::f32::consts::PI * uniform2).cos()
                } else {
                    0.0
                };
                
                x + mean + std * noise
            })
            .collect()
    }

    /// Train-test split for datasets
    /// データセットの訓練・テスト分割
    #[wasm_bindgen]
    pub fn train_test_split(
        features: Vec<f32>, 
        targets: Vec<f32>, 
        feature_size: usize,
        test_ratio: f32,
        seed: u32
    ) -> js_sys::Object {
        let num_samples = features.len() / feature_size;
        let test_size = (num_samples as f32 * test_ratio) as usize;
        let train_size = num_samples - test_size;
        
        // Simple deterministic shuffle based on seed
        let mut indices: Vec<usize> = (0..num_samples).collect();
        let mut rng_state = seed as u64;
        
        // Fisher-Yates shuffle with LCG
        for i in (1..num_samples).rev() {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (rng_state % (i as u64 + 1)) as usize;
            indices.swap(i, j);
        }
        
        // Split indices
        let train_indices = &indices[..train_size];
        let test_indices = &indices[train_size..];
        
        // Create train and test sets
        let mut train_features = Vec::with_capacity(train_size * feature_size);
        let mut train_targets = Vec::with_capacity(train_size);
        let mut test_features = Vec::with_capacity(test_size * feature_size);
        let mut test_targets = Vec::with_capacity(test_size);
        
        for &idx in train_indices {
            let start = idx * feature_size;
            let end = start + feature_size;
            train_features.extend_from_slice(&features[start..end]);
            train_targets.push(targets[idx]);
        }
        
        for &idx in test_indices {
            let start = idx * feature_size;
            let end = start + feature_size;
            test_features.extend_from_slice(&features[start..end]);
            test_targets.push(targets[idx]);
        }
        
        // Return as JavaScript object
        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"trainFeatures".into(), &js_sys::Array::from_iter(train_features.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
        js_sys::Reflect::set(&result, &"trainTargets".into(), &js_sys::Array::from_iter(train_targets.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
        js_sys::Reflect::set(&result, &"testFeatures".into(), &js_sys::Array::from_iter(test_features.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
        js_sys::Reflect::set(&result, &"testTargets".into(), &js_sys::Array::from_iter(test_targets.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
        
        result
    }

    /// Batch data for training
    /// 訓練用のデータバッチ化
    #[wasm_bindgen]
    pub fn create_batches(
        features: Vec<f32>,
        targets: Vec<f32>,
        feature_size: usize,
        batch_size: usize
    ) -> js_sys::Array {
        let num_samples = features.len() / feature_size;
        let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division
        
        let batches = js_sys::Array::new();
        
        for batch_idx in 0..num_batches {
            let start_sample = batch_idx * batch_size;
            let end_sample = (start_sample + batch_size).min(num_samples);
            let current_batch_size = end_sample - start_sample;
            
            let mut batch_features = Vec::with_capacity(current_batch_size * feature_size);
            let mut batch_targets = Vec::with_capacity(current_batch_size);
            
            for sample_idx in start_sample..end_sample {
                let feature_start = sample_idx * feature_size;
                let feature_end = feature_start + feature_size;
                batch_features.extend_from_slice(&features[feature_start..feature_end]);
                batch_targets.push(targets[sample_idx]);
            }
            
            let batch = js_sys::Object::new();
            js_sys::Reflect::set(&batch, &"features".into(), &js_sys::Array::from_iter(batch_features.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
            js_sys::Reflect::set(&batch, &"targets".into(), &js_sys::Array::from_iter(batch_targets.iter().map(|&x| JsValue::from_f64(x as f64)))).unwrap();
            js_sys::Reflect::set(&batch, &"batchSize".into(), &JsValue::from_f64(current_batch_size as f64)).unwrap();
            
            batches.push(&batch);
        }
        
        batches
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_min_max_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = WasmPreprocessor::min_max_normalize(data, 1.0, 5.0);
        assert_eq!(normalized, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[wasm_bindgen_test]
    fn test_z_score_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Mean = 3.0, Std ≈ 1.58
        let normalized = WasmPreprocessor::z_score_normalize(data, 3.0, 1.58);
        // Results should be approximately [-1.27, -0.63, 0.0, 0.63, 1.27]
        assert!((normalized[2] - 0.0).abs() < 0.01); // Middle value should be ~0
    }

    #[wasm_bindgen_test]
    fn test_one_hot_encode() {
        let labels = vec![0, 1, 2, 1];
        let encoded = WasmPreprocessor::one_hot_encode(labels, 3);
        let expected = vec![
            1.0, 0.0, 0.0,  // Class 0
            0.0, 1.0, 0.0,  // Class 1
            0.0, 0.0, 1.0,  // Class 2
            0.0, 1.0, 0.0,  // Class 1
        ];
        assert_eq!(encoded, expected);
    }

    #[wasm_bindgen_test]
    fn test_one_hot_decode() {
        let one_hot = vec![
            1.0, 0.0, 0.0,  // Class 0
            0.0, 1.0, 0.0,  // Class 1
            0.0, 0.0, 1.0,  // Class 2
        ];
        let decoded = WasmPreprocessor::one_hot_decode(one_hot, 3);
        assert_eq!(decoded, vec![0, 1, 2]);
    }
}