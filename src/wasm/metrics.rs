//! Model evaluation metrics for WASM
//! WASM用のモデル評価メトリクス

#[cfg(feature = "wasm")]
use std::collections::HashMap;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Model evaluation metrics calculator
/// モデル評価メトリクス計算機
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmMetrics;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmMetrics {
    /// Calculate accuracy for classification tasks
    /// 分類タスクの精度を計算
    #[wasm_bindgen]
    pub fn accuracy(predictions: Vec<u32>, targets: Vec<u32>) -> f32 {
        if predictions.len() != targets.len() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        correct as f32 / predictions.len() as f32
    }

    /// Calculate precision for binary classification
    /// バイナリ分類の適合率を計算
    #[wasm_bindgen]
    pub fn precision(predictions: Vec<u32>, targets: Vec<u32>, positive_class: u32) -> f32 {
        let mut true_positives = 0;
        let mut false_positives = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *pred == positive_class {
                if *target == positive_class {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }
        }

        if true_positives + false_positives == 0 {
            0.0
        } else {
            true_positives as f32 / (true_positives + false_positives) as f32
        }
    }

    /// Calculate recall for binary classification
    /// バイナリ分類の再現率を計算
    #[wasm_bindgen]
    pub fn recall(predictions: Vec<u32>, targets: Vec<u32>, positive_class: u32) -> f32 {
        let mut true_positives = 0;
        let mut false_negatives = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *target == positive_class {
                if *pred == positive_class {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }

        if true_positives + false_negatives == 0 {
            0.0
        } else {
            true_positives as f32 / (true_positives + false_negatives) as f32
        }
    }

    /// Calculate F1 score
    /// F1スコアを計算
    #[wasm_bindgen]
    pub fn f1_score(predictions: Vec<u32>, targets: Vec<u32>, positive_class: u32) -> f32 {
        let p = Self::precision(predictions.clone(), targets.clone(), positive_class);
        let r = Self::recall(predictions, targets, positive_class);

        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Calculate confusion matrix for multi-class classification
    /// 多クラス分類の混同行列を計算
    #[wasm_bindgen]
    pub fn confusion_matrix(
        predictions: Vec<u32>,
        targets: Vec<u32>,
        num_classes: u32,
    ) -> Vec<u32> {
        let mut matrix = vec![0u32; (num_classes * num_classes) as usize];

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if *pred < num_classes && *target < num_classes {
                let index = (*target * num_classes + *pred) as usize;
                matrix[index] += 1;
            }
        }

        matrix
    }

    /// Calculate Mean Absolute Error (MAE) for regression
    /// 回帰のための平均絶対誤差（MAE）を計算
    #[wasm_bindgen]
    pub fn mae(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            return f32::INFINITY;
        }

        let sum_abs_diff: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum();

        sum_abs_diff / predictions.len() as f32
    }

    /// Calculate Mean Squared Error (MSE) for regression
    /// 回帰のための平均二乗誤差（MSE）を計算
    #[wasm_bindgen]
    pub fn mse(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            return f32::INFINITY;
        }

        let sum_sq_diff: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum();

        sum_sq_diff / predictions.len() as f32
    }

    /// Calculate Root Mean Squared Error (RMSE) for regression
    /// 回帰のための平方根平均二乗誤差（RMSE）を計算
    #[wasm_bindgen]
    pub fn rmse(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        Self::mse(predictions, targets).sqrt()
    }

    /// Calculate R-squared coefficient for regression
    /// 回帰のための決定係数（R二乗）を計算
    #[wasm_bindgen]
    pub fn r2_score(predictions: Vec<f32>, targets: Vec<f32>) -> f32 {
        if predictions.len() != targets.len() {
            return f32::NEG_INFINITY;
        }

        let target_mean = targets.iter().sum::<f32>() / targets.len() as f32;

        let ss_res: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(pred, target)| (target - pred).powi(2))
            .sum();

        let ss_tot: f32 = targets
            .iter()
            .map(|target| (target - target_mean).powi(2))
            .sum();

        if ss_tot == 0.0 {
            1.0 // Perfect prediction
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }

    /// Calculate top-k accuracy for multi-class classification
    /// 多クラス分類のためのtop-k精度を計算
    #[wasm_bindgen]
    pub fn top_k_accuracy(logits: Vec<f32>, targets: Vec<u32>, num_classes: u32, k: u32) -> f32 {
        let batch_size = targets.len();
        if logits.len() != batch_size * num_classes as usize {
            return 0.0;
        }

        let mut correct = 0;

        for i in 0..batch_size {
            let start_idx = i * num_classes as usize;
            let end_idx = start_idx + num_classes as usize;
            let sample_logits = &logits[start_idx..end_idx];

            // Get top-k indices
            let mut indexed_logits: Vec<(usize, f32)> = sample_logits
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();

            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_classes: Vec<usize> = indexed_logits
                .iter()
                .take(k as usize)
                .map(|(idx, _)| *idx)
                .collect();

            if top_k_classes.contains(&(targets[i] as usize)) {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }

    /// Calculate comprehensive classification report
    /// 包括的な分類レポートを計算
    #[wasm_bindgen]
    pub fn classification_report(
        predictions: Vec<u32>,
        targets: Vec<u32>,
        num_classes: u32,
    ) -> js_sys::Object {
        let report = js_sys::Object::new();

        // Overall accuracy
        let overall_accuracy = Self::accuracy(predictions.clone(), targets.clone());
        js_sys::Reflect::set(
            &report,
            &"accuracy".into(),
            &JsValue::from_f64(overall_accuracy as f64),
        )
        .unwrap();

        // Per-class metrics
        let classes = js_sys::Object::new();

        for class in 0..num_classes {
            let precision = Self::precision(predictions.clone(), targets.clone(), class);
            let recall = Self::recall(predictions.clone(), targets.clone(), class);
            let f1 = Self::f1_score(predictions.clone(), targets.clone(), class);

            let class_metrics = js_sys::Object::new();
            js_sys::Reflect::set(
                &class_metrics,
                &"precision".into(),
                &JsValue::from_f64(precision as f64),
            )
            .unwrap();
            js_sys::Reflect::set(
                &class_metrics,
                &"recall".into(),
                &JsValue::from_f64(recall as f64),
            )
            .unwrap();
            js_sys::Reflect::set(&class_metrics, &"f1".into(), &JsValue::from_f64(f1 as f64))
                .unwrap();

            let class_key = format!("class_{}", class);
            js_sys::Reflect::set(&classes, &class_key.into(), &class_metrics).unwrap();
        }

        js_sys::Reflect::set(&report, &"classes".into(), &classes).unwrap();

        // Confusion matrix
        let confusion = Self::confusion_matrix(predictions, targets, num_classes);
        let confusion_array =
            js_sys::Array::from_iter(confusion.iter().map(|&x| JsValue::from_f64(x as f64)));
        js_sys::Reflect::set(&report, &"confusionMatrix".into(), &confusion_array).unwrap();

        report
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_accuracy() {
        let predictions = vec![0, 1, 1, 0];
        let targets = vec![0, 1, 0, 0];
        let acc = WasmMetrics::accuracy(predictions, targets);
        assert_eq!(acc, 0.75); // 3/4 correct
    }

    #[wasm_bindgen_test]
    fn test_precision_recall() {
        let predictions = vec![1, 1, 0, 1];
        let targets = vec![1, 0, 0, 1];

        let precision = WasmMetrics::precision(predictions.clone(), targets.clone(), 1);
        let recall = WasmMetrics::recall(predictions, targets, 1);

        // True positives: 2, False positives: 1, False negatives: 0
        assert_eq!(precision, 2.0 / 3.0);
        assert_eq!(recall, 1.0);
    }

    #[wasm_bindgen_test]
    fn test_mse_mae() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 1.5, 2.5];

        let mse = WasmMetrics::mse(predictions.clone(), targets.clone());
        let mae = WasmMetrics::mae(predictions, targets);

        // MSE = ((0.5)² + (0.5)² + (0.5)²) / 3 = 0.25
        assert!((mse - 0.25).abs() < 1e-5);

        // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert!((mae - 0.5).abs() < 1e-5);
    }

    #[wasm_bindgen_test]
    fn test_top_k_accuracy() {
        let logits = vec![
            0.1, 0.8, 0.1, // Prediction: class 1, Target: class 1 ✓
            0.7, 0.2, 0.1, // Prediction: class 0, Target: class 1, but class 1 in top-2 ✓
        ];
        let targets = vec![1, 1];

        let top1_acc = WasmMetrics::top_k_accuracy(logits.clone(), targets.clone(), 3, 1);
        let top2_acc = WasmMetrics::top_k_accuracy(logits, targets, 3, 2);

        assert_eq!(top1_acc, 0.5); // Only first sample correct for top-1
        assert_eq!(top2_acc, 1.0); // Both samples correct for top-2
    }
}
