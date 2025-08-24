//! Summary data structures for TensorBoard
//! TensorBoard用サマリーデータ構造

use serde::{Serialize, Deserialize};
use super::{ImageData, GraphDef};

/// Summary for TensorBoard logging
/// TensorBoardログ用サマリー
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// Tag name
    pub tag: String,
    /// Step number
    pub step: usize,
    /// Summary value
    pub value: SummaryValue,
}

/// Types of summary values
/// サマリー値の種類
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SummaryValue {
    /// Scalar value
    Scalar(f32),
    /// Histogram data
    Histogram(HistogramData),
    /// Image data
    Image(ImageSummary),
    /// Text data
    Text(String),
    /// Graph definition
    Graph(GraphDef),
    /// PR curve data
    PrCurve(PrCurveData),
    /// Audio data
    Audio(AudioSummary),
}

/// Histogram data for TensorBoard
/// TensorBoard用ヒストグラムデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    /// Number of values
    pub count: usize,
    /// Sum of values
    pub sum: f64,
    /// Sum of squares
    pub sum_squares: f64,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Histogram buckets
    pub buckets: Vec<HistogramBucket>,
}

/// Histogram bucket
/// ヒストグラムバケット
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Upper edge of bucket
    pub edge: f32,
    /// Count in bucket
    pub count: usize,
}

/// Image summary data
/// 画像サマリーデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSummary {
    /// Image height
    pub height: u32,
    /// Image width
    pub width: u32,
    /// Number of channels
    pub channels: u32,
    /// Encoded image data (PNG format)
    pub encoded_image: Vec<u8>,
}

/// PR curve data
/// PR曲線データ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrCurveData {
    /// True positive counts
    pub tp: Vec<usize>,
    /// False positive counts
    pub fp: Vec<usize>,
    /// True negative counts
    pub tn: Vec<usize>,
    /// False negative counts
    pub fn_: Vec<usize>,
    /// Precision values
    pub precision: Vec<f32>,
    /// Recall values
    pub recall: Vec<f32>,
    /// Threshold values
    pub thresholds: Vec<f32>,
}

/// Audio summary data
/// 音声サマリーデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSummary {
    /// Sample rate
    pub sample_rate: u32,
    /// Audio data
    pub audio_data: Vec<f32>,
    /// Content type
    pub content_type: String,
}

impl Summary {
    /// Create scalar summary
    pub fn scalar(tag: &str, value: f32, step: usize) -> Self {
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::Scalar(value),
        }
    }
    
    /// Create histogram summary
    pub fn histogram(tag: &str, values: &[f32], step: usize) -> Self {
        let histogram_data = create_histogram(values);
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::Histogram(histogram_data),
        }
    }
    
    /// Create image summary
    pub fn image(tag: &str, image: &ImageData, step: usize) -> Self {
        let encoded_image = encode_image_as_png(image);
        let image_summary = ImageSummary {
            height: image.height,
            width: image.width,
            channels: image.channels,
            encoded_image,
        };
        
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::Image(image_summary),
        }
    }
    
    /// Create text summary
    pub fn text(tag: &str, text: &str, step: usize) -> Self {
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::Text(text.to_string()),
        }
    }
    
    /// Create graph summary
    pub fn graph(graph: &GraphDef) -> Self {
        Self {
            tag: "graph".to_string(),
            step: 0,
            value: SummaryValue::Graph(graph.clone()),
        }
    }
    
    /// Create PR curve summary
    pub fn pr_curve(tag: &str, labels: &[bool], predictions: &[f32], step: usize) -> Self {
        let pr_data = create_pr_curve(labels, predictions);
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::PrCurve(pr_data),
        }
    }
    
    /// Create audio summary
    pub fn audio(tag: &str, audio_data: &[f32], sample_rate: u32, step: usize) -> Self {
        let audio_summary = AudioSummary {
            sample_rate,
            audio_data: audio_data.to_vec(),
            content_type: "audio/wav".to_string(),
        };
        
        Self {
            tag: tag.to_string(),
            step,
            value: SummaryValue::Audio(audio_summary),
        }
    }
}

/// Create histogram from values
fn create_histogram(values: &[f32]) -> HistogramData {
    if values.is_empty() {
        return HistogramData {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: 0.0,
            max: 0.0,
            buckets: Vec::new(),
        };
    }
    
    let count = values.len();
    let sum: f64 = values.iter().map(|&x| x as f64).sum();
    let sum_squares: f64 = values.iter().map(|&x| (x as f64).powi(2)).sum();
    let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Create histogram buckets
    let num_buckets = 30;
    let bucket_width = (max - min) / num_buckets as f32;
    let mut buckets = Vec::new();
    
    for i in 0..num_buckets {
        let edge = min + (i + 1) as f32 * bucket_width;
        let count = values.iter()
            .filter(|&&x| x <= edge && (i == 0 || x > min + i as f32 * bucket_width))
            .count();
        
        buckets.push(HistogramBucket { edge, count });
    }
    
    HistogramData {
        count,
        sum,
        sum_squares,
        min,
        max,
        buckets,
    }
}

/// Create PR curve from labels and predictions
fn create_pr_curve(labels: &[bool], predictions: &[f32]) -> PrCurveData {
    // Sort by prediction scores (descending)
    let mut sorted_pairs: Vec<_> = labels.iter()
        .zip(predictions.iter())
        .enumerate()
        .collect();
    sorted_pairs.sort_by(|a, b| b.1.1.partial_cmp(a.1.1).unwrap());
    
    let total_positive = labels.iter().filter(|&&x| x).count();
    let total_negative = labels.len() - total_positive;
    
    let mut tp = Vec::new();
    let mut fp = Vec::new();
    let mut tn = Vec::new();
    let mut fn_ = Vec::new();
    let mut precision = Vec::new();
    let mut recall = Vec::new();
    let mut thresholds = Vec::new();
    
    let mut true_positives = 0;
    let mut false_positives = 0;
    
    for (_i, &(_, (&label, &pred))) in sorted_pairs.iter().enumerate() {
        if label {
            true_positives += 1;
        } else {
            false_positives += 1;
        }
        
        let true_negatives = total_negative - false_positives;
        let false_negatives = total_positive - true_positives;
        
        let prec = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            1.0
        };
        
        let rec = if total_positive > 0 {
            true_positives as f32 / total_positive as f32
        } else {
            0.0
        };
        
        tp.push(true_positives);
        fp.push(false_positives);
        tn.push(true_negatives);
        fn_.push(false_negatives);
        precision.push(prec);
        recall.push(rec);
        thresholds.push(pred);
    }
    
    PrCurveData {
        tp,
        fp,
        tn,
        fn_,
        precision,
        recall,
        thresholds,
    }
}

/// Encode image as PNG (simplified - real implementation needs proper PNG encoder)
fn encode_image_as_png(image: &ImageData) -> Vec<u8> {
    // Simplified implementation - just return the raw data
    // Real implementation should use a proper PNG encoder like `png` crate
    image.data.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scalar_summary() {
        let summary = Summary::scalar("test_loss", 0.5, 100);
        
        assert_eq!(summary.tag, "test_loss");
        assert_eq!(summary.step, 100);
        
        if let SummaryValue::Scalar(value) = summary.value {
            assert_eq!(value, 0.5);
        } else {
            panic!("Expected scalar value");
        }
    }
    
    #[test]
    fn test_histogram_creation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let histogram = create_histogram(&values);
        
        assert_eq!(histogram.count, 5);
        assert_eq!(histogram.sum, 15.0);
        assert_eq!(histogram.min, 1.0);
        assert_eq!(histogram.max, 5.0);
        assert!(!histogram.buckets.is_empty());
    }
    
    #[test]
    fn test_pr_curve_creation() {
        let labels = vec![true, false, true, false, true];
        let predictions = vec![0.9, 0.1, 0.8, 0.3, 0.7];
        
        let pr_curve = create_pr_curve(&labels, &predictions);
        
        assert_eq!(pr_curve.precision.len(), predictions.len());
        assert_eq!(pr_curve.recall.len(), predictions.len());
        assert_eq!(pr_curve.thresholds.len(), predictions.len());
    }
}