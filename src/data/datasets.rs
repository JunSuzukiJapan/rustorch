//! Advanced dataset implementations
//! 高度なデータセット実装
//!
//! This module provides various dataset types for different data formats
//! このモジュールは様々なデータ形式に対応するデータセットタイプを提供します

#![allow(deprecated)] // Allow deprecated APIs for backward compatibility

use super::LegacyDataset;
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// CSV Dataset for loading tabular data
/// 表形式データを読み込むためのCSVデータセット
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 Dataset API with custom implementations"
)]
pub struct CSVDataset<T: Float + num_traits::FromPrimitive> {
    features: Vec<Tensor<T>>,
    targets: Vec<Tensor<T>>,
    feature_names: Vec<String>,
    target_names: Vec<String>,
    metadata: HashMap<String, String>,
}

impl<T: Float + num_traits::FromPrimitive + std::str::FromStr + 'static> CSVDataset<T> {
    /// Create a new CSV dataset from file
    /// ファイルから新しいCSVデータセットを作成
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        feature_cols: &[usize],
        target_cols: &[usize],
        has_header: bool,
        delimiter: char,
    ) -> RusTorchResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut feature_names = Vec::new();
        let mut target_names = Vec::new();

        // Parse header if present
        if has_header {
            if let Some(Ok(header_line)) = lines.next() {
                let headers: Vec<&str> = header_line.split(delimiter).collect();
                for &col in feature_cols {
                    if col < headers.len() {
                        feature_names.push(headers[col].to_string());
                    }
                }
                for &col in target_cols {
                    if col < headers.len() {
                        target_names.push(headers[col].to_string());
                    }
                }
            }
        } else {
            // Generate default column names
            for i in feature_cols {
                feature_names.push(format!("feature_{}", i));
            }
            for i in target_cols {
                target_names.push(format!("target_{}", i));
            }
        }

        let mut features = Vec::new();
        let mut targets = Vec::new();

        // Parse data lines
        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.split(delimiter).collect();

            // Parse feature values
            let mut feature_data = Vec::new();
            for &col in feature_cols {
                if col < values.len() {
                    if let Ok(val) = values[col].trim().parse::<f64>() {
                        if let Some(converted) = T::from_f64(val) {
                            feature_data.push(converted);
                        } else {
                            feature_data.push(T::zero());
                        }
                    } else {
                        feature_data.push(T::zero());
                    }
                }
            }

            // Parse target values
            let mut target_data = Vec::new();
            for &col in target_cols {
                if col < values.len() {
                    if let Ok(val) = values[col].trim().parse::<f64>() {
                        if let Some(converted) = T::from_f64(val) {
                            target_data.push(converted);
                        } else {
                            target_data.push(T::zero());
                        }
                    } else {
                        target_data.push(T::zero());
                    }
                }
            }

            if !feature_data.is_empty() && !target_data.is_empty() {
                features.push(Tensor::from_vec(feature_data, vec![feature_cols.len()]));
                targets.push(Tensor::from_vec(target_data, vec![target_cols.len()]));
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "csv".to_string());
        metadata.insert("delimiter".to_string(), delimiter.to_string());
        metadata.insert("samples".to_string(), features.len().to_string());

        Ok(CSVDataset {
            features,
            targets,
            feature_names,
            target_names,
            metadata,
        })
    }

    /// Get feature column names
    /// 特徴量カラム名を取得
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Get target column names
    /// ターゲットカラム名を取得
    pub fn target_names(&self) -> &[String] {
        &self.target_names
    }

    /// Get dataset metadata
    /// データセットメタデータを取得
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Split dataset into train and test sets
    /// データセットを訓練とテストセットに分割
    pub fn train_test_split(&self, train_ratio: f64) -> (Self, Self) {
        let split_idx = (self.features.len() as f64 * train_ratio) as usize;

        let train_features = self.features[..split_idx].to_vec();
        let train_targets = self.targets[..split_idx].to_vec();
        let test_features = self.features[split_idx..].to_vec();
        let test_targets = self.targets[split_idx..].to_vec();

        let train_dataset = CSVDataset {
            features: train_features,
            targets: train_targets,
            feature_names: self.feature_names.clone(),
            target_names: self.target_names.clone(),
            metadata: {
                let mut meta = self.metadata.clone();
                meta.insert("split".to_string(), "train".to_string());
                meta
            },
        };

        let test_dataset = CSVDataset {
            features: test_features,
            targets: test_targets,
            feature_names: self.feature_names.clone(),
            target_names: self.target_names.clone(),
            metadata: {
                let mut meta = self.metadata.clone();
                meta.insert("split".to_string(), "test".to_string());
                meta
            },
        };

        (train_dataset, test_dataset)
    }
}

impl<T: Float + num_traits::FromPrimitive> LegacyDataset<T> for CSVDataset<T> {
    fn len(&self) -> usize {
        self.features.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        if index < self.len() {
            Some((self.features[index].clone(), self.targets[index].clone()))
        } else {
            None
        }
    }
}

/// Image dataset for computer vision tasks
/// コンピュータビジョンタスク用画像データセット
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 Dataset API with custom implementations"
)]
pub struct ImageDataset<T: Float> {
    image_paths: Vec<PathBuf>,
    labels: Vec<Tensor<T>>,
    transform: Option<Box<dyn Fn(&[u8]) -> Result<Vec<T>, String> + Send + Sync>>,
    class_names: Vec<String>,
    image_size: (usize, usize), // (height, width)
    channels: usize,
}

impl<T: Float + num_traits::FromPrimitive + 'static> ImageDataset<T> {
    /// Create a new image dataset
    /// 新しい画像データセットを作成
    pub fn new(
        image_paths: Vec<PathBuf>,
        labels: Vec<usize>,
        class_names: Vec<String>,
        image_size: (usize, usize),
        channels: usize,
    ) -> Result<Self, String> {
        if image_paths.len() != labels.len() {
            return Err("Image paths and labels must have the same length".to_string());
        }

        let tensor_labels: Vec<Tensor<T>> = labels
            .into_iter()
            .map(|label| {
                if let Some(label_val) = T::from_usize(label) {
                    Tensor::from_vec(vec![label_val], vec![1])
                } else {
                    Tensor::from_vec(vec![T::zero()], vec![1])
                }
            })
            .collect();

        Ok(ImageDataset {
            image_paths,
            labels: tensor_labels,
            transform: None,
            class_names,
            image_size,
            channels,
        })
    }

    /// Create dataset from directory structure
    /// ディレクトリ構造からデータセットを作成
    pub fn from_directory<P: AsRef<Path>>(
        root_dir: P,
        image_size: (usize, usize),
        channels: usize,
    ) -> RusTorchResult<Self> {
        let root_path = root_dir.as_ref();
        let mut image_paths = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();

        // Collect class directories
        let mut class_to_idx = HashMap::new();
        for entry in std::fs::read_dir(root_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(class_name) = path.file_name().and_then(|n| n.to_str()) {
                    class_to_idx.insert(class_name.to_string(), class_names.len());
                    class_names.push(class_name.to_string());
                }
            }
        }

        // Collect image files
        for (class_name, &class_idx) in &class_to_idx {
            let class_dir = root_path.join(class_name);
            for entry in std::fs::read_dir(class_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                            image_paths.push(path);
                            labels.push(class_idx);
                        }
                    }
                }
            }
        }

        Ok(Self::new(
            image_paths,
            labels,
            class_names,
            image_size,
            channels,
        )?)
    }

    /// Set image transform function
    /// 画像変換関数を設定
    pub fn set_transform<F>(&mut self, transform: F)
    where
        F: Fn(&[u8]) -> Result<Vec<T>, String> + Send + Sync + 'static,
    {
        self.transform = Some(Box::new(transform));
    }

    /// Get class names
    /// クラス名を取得
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// Load and process image
    /// 画像を読み込み・処理
    fn load_image(&self, path: &Path) -> Result<Tensor<T>, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read image: {}", e))?;

        let image_data = if let Some(ref transform) = self.transform {
            transform(&buffer)?
        } else {
            // Simple placeholder transformation - convert bytes to normalized float values
            buffer
                .into_iter()
                .map(|b| T::from_f32(b as f32 / 255.0).unwrap_or(T::zero()))
                .collect()
        };

        // Reshape to expected dimensions
        let expected_size = self.channels * self.image_size.0 * self.image_size.1;
        let mut final_data = image_data;

        // Resize or pad data to expected size
        final_data.resize(expected_size, T::zero());

        Ok(Tensor::from_vec(
            final_data,
            vec![self.channels, self.image_size.0, self.image_size.1],
        ))
    }
}

impl<T: Float + num_traits::FromPrimitive + 'static> LegacyDataset<T> for ImageDataset<T> {
    fn len(&self) -> usize {
        self.image_paths.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        if index < self.len() {
            match self.load_image(&self.image_paths[index]) {
                Ok(image) => Some((image, self.labels[index].clone())),
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

/// Text dataset for NLP tasks
/// NLPタスク用テキストデータセット
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 Dataset API with custom implementations"
)]
pub struct TextDataset<T: Float> {
    texts: Vec<String>,
    labels: Vec<Tensor<T>>,
    vocab: HashMap<String, usize>,
    max_length: usize,
    tokenizer: Box<dyn Fn(&str) -> Vec<String> + Send + Sync>,
}

impl<T: Float + num_traits::FromPrimitive + 'static> TextDataset<T> {
    /// Create a new text dataset
    /// 新しいテキストデータセットを作成
    pub fn new(texts: Vec<String>, labels: Vec<usize>, max_length: usize) -> Result<Self, String> {
        if texts.len() != labels.len() {
            return Err("Texts and labels must have the same length".to_string());
        }

        // Build vocabulary
        let mut vocab = HashMap::new();
        vocab.insert("<PAD>".to_string(), 0);
        vocab.insert("<UNK>".to_string(), 1);

        let simple_tokenizer = |text: &str| -> Vec<String> {
            text.split_whitespace().map(|s| s.to_lowercase()).collect()
        };

        let mut word_count = HashMap::new();
        for text in &texts {
            for token in simple_tokenizer(text) {
                *word_count.entry(token).or_insert(0) += 1;
            }
        }

        // Add frequent words to vocabulary
        let mut sorted_words: Vec<_> = word_count.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));

        for (word, _) in sorted_words {
            if vocab.len() < 10000 {
                // Limit vocabulary size
                vocab.insert(word, vocab.len());
            }
        }

        let tensor_labels: Vec<Tensor<T>> = labels
            .into_iter()
            .map(|label| {
                if let Some(label_val) = T::from_usize(label) {
                    Tensor::from_vec(vec![label_val], vec![1])
                } else {
                    Tensor::from_vec(vec![T::zero()], vec![1])
                }
            })
            .collect();

        Ok(TextDataset {
            texts,
            labels: tensor_labels,
            vocab,
            max_length,
            tokenizer: Box::new(simple_tokenizer),
        })
    }

    /// Set custom tokenizer
    /// カスタムトークナイザーを設定
    pub fn set_tokenizer<F>(&mut self, tokenizer: F)
    where
        F: Fn(&str) -> Vec<String> + Send + Sync + 'static,
    {
        self.tokenizer = Box::new(tokenizer);
    }

    /// Get vocabulary
    /// 語彙を取得
    pub fn vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    /// Encode text to token indices
    /// テキストをトークンインデックスにエンコード
    fn encode_text(&self, text: &str) -> Vec<T> {
        let tokens = (self.tokenizer)(text);
        let mut encoded = Vec::new();

        for token in tokens.into_iter().take(self.max_length) {
            let idx = self.vocab.get(&token).unwrap_or(&1); // Use <UNK> if not found
            if let Some(idx_val) = T::from_usize(*idx) {
                encoded.push(idx_val);
            } else {
                encoded.push(T::one()); // <UNK>
            }
        }

        // Pad to max_length
        while encoded.len() < self.max_length {
            encoded.push(T::zero()); // <PAD>
        }

        encoded
    }
}

impl<T: Float + num_traits::FromPrimitive + 'static> LegacyDataset<T> for TextDataset<T> {
    fn len(&self) -> usize {
        self.texts.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        if index < self.len() {
            let encoded_text = self.encode_text(&self.texts[index]);
            let text_tensor = Tensor::from_vec(encoded_text, vec![self.max_length]);
            Some((text_tensor, self.labels[index].clone()))
        } else {
            None
        }
    }
}

/// Memory-mapped dataset for large files
/// 大容量ファイル用メモリマップドデータセット
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 Dataset API with custom implementations"
)]
pub struct MemoryMappedDataset<T: Float> {
    _file_path: PathBuf,
    _data_type: std::marker::PhantomData<T>,
    _sample_size: usize,
    _num_samples: usize,
    // Note: In a real implementation, this would use memmap2 crate
    // 実際の実装では memmap2 クレートを使用します
    _placeholder: Vec<u8>,
}

impl<T: Float + num_traits::FromPrimitive + 'static> MemoryMappedDataset<T> {
    /// Create a new memory-mapped dataset
    /// 新しいメモリマップドデータセットを作成
    pub fn new<P: AsRef<Path>>(file_path: P, sample_size: usize) -> RusTorchResult<Self> {
        let path = file_path.as_ref().to_path_buf();
        let metadata = std::fs::metadata(&path)?;
        let file_size = metadata.len() as usize;
        let num_samples = file_size / (sample_size * std::mem::size_of::<T>());

        Ok(MemoryMappedDataset {
            _file_path: path,
            _data_type: std::marker::PhantomData,
            _sample_size: sample_size,
            _num_samples: num_samples,
            _placeholder: Vec::new(),
        })
    }
}

impl<T: Float + num_traits::FromPrimitive + 'static> LegacyDataset<T> for MemoryMappedDataset<T> {
    fn len(&self) -> usize {
        self._num_samples
    }

    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        if index < self.len() {
            // Placeholder implementation
            // In real implementation, this would read from memory-mapped file
            let features = vec![T::zero(); self._sample_size];
            let target = vec![T::zero()];
            Some((
                Tensor::from_vec(features, vec![self._sample_size]),
                Tensor::from_vec(target, vec![1]),
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_dataset() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "feature1,feature2,target").unwrap();
        writeln!(temp_file, "1.0,2.0,0").unwrap();
        writeln!(temp_file, "3.0,4.0,1").unwrap();
        writeln!(temp_file, "5.0,6.0,0").unwrap();
        temp_file.flush().unwrap();

        let dataset = CSVDataset::<f32>::from_file(
            temp_file.path(),
            &[0, 1], // feature columns
            &[2],    // target column
            true,    // has header
            ',',     // delimiter
        )
        .unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.feature_names(), &["feature1", "feature2"]);
        assert_eq!(dataset.target_names(), &["target"]);

        // Test first sample
        if let Some((features, targets)) = dataset.get(0) {
            assert_eq!(features.as_slice().unwrap(), &[1.0, 2.0]);
            assert_eq!(targets.as_slice().unwrap(), &[0.0]);
        }

        // Test train/test split
        let (train_set, test_set) = dataset.train_test_split(0.67);
        assert_eq!(train_set.len(), 2);
        assert_eq!(test_set.len(), 1);
    }

    #[test]
    fn test_text_dataset() {
        let texts = vec![
            "hello world".to_string(),
            "machine learning is fun".to_string(),
            "natural language processing".to_string(),
        ];
        let labels = vec![0, 1, 1];

        let dataset = TextDataset::<f32>::new(texts, labels, 10).unwrap();

        assert_eq!(dataset.len(), 3);
        assert!(dataset.vocab().len() > 2); // Should have at least PAD, UNK + some words

        // Test first sample
        if let Some((text_tensor, label_tensor)) = dataset.get(0) {
            assert_eq!(text_tensor.shape(), &[10]); // max_length
            assert_eq!(label_tensor.as_slice().unwrap(), &[0.0]);
        }
    }

    #[test]
    fn test_image_dataset() {
        let image_paths = vec![PathBuf::from("test1.jpg"), PathBuf::from("test2.jpg")];
        let labels = vec![0, 1];
        let class_names = vec!["cat".to_string(), "dog".to_string()];

        let dataset = ImageDataset::<f32>::new(
            image_paths,
            labels,
            class_names.clone(),
            (224, 224), // image size
            3,          // channels (RGB)
        )
        .unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.class_names(), &class_names);
    }

    #[test]
    fn test_memory_mapped_dataset() {
        // Create a temporary binary file
        let mut temp_file = NamedTempFile::new().unwrap();
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        temp_file.write_all(bytes).unwrap();
        temp_file.flush().unwrap();

        let dataset = MemoryMappedDataset::<f32>::new(temp_file.path(), 10).unwrap();

        assert_eq!(dataset.len(), 100); // 1000 elements / 10 per sample

        // Test sample access (placeholder implementation)
        if let Some((features, _)) = dataset.get(0) {
            assert_eq!(features.shape(), &[10]);
        }
    }
}
