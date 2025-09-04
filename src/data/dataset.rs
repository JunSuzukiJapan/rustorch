//! Dataset traits and implementations for Phase 5
//! フェーズ5用データセットトレイトと実装

use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::HashMap;

/// Data loading error type
/// データ読み込みエラータイプ
pub type DataError = RusTorchError;

/// Core Dataset trait following PyTorch API (Phase 5 - Main API)
/// PyTorch APIに従うコアDatasetトレイト（フェーズ5 - メインAPI）
pub trait Dataset<T> {
    /// Returns the number of samples in the dataset
    /// データセット内のサンプル数を返す
    fn len(&self) -> usize;

    /// Returns true if dataset is empty
    /// データセットが空の場合trueを返す
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get item at index
    /// インデックスでアイテムを取得
    fn get_item(&self, index: usize) -> Result<T, DataError>;
}

/// Iterable Dataset trait for streaming data
/// ストリーミングデータ用反復可能データセットトレイト
pub trait IterableDataset<T> {
    type Iterator: Iterator<Item = Result<T, DataError>>;

    /// Create iterator over dataset
    /// データセット上のイテレータを作成
    fn iter(&self) -> Self::Iterator;
}

/// Simple tensor dataset implementation
/// シンプルなテンソルデータセット実装
#[derive(Debug, Clone)]
pub struct TensorDataset<T: Float> {
    tensors: Vec<Tensor<T>>,
}

impl<T: Float + 'static> TensorDataset<T> {
    /// Create new tensor dataset
    /// 新しいテンソルデータセットを作成
    pub fn new(tensors: Vec<Tensor<T>>) -> Result<Self, DataError> {
        if tensors.is_empty() {
            return Err(RusTorchError::InvalidParameters {
                operation: "TensorDataset::new".to_string(),
                message: "Tensors cannot be empty".to_string(),
            });
        }

        // For new() method, all tensors should represent batches with consistent batch size
        let first_dim = tensors[0].shape()[0];
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape()[0] != first_dim {
                return Err(RusTorchError::ShapeMismatch {
                    expected: vec![first_dim],
                    actual: vec![tensor.shape()[0]],
                });
            }
        }

        Ok(Self { tensors })
    }

    /// Get reference to tensors
    /// テンソルの参照を取得
    pub fn tensors(&self) -> &[Tensor<T>] {
        &self.tensors
    }

    /// Add tensor to dataset
    /// データセットにテンソルを追加
    pub fn add_tensor(&mut self, tensor: Tensor<T>) -> Result<(), DataError> {
        if !self.tensors.is_empty() && tensor.shape()[0] != self.tensors[0].shape()[0] {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self.tensors[0].shape()[0]],
                actual: vec![tensor.shape()[0]],
            });
        }
        self.tensors.push(tensor);
        Ok(())
    }

    /// Create tensor dataset from features and targets
    /// 特徴量とターゲットからテンソルデータセットを作成
    pub fn from_features_targets(
        features: Vec<Tensor<T>>,
        targets: Vec<Tensor<T>>,
    ) -> Result<Self, DataError> {
        if features.len() != targets.len() {
            return Err(RusTorchError::InvalidParameters {
                operation: "TensorDataset::from_features_targets".to_string(),
                message: "Features and targets must have the same length".to_string(),
            });
        }

        // Convert individual samples to batch tensors
        let mut batch_tensors = Vec::new();

        if !features.is_empty() {
            // Combine features into a single batch tensor
            let sample_shape = &features[0].shape()[..];
            let batch_size = features.len();
            let mut combined_shape = vec![batch_size];
            combined_shape.extend_from_slice(sample_shape);

            let total_elements: usize = combined_shape.iter().product();
            let mut combined_data = Vec::with_capacity(total_elements);

            for feature in features {
                if let Some(data) = feature.data.as_slice() {
                    combined_data.extend_from_slice(data);
                }
            }

            let features_batch = Tensor::from_vec(combined_data, combined_shape);
            batch_tensors.push(features_batch);
        }

        if !targets.is_empty() {
            // Combine targets into a single batch tensor
            let sample_shape = &targets[0].shape()[..];
            let batch_size = targets.len();
            let mut combined_shape = vec![batch_size];
            combined_shape.extend_from_slice(sample_shape);

            let total_elements: usize = combined_shape.iter().product();
            let mut combined_data = Vec::with_capacity(total_elements);

            for target in targets {
                if let Some(data) = target.data.as_slice() {
                    combined_data.extend_from_slice(data);
                }
            }

            let targets_batch = Tensor::from_vec(combined_data, combined_shape);
            batch_tensors.push(targets_batch);
        }

        Self::new(batch_tensors)
    }
}

impl<T: Float + Clone + 'static> Dataset<Vec<Tensor<T>>> for TensorDataset<T> {
    fn len(&self) -> usize {
        if self.tensors.is_empty() {
            0
        } else {
            self.tensors[0].shape()[0]
        }
    }

    fn get_item(&self, index: usize) -> Result<Vec<Tensor<T>>, DataError> {
        if index >= self.len() {
            return Err(RusTorchError::InvalidParameters {
                operation: "TensorDataset::get_item".to_string(),
                message: format!(
                    "Index {} out of bounds for dataset of size {}",
                    index,
                    self.len()
                ),
            });
        }

        let mut result = Vec::new();
        for tensor in &self.tensors {
            // Extract slice at index
            let shape = tensor.shape();

            if shape.is_empty() {
                return Err(RusTorchError::InvalidParameters {
                    operation: "TensorDataset::get_item".to_string(),
                    message: "Cannot index into scalar tensor".to_string(),
                });
            }

            // For simplicity, assume the tensor can be sliced along first dimension
            let slice_shape = if shape.len() == 1 {
                vec![1]
            } else {
                shape[1..].to_vec()
            };

            let total_elements: usize = slice_shape.iter().product();
            let start_idx = index * total_elements;
            let end_idx = start_idx + total_elements;

            let slice_data = tensor.data.as_slice().unwrap()[start_idx..end_idx].to_vec();
            let slice_tensor = Tensor::from_vec(slice_data, slice_shape);
            result.push(slice_tensor);
        }

        Ok(result)
    }
}


/// Concatenated dataset implementation
/// 連結データセット実装
pub struct ConcatDataset<T> {
    datasets: Vec<Box<dyn Dataset<T>>>,
    cumulative_sizes: Vec<usize>,
}

impl<T> ConcatDataset<T> {
    /// Create new concatenated dataset
    /// 新しい連結データセットを作成
    pub fn new(datasets: Vec<Box<dyn Dataset<T>>>) -> Result<Self, DataError> {
        if datasets.is_empty() {
            return Err(RusTorchError::InvalidParameters {
                operation: "ConcatDataset::new".to_string(),
                message: "Datasets cannot be empty".to_string(),
            });
        }

        let mut cumulative_sizes = Vec::new();
        let mut total_size = 0;

        for dataset in &datasets {
            total_size += dataset.len();
            cumulative_sizes.push(total_size);
        }

        Ok(Self {
            datasets,
            cumulative_sizes,
        })
    }

    /// Find which dataset contains the given index
    /// 指定されたインデックスを含むデータセットを見つける
    fn find_dataset(&self, index: usize) -> Result<(usize, usize), DataError> {
        if index >= self.len() {
            return Err(RusTorchError::InvalidParameters {
                operation: "ConcatDataset::find_dataset".to_string(),
                message: format!("Index {} out of bounds", index),
            });
        }

        for (dataset_idx, &cumulative_size) in self.cumulative_sizes.iter().enumerate() {
            if index < cumulative_size {
                let local_index = if dataset_idx == 0 {
                    index
                } else {
                    index - self.cumulative_sizes[dataset_idx - 1]
                };
                return Ok((dataset_idx, local_index));
            }
        }

        unreachable!()
    }
}

impl<T> Dataset<T> for ConcatDataset<T> {
    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get_item(&self, index: usize) -> Result<T, DataError> {
        let (dataset_idx, local_index) = self.find_dataset(index)?;
        self.datasets[dataset_idx].get_item(local_index)
    }
}
