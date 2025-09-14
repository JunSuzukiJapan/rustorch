//! Python bindings for data loading and processing
//! データローディングと処理のPythonバインディング

use crate::data::{DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset};
use crate::python::error::to_py_err;
use crate::python::tensor::PyTensor;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for TensorDataset
/// TensorDatasetのPythonラッパー
#[pyclass]
pub struct PyTensorDataset {
    pub(crate) dataset: TensorDataset<f32>,
}

#[pymethods]
impl PyTensorDataset {
    #[new]
    pub fn new(data: &PyTensor, targets: &PyTensor) -> PyResult<Self> {
        match TensorDataset::new(data.tensor.clone(), targets.tensor.clone()) {
            Ok(dataset) => Ok(PyTensorDataset { dataset }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Create dataset from multiple tensors
    /// 複数のテンソルからデータセットを作成
    #[staticmethod]
    pub fn from_tensors(py: Python, tensors: &pyo3::types::PyList) -> PyResult<PyTensorDataset> {
        if tensors.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least 2 tensors are required (data and targets)",
            ));
        }

        let data_item: &PyTensor = tensors.get_item(0)?.extract()?;
        let targets_item: &PyTensor = tensors.get_item(1)?.extract()?;
        let data = data_item.tensor.clone();
        let targets = targets_item.tensor.clone();

        match TensorDataset::new(data, targets) {
            Ok(dataset) => Ok(PyTensorDataset { dataset }),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Get dataset length
    /// データセット長を取得
    pub fn __len__(&self) -> usize {
        self.dataset.len()
    }

    /// Get item at index
    /// インデックスのアイテムを取得
    pub fn __getitem__(&self, index: usize) -> PyResult<Vec<PyTensor>> {
        match self.dataset.get_item(index) {
            Ok(tensors) => {
                let py_tensors: Vec<PyTensor> = tensors
                    .into_iter()
                    .map(|t| PyTensor { tensor: t })
                    .collect();
                Ok(py_tensors)
            }
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Get batch of items
    /// アイテムのバッチを取得
    pub fn get_batch(&self, indices: Vec<usize>) -> PyResult<Vec<Vec<PyTensor>>> {
        let mut batch = Vec::new();

        for index in indices {
            let items = self.__getitem__(index)?;
            batch.push(items);
        }

        Ok(batch)
    }

    /// Get data tensor
    /// データテンソルを取得
    pub fn data(&self) -> PyTensor {
        // Assume first tensor is data
        PyTensor {
            tensor: self.dataset.tensors[0].clone(),
        }
    }

    /// Get targets tensor  
    /// ターゲットテンソルを取得
    pub fn targets(&self) -> PyTensor {
        // Assume second tensor is targets
        PyTensor {
            tensor: self.dataset.tensors[1].clone(),
        }
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "TensorDataset(length={}, data_shape={:?})",
            self.__len__(),
            self.dataset.tensors[0].shape()
        )
    }
}

/// Python wrapper for DataLoader
/// DataLoaderのPythonラッパー  
#[pyclass]
pub struct PyDataLoader {
    // Simplified implementation for Python bindings
    pub(crate) dataset: TensorDataset<f32>,
    pub(crate) batch_size: usize,
    pub(crate) shuffle: bool,
    pub(crate) current_batch: usize,
    pub(crate) indices: Vec<usize>,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    pub fn new(
        dataset: &PyTensorDataset,
        batch_size: Option<usize>,
        shuffle: Option<bool>,
        drop_last: Option<bool>,
        num_workers: Option<usize>,
    ) -> PyResult<Self> {
        let batch_size = batch_size.unwrap_or(1);
        let shuffle = shuffle.unwrap_or(false);
        let _drop_last = drop_last.unwrap_or(false);
        let _num_workers = num_workers.unwrap_or(0);

        let dataset_len = dataset.dataset.len();
        let mut indices: Vec<usize> = (0..dataset_len).collect();

        if shuffle {
            // Simple shuffle - in real implementation would use proper random seed
            indices.reverse();
        }

        Ok(PyDataLoader {
            dataset: dataset.dataset.clone(),
            batch_size,
            shuffle,
            current_batch: 0,
            indices,
        })
    }

    /// Get next batch
    /// 次のバッチを取得
    pub fn next_batch(&mut self) -> Option<Vec<PyTensor>> {
        let start_idx = self.current_batch * self.batch_size;
        let end_idx = std::cmp::min(start_idx + self.batch_size, self.indices.len());

        if start_idx >= self.indices.len() {
            return None;
        }

        let batch_indices = &self.indices[start_idx..end_idx];
        let mut batch = Vec::new();

        for &idx in batch_indices {
            if let Ok(tensors) = self.dataset.get_item(idx) {
                batch.extend(tensors.into_iter().map(|t| PyTensor { tensor: t }));
            }
        }

        self.current_batch += 1;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }

    /// Reset dataloader to beginning
    /// データローダーを開始位置にリセット
    pub fn reset(&mut self) {
        self.current_batch = 0;
        if self.shuffle {
            self.indices.reverse(); // Simple reshuffle
        }
    }

    /// Get batch size
    /// バッチサイズを取得
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if dataloader has more batches
    /// データローダーにさらにバッチがあるかチェック
    pub fn has_next(&self) -> bool {
        self.current_batch * self.batch_size < self.indices.len()
    }

    /// Get number of batches
    /// バッチ数を取得
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get current batch index
    /// 現在のバッチインデックスを取得
    pub fn current_batch_index(&self) -> usize {
        self.current_batch
    }

    /// Iterate over batches (Python iterator protocol)
    /// バッチを反復（Pythonイテレータープロトコル）
    pub fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Get next item for Python iterator
    /// Pythonイテレーター用次のアイテムを取得
    pub fn __next__(&mut self) -> Option<Vec<PyTensor>> {
        self.next_batch()
    }

    /// Get length (number of batches)
    /// 長さ（バッチ数）を取得
    pub fn __len__(&self) -> usize {
        self.num_batches()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!(
            "DataLoader(batch_size={}, num_batches={}, current_batch={})",
            self.batch_size(),
            self.num_batches(),
            self.current_batch
        )
    }
}

/// Python wrapper for data transforms
/// データ変換のPythonラッパー
#[pyclass]
#[derive(Clone)]
pub struct PyTransform {
    pub(crate) name: String,
    pub(crate) params: HashMap<String, f32>,
}

#[pymethods]
impl PyTransform {
    #[new]
    pub fn new(name: String, params: Option<HashMap<String, f32>>) -> Self {
        PyTransform {
            name,
            params: params.unwrap_or_default(),
        }
    }

    /// Apply transform to tensor
    /// テンソルに変換を適用
    pub fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = match self.name.as_str() {
            "normalize" => {
                let mean = self.params.get("mean").unwrap_or(&0.0);
                let std = self.params.get("std").unwrap_or(&1.0);
                self.normalize(&input.tensor, *mean, *std)?
            }
            "resize" => {
                let height = self.params.get("height").unwrap_or(&224.0) as &f32;
                let width = self.params.get("width").unwrap_or(&224.0) as &f32;
                self.resize(&input.tensor, *height as usize, *width as usize)?
            }
            "random_crop" => {
                let size = self.params.get("size").unwrap_or(&224.0);
                self.random_crop(&input.tensor, *size as usize)?
            }
            "horizontal_flip" => {
                let p = self.params.get("p").unwrap_or(&0.5);
                self.random_horizontal_flip(&input.tensor, *p)?
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown transform: {}",
                    self.name
                )));
            }
        };

        Ok(PyTensor { tensor: output })
    }

    /// Get transform name
    /// 変換名を取得
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get transform parameters
    /// 変換パラメータを取得
    pub fn params(&self) -> HashMap<String, f32> {
        self.params.clone()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        format!("Transform(name='{}', params={:?})", self.name, self.params)
    }
}

impl PyTransform {
    /// Normalize tensor
    /// テンソルを正規化
    fn normalize(
        &self,
        tensor: &crate::tensor::Tensor<f32>,
        mean: f32,
        std: f32,
    ) -> PyResult<crate::tensor::Tensor<f32>> {
        // Simple normalization: (x - mean) / std
        let data: Vec<f32> = tensor.iter().map(|&x| (x - mean) / std).collect();

        match crate::tensor::Tensor::from_vec(data, tensor.shape().to_vec()) {
            Ok(normalized) => Ok(normalized),
            Err(e) => Err(to_py_err(e)),
        }
    }

    /// Resize tensor (simplified implementation)
    /// テンソルサイズ変更（簡略実装）
    fn resize(
        &self,
        tensor: &crate::tensor::Tensor<f32>,
        height: usize,
        width: usize,
    ) -> PyResult<crate::tensor::Tensor<f32>> {
        // Simplified resize - just reshape if dimensions allow
        let total_elements = height * width;
        let current_elements: usize = tensor.shape().iter().product();

        if total_elements == current_elements {
            match tensor.reshape(&[height, width]) {
                Ok(resized) => Ok(resized),
                Err(e) => Err(to_py_err(e)),
            }
        } else {
            // For actual implementation, would need proper interpolation
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Resize with interpolation not implemented",
            ))
        }
    }

    /// Random crop (simplified implementation)
    /// ランダムクロップ（簡略実装）
    fn random_crop(
        &self,
        tensor: &crate::tensor::Tensor<f32>,
        size: usize,
    ) -> PyResult<crate::tensor::Tensor<f32>> {
        // Simplified - return original tensor for now
        // Actual implementation would extract random crop
        Ok(tensor.clone())
    }

    /// Random horizontal flip
    /// ランダム水平フリップ
    fn random_horizontal_flip(
        &self,
        tensor: &crate::tensor::Tensor<f32>,
        p: f32,
    ) -> PyResult<crate::tensor::Tensor<f32>> {
        // Simplified - flip with probability p
        let should_flip = rand::random::<f32>() < p;

        if should_flip {
            // Simplified flip - reverse order of elements
            let mut data: Vec<f32> = tensor.iter().cloned().collect();
            data.reverse();
            crate::tensor::Tensor::from_vec(data, tensor.shape().to_vec())
                .map_err(to_py_err)
        } else {
            Ok(tensor.clone())
        }
    }
}

/// Collection of transforms
/// 変換のコレクション
#[pyclass]
pub struct PyTransforms {
    pub(crate) transforms: Vec<PyTransform>,
}

#[pymethods]
impl PyTransforms {
    #[new]
    pub fn new(transforms: Vec<PyTransform>) -> PyResult<Self> {
        Ok(PyTransforms {
            transforms,
        })
    }

    /// Apply all transforms sequentially
    /// 全ての変換を順次適用
    pub fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let mut current = input.clone();

        for transform in &self.transforms {
            current = transform.__call__(&current)?;
        }

        Ok(current)
    }

    /// Add transform
    /// 変換を追加
    pub fn append(&mut self, transform: &PyTransform) {
        self.transforms.push(transform.clone());
    }

    /// Get number of transforms
    /// 変換数を取得
    pub fn __len__(&self) -> usize {
        self.transforms.len()
    }

    /// String representation
    /// 文字列表現
    pub fn __repr__(&self) -> String {
        let names: Vec<&str> = self.transforms.iter().map(|t| t.name()).collect();
        format!("Transforms([{}])", names.join(", "))
    }
}

// Common transform factory functions
// 一般的な変換ファクトリ関数

/// Create normalization transform
/// 正規化変換を作成
#[pyfunction]
pub fn normalize(mean: f32, std: f32) -> PyTransform {
    let mut params = HashMap::new();
    params.insert("mean".to_string(), mean);
    params.insert("std".to_string(), std);
    PyTransform::new("normalize".to_string(), Some(params))
}

/// Create resize transform
/// リサイズ変換を作成
#[pyfunction]
pub fn resize(height: usize, width: usize) -> PyTransform {
    let mut params = HashMap::new();
    params.insert("height".to_string(), height as f32);
    params.insert("width".to_string(), width as f32);
    PyTransform::new("resize".to_string(), Some(params))
}

/// Create random horizontal flip transform
/// ランダム水平フリップ変換を作成
#[pyfunction]
pub fn random_horizontal_flip(p: Option<f32>) -> PyTransform {
    let mut params = HashMap::new();
    params.insert("p".to_string(), p.unwrap_or(0.5));
    PyTransform::new("horizontal_flip".to_string(), Some(params))
}

/// Create random crop transform
/// ランダムクロップ変換を作成
#[pyfunction]
pub fn random_crop(size: usize) -> PyTransform {
    let mut params = HashMap::new();
    params.insert("size".to_string(), size as f32);
    PyTransform::new("random_crop".to_string(), Some(params))
}
