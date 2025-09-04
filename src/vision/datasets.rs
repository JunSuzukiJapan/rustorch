//! Built-in datasets for computer vision
//! コンピュータビジョン用組み込みデータセット
//!
//! This module provides popular computer vision datasets similar to torchvision.datasets,
//! including MNIST, CIFAR-10, CIFAR-100, and utilities for custom datasets.
//!
//! このモジュールはtorchvision.datasetsと同様の人気のコンピュータビジョンデータセットを提供し、
//! MNIST、CIFAR-10、CIFAR-100、カスタムデータセット用ユーティリティを含みます。

use crate::data::Dataset;
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::vision::transforms::Transform;
use crate::vision::{Image, ImageFormat};
use num_traits::Float;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// MNIST dataset
/// MNISTデータセット
#[derive(Debug)]
pub struct MNIST<T: Float> {
    /// Dataset root directory
    /// データセットルートディレクトリ
    pub root: PathBuf,
    /// Training or test split
    /// 訓練または検証分割
    pub train: bool,
    /// List of image data
    /// 画像データのリスト
    images: Vec<Tensor<T>>,
    /// List of labels
    /// ラベルのリスト
    labels: Vec<i64>,
    /// Optional transforms to apply
    /// 適用するオプション変換
    transform: Option<Box<dyn Transform<T>>>,
    /// Whether to download dataset if not found
    /// 見つからない場合にデータセットをダウンロードするかどうか
    download: bool,
}

impl<T: Float + From<f32> + From<u8> + Copy + 'static> MNIST<T> {
    /// Create new MNIST dataset
    /// 新しいMNISTデータセットを作成
    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> RusTorchResult<Self> {
        let root = root.as_ref().to_path_buf();

        let mut dataset = MNIST {
            root,
            train,
            images: Vec::new(),
            labels: Vec::new(),
            transform: None,
            download,
        };

        dataset.load_data()?;
        Ok(dataset)
    }

    /// Set transform
    /// 変換を設定
    pub fn with_transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Load MNIST data from files
    /// ファイルからMNISTデータを読み込み
    fn load_data(&mut self) -> RusTorchResult<()> {
        // Check if data files exist
        // データファイルの存在を確認
        let data_dir = self.root.join("MNIST").join("raw");

        if !data_dir.exists() {
            if self.download {
                self.download_data()?;
            } else {
                return Err(RusTorchError::DatasetError(format!(
                    "MNIST data not found at {:?}. Set download=true to download.",
                    data_dir
                )));
            }
        }

        // For now, create dummy data
        // 現在はダミーデータを作成
        let num_samples = if self.train { 60000 } else { 10000 };

        for i in 0..num_samples {
            // Create dummy 28x28 image
            // ダミーの28x28画像を作成
            let image_data: Vec<T> = (0..784).map(|_| <T as From<f32>>::from(0.5f32)).collect();
            let image_tensor = Tensor::from_vec(image_data, vec![1, 28, 28]);
            self.images.push(image_tensor);

            // Create dummy label (0-9)
            // ダミーラベル（0-9）を作成
            self.labels.push((i % 10) as i64);
        }

        Ok(())
    }

    /// Download MNIST data
    /// MNISTデータをダウンロード
    fn download_data(&self) -> RusTorchResult<()> {
        // Create directory structure
        // ディレクトリ構造を作成
        let data_dir = self.root.join("MNIST").join("raw");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| RusTorchError::IoError(format!("Failed to create directory: {}", e)))?;

        // In a real implementation, this would download from the official MNIST URLs
        // 実際の実装では、公式のMNIST URLからダウンロード
        println!("Note: MNIST download not implemented - using dummy data");

        Ok(())
    }

    /// Get number of classes
    /// クラス数を取得
    pub fn num_classes(&self) -> usize {
        10
    }
}

// Phase 5 Dataset implementation
impl<T: Float + From<f32> + From<u8> + Copy + 'static> Dataset<(Tensor<T>, Tensor<T>)> for MNIST<T> {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), crate::data::DataError> {
        if index >= self.images.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "MNIST::get_item".to_string(),
                message: format!("Index {} out of bounds for dataset of size {}", index, self.images.len()),
            });
        }

        let mut image = self.images[index].clone();
        let label = Tensor::from_vec(
            vec![<T as From<u8>>::from(self.labels[index] as u8)],
            vec![1],
        );

        // Apply transforms if available
        // 変換が利用可能な場合は適用
        if let Some(ref transform) = self.transform {
            if let Ok(img) = Image::new(image.clone(), ImageFormat::CHW) {
                if let Ok(transformed_img) = transform.apply(&img) {
                    image = transformed_img.data;
                }
            }
        }

        Ok((image, label))
    }
}


/// CIFAR-10 dataset
/// CIFAR-10データセット
#[derive(Debug)]
pub struct CIFAR10<T: Float> {
    /// Dataset root directory
    /// データセットルートディレクトリ
    pub root: PathBuf,
    /// Training or test split
    /// 訓練または検証分割
    pub train: bool,
    /// List of image data
    /// 画像データのリスト
    images: Vec<Tensor<T>>,
    /// List of labels
    /// ラベルのリスト
    labels: Vec<i64>,
    /// Optional transforms to apply
    /// 適用するオプション変換
    transform: Option<Box<dyn Transform<T>>>,
    /// Whether to download dataset if not found
    /// 見つからない場合にデータセットをダウンロードするかどうか
    download: bool,
    /// Class names
    /// クラス名
    classes: Vec<String>,
}

impl<T: Float + From<f32> + From<u8> + Copy + 'static> CIFAR10<T> {
    /// Create new CIFAR-10 dataset
    /// 新しいCIFAR-10データセットを作成
    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> RusTorchResult<Self> {
        let root = root.as_ref().to_path_buf();

        let classes = vec![
            "airplane".to_string(),
            "automobile".to_string(),
            "bird".to_string(),
            "cat".to_string(),
            "deer".to_string(),
            "dog".to_string(),
            "frog".to_string(),
            "horse".to_string(),
            "ship".to_string(),
            "truck".to_string(),
        ];

        let mut dataset = CIFAR10 {
            root,
            train,
            images: Vec::new(),
            labels: Vec::new(),
            transform: None,
            download,
            classes,
        };

        dataset.load_data()?;
        Ok(dataset)
    }

    /// Set transform
    /// 変換を設定
    pub fn with_transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Load CIFAR-10 data from files
    /// ファイルからCIFAR-10データを読み込み
    fn load_data(&mut self) -> RusTorchResult<()> {
        let data_dir = self.root.join("cifar-10-batches-py");

        if !data_dir.exists() {
            if self.download {
                self.download_data()?;
            } else {
                return Err(RusTorchError::DatasetError(format!(
                    "CIFAR-10 data not found at {:?}. Set download=true to download.",
                    data_dir
                )));
            }
        }

        // For now, create dummy data
        // 現在はダミーデータを作成
        let num_samples = if self.train { 50000 } else { 10000 };

        for i in 0..num_samples {
            // Create dummy 32x32x3 RGB image
            // ダミーの32x32x3 RGB画像を作成
            let image_data: Vec<T> = (0..3072).map(|_| <T as From<f32>>::from(0.5f32)).collect();
            let image_tensor = Tensor::from_vec(image_data, vec![3, 32, 32]);
            self.images.push(image_tensor);

            // Create dummy label (0-9)
            // ダミーラベル（0-9）を作成
            self.labels.push((i % 10) as i64);
        }

        Ok(())
    }

    /// Download CIFAR-10 data
    /// CIFAR-10データをダウンロード
    fn download_data(&self) -> RusTorchResult<()> {
        let data_dir = self.root.join("cifar-10-batches-py");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| RusTorchError::IoError(format!("Failed to create directory: {}", e)))?;

        println!("Note: CIFAR-10 download not implemented - using dummy data");

        Ok(())
    }

    /// Get number of classes
    /// クラス数を取得
    pub fn num_classes(&self) -> usize {
        10
    }

    /// Get class names
    /// クラス名を取得
    pub fn class_names(&self) -> &[String] {
        &self.classes
    }
}

// Phase 5 Dataset implementation
impl<T: Float + From<f32> + From<u8> + Copy + 'static> Dataset<(Tensor<T>, Tensor<T>)> for CIFAR10<T> {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), crate::data::DataError> {
        if index >= self.images.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "CIFAR10::get_item".to_string(),
                message: format!("Index {} out of bounds for dataset of size {}", index, self.images.len()),
            });
        }

        let mut image = self.images[index].clone();
        let label = Tensor::from_vec(
            vec![<T as From<u8>>::from(self.labels[index] as u8)],
            vec![1],
        );

        // Apply transforms if available
        // 変換が利用可能な場合は適用
        if let Some(ref transform) = self.transform {
            if let Ok(img) = Image::new(image.clone(), ImageFormat::CHW) {
                if let Ok(transformed_img) = transform.apply(&img) {
                    image = transformed_img.data;
                }
            }
        }

        Ok((image, label))
    }
}


/// CIFAR-100 dataset
/// CIFAR-100データセット
#[derive(Debug)]
pub struct CIFAR100<T: Float> {
    /// Dataset root directory
    /// データセットルートディレクトリ
    pub root: PathBuf,
    /// Training or test split
    /// 訓練または検証分割
    pub train: bool,
    /// List of image data
    /// 画像データのリスト
    images: Vec<Tensor<T>>,
    /// List of fine labels
    /// 細かいラベルのリスト
    fine_labels: Vec<i64>,
    /// List of coarse labels
    /// 粗いラベルのリスト
    coarse_labels: Vec<i64>,
    /// Optional transforms to apply
    /// 適用するオプション変換
    transform: Option<Box<dyn Transform<T>>>,
    /// Whether to download dataset if not found
    /// 見つからない場合にデータセットをダウンロードするかどうか
    download: bool,
    /// Fine class names
    /// 細かいクラス名
    fine_classes: Vec<String>,
    /// Coarse class names
    /// 粗いクラス名
    coarse_classes: Vec<String>,
}

impl<T: Float + From<f32> + From<u8> + Copy + 'static> CIFAR100<T> {
    /// Create new CIFAR-100 dataset
    /// 新しいCIFAR-100データセットを作成
    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> RusTorchResult<Self> {
        let root = root.as_ref().to_path_buf();

        // Simplified class names - in real implementation would load from meta file
        // 簡略化クラス名 - 実際の実装ではメタファイルから読み込み
        let fine_classes: Vec<String> = (0..100).map(|i| format!("class_{}", i)).collect();
        let coarse_classes: Vec<String> = (0..20).map(|i| format!("superclass_{}", i)).collect();

        let mut dataset = CIFAR100 {
            root,
            train,
            images: Vec::new(),
            fine_labels: Vec::new(),
            coarse_labels: Vec::new(),
            transform: None,
            download,
            fine_classes,
            coarse_classes,
        };

        dataset.load_data()?;
        Ok(dataset)
    }

    /// Set transform
    /// 変換を設定
    pub fn with_transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Load CIFAR-100 data from files
    /// ファイルからCIFAR-100データを読み込み
    fn load_data(&mut self) -> RusTorchResult<()> {
        let data_dir = self.root.join("cifar-100-python");

        if !data_dir.exists() {
            if self.download {
                self.download_data()?;
            } else {
                return Err(RusTorchError::DatasetError(format!(
                    "CIFAR-100 data not found at {:?}. Set download=true to download.",
                    data_dir
                )));
            }
        }

        // For now, create dummy data
        // 現在はダミーデータを作成
        let num_samples = if self.train { 50000 } else { 10000 };

        for i in 0..num_samples {
            // Create dummy 32x32x3 RGB image
            // ダミーの32x32x3 RGB画像を作成
            let image_data: Vec<T> = (0..3072).map(|_| <T as From<f32>>::from(0.5f32)).collect();
            let image_tensor = Tensor::from_vec(image_data, vec![3, 32, 32]);
            self.images.push(image_tensor);

            // Create dummy fine and coarse labels
            // ダミーの細かい・粗いラベルを作成
            self.fine_labels.push((i % 100) as i64);
            self.coarse_labels.push((i % 20) as i64);
        }

        Ok(())
    }

    /// Download CIFAR-100 data
    /// CIFAR-100データをダウンロード
    fn download_data(&self) -> RusTorchResult<()> {
        let data_dir = self.root.join("cifar-100-python");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| RusTorchError::IoError(format!("Failed to create directory: {}", e)))?;

        println!("Note: CIFAR-100 download not implemented - using dummy data");

        Ok(())
    }

    /// Get number of fine classes
    /// 細かいクラス数を取得
    pub fn num_fine_classes(&self) -> usize {
        100
    }

    /// Get number of coarse classes
    /// 粗いクラス数を取得
    pub fn num_coarse_classes(&self) -> usize {
        20
    }

    /// Get fine class names
    /// 細かいクラス名を取得
    pub fn fine_class_names(&self) -> &[String] {
        &self.fine_classes
    }

    /// Get coarse class names
    /// 粗いクラス名を取得
    pub fn coarse_class_names(&self) -> &[String] {
        &self.coarse_classes
    }
}

// Phase 5 Dataset implementation
impl<T: Float + From<f32> + From<u8> + Copy + 'static> Dataset<(Tensor<T>, Tensor<T>)> for CIFAR100<T> {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), crate::data::DataError> {
        if index >= self.images.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "CIFAR100::get_item".to_string(),
                message: format!("Index {} out of bounds for dataset of size {}", index, self.images.len()),
            });
        }

        let mut image = self.images[index].clone();
        // Return fine labels by default
        // デフォルトで細かいラベルを返す
        let label = Tensor::from_vec(
            vec![<T as From<u8>>::from(self.fine_labels[index] as u8)],
            vec![1],
        );

        // Apply transforms if available
        // 変換が利用可能な場合は適用
        if let Some(ref transform) = self.transform {
            if let Ok(img) = Image::new(image.clone(), ImageFormat::CHW) {
                if let Ok(transformed_img) = transform.apply(&img) {
                    image = transformed_img.data;
                }
            }
        }

        Ok((image, label))
    }
}


/// Custom image folder dataset
/// カスタム画像フォルダデータセット
#[derive(Debug)]
pub struct ImageFolder<T: Float> {
    /// Dataset root directory
    /// データセットルートディレクトリ
    pub root: PathBuf,
    /// Mapping from class names to indices
    /// クラス名からインデックスへのマッピング
    class_to_idx: HashMap<String, usize>,
    /// List of image paths and labels
    /// 画像パスとラベルのリスト
    samples: Vec<(PathBuf, usize)>,
    /// Optional transforms to apply
    /// 適用するオプション変換
    transform: Option<Box<dyn Transform<T>>>,
}

impl<T: Float + From<f32> + From<u8> + Copy + 'static> ImageFolder<T> {
    /// Create new image folder dataset
    /// 新しい画像フォルダデータセットを作成
    pub fn new<P: AsRef<Path>>(root: P) -> RusTorchResult<Self> {
        let root = root.as_ref().to_path_buf();

        if !root.exists() || !root.is_dir() {
            return Err(RusTorchError::DatasetError(format!(
                "Root directory {:?} does not exist or is not a directory",
                root
            )));
        }

        let mut dataset = ImageFolder {
            root,
            class_to_idx: HashMap::new(),
            samples: Vec::new(),
            transform: None,
        };

        dataset.scan_directory()?;
        Ok(dataset)
    }

    /// Set transform
    /// 変換を設定
    pub fn with_transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Scan directory for images
    /// 画像のディレクトリをスキャン
    fn scan_directory(&mut self) -> RusTorchResult<()> {
        let mut class_names = Vec::new();

        // Get class directories
        // クラスディレクトリを取得
        for entry in std::fs::read_dir(&self.root)
            .map_err(|e| RusTorchError::IoError(format!("Failed to read directory: {}", e)))?
        {
            let entry = entry.map_err(|e| RusTorchError::IoError(e.to_string()))?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(class_name) = path.file_name() {
                    if let Some(class_name_str) = class_name.to_str() {
                        class_names.push(class_name_str.to_string());
                    }
                }
            }
        }

        // Sort class names for consistent ordering
        // 一貫した順序のためにクラス名をソート
        class_names.sort();

        // Create class to index mapping
        // クラスからインデックスへのマッピングを作成
        for (idx, class_name) in class_names.iter().enumerate() {
            self.class_to_idx.insert(class_name.clone(), idx);
        }

        // Scan for image files in each class directory
        // 各クラスディレクトリで画像ファイルをスキャン
        for class_name in class_names {
            let class_dir = self.root.join(&class_name);
            let class_idx = self.class_to_idx[&class_name];

            for entry in std::fs::read_dir(&class_dir).map_err(|e| {
                RusTorchError::IoError(format!("Failed to read class directory: {}", e))
            })? {
                let entry = entry.map_err(|e| RusTorchError::IoError(e.to_string()))?;
                let path = entry.path();

                if path.is_file() {
                    if let Some(extension) = path.extension() {
                        if let Some(ext_str) = extension.to_str() {
                            // Check if file is an image
                            // ファイルが画像かどうかをチェック
                            if matches!(
                                ext_str.to_lowercase().as_str(),
                                "jpg" | "jpeg" | "png" | "bmp" | "tiff"
                            ) {
                                self.samples.push((path, class_idx));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get class names
    /// クラス名を取得
    pub fn classes(&self) -> Vec<String> {
        let mut classes: Vec<_> = self.class_to_idx.iter().collect();
        classes.sort_by_key(|(_, &idx)| idx);
        classes.into_iter().map(|(name, _)| name.clone()).collect()
    }

    /// Get number of classes
    /// クラス数を取得
    pub fn num_classes(&self) -> usize {
        self.class_to_idx.len()
    }
}

// Phase 5 Dataset implementation
impl<T: Float + From<f32> + From<u8> + Copy + 'static> Dataset<(Tensor<T>, Tensor<T>)> for ImageFolder<T> {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_item(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>), crate::data::DataError> {
        if index >= self.samples.len() {
            return Err(crate::error::RusTorchError::InvalidParameters {
                operation: "ImageFolder::get_item".to_string(),
                message: format!("Index {} out of bounds for dataset of size {}", index, self.samples.len()),
            });
        }

        let (_path, class_idx) = &self.samples[index];

        // For now, create dummy image data - real implementation would load image from file
        // 現在はダミー画像データを作成 - 実際の実装ではファイルから画像を読み込み
        let image_data: Vec<T> = (0..3072).map(|_| <T as From<f32>>::from(0.5f32)).collect();
        let mut image = Tensor::from_vec(image_data, vec![3, 32, 32]);
        let label = Tensor::from_vec(vec![<T as From<u8>>::from(*class_idx as u8)], vec![1]);

        // Apply transforms if available
        // 変換が利用可能な場合は適用
        if let Some(ref transform) = self.transform {
            if let Ok(img) = Image::new(image.clone(), ImageFormat::CHW) {
                if let Ok(transformed_img) = transform.apply(&img) {
                    image = transformed_img.data;
                }
            }
        }

        Ok((image, label))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_mnist_creation() {
        let temp_dir = env::temp_dir().join("test_mnist");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mnist = MNIST::<f32>::new(&temp_dir, true, true);
        assert!(mnist.is_ok());

        let mnist = mnist.unwrap();
        assert!(Dataset::len(&mnist) > 0);
        assert_eq!(mnist.num_classes(), 10);
    }

    #[test]
    fn test_cifar10_creation() {
        let temp_dir = env::temp_dir().join("test_cifar10");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let cifar10 = CIFAR10::<f32>::new(&temp_dir, true, true);
        assert!(cifar10.is_ok());

        let cifar10 = cifar10.unwrap();
        assert!(Dataset::len(&cifar10) > 0);
        assert_eq!(cifar10.num_classes(), 10);
        assert_eq!(cifar10.class_names().len(), 10);
    }

    #[test]
    fn test_cifar100_creation() {
        let temp_dir = env::temp_dir().join("test_cifar100");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let cifar100 = CIFAR100::<f32>::new(&temp_dir, true, true);
        assert!(cifar100.is_ok());

        let cifar100 = cifar100.unwrap();
        assert!(Dataset::len(&cifar100) > 0);
        assert_eq!(cifar100.num_fine_classes(), 100);
        assert_eq!(cifar100.num_coarse_classes(), 20);

        // Test new accessor methods
        // 新しいアクセサメソッドをテスト
        let fine_classes = cifar100.fine_class_names();
        let coarse_classes = cifar100.coarse_class_names();
        assert_eq!(fine_classes.len(), 100);
        assert_eq!(coarse_classes.len(), 20);
    }
}
