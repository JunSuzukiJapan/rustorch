//! Computer Vision utilities for RusTorch
//! RusTorch用コンピュータビジョンユーティリティ
//!
//! This module provides computer vision functionality similar to torchvision,
//! including image transformations, data augmentation, and built-in datasets.
//!
//! このモジュールはtorchvisionと同様のコンピュータビジョン機能を提供し、
//! 画像変換、データ拡張、組み込みデータセットを含みます。

pub mod transforms;
pub mod datasets;
pub mod utils;
pub mod pipeline;
pub mod presets;

pub use transforms::*;
pub use datasets::*;
pub use pipeline::*;

use crate::tensor::Tensor;
use num_traits::Float;

/// Common image format representation
/// 共通画像形式表現
#[derive(Debug, Clone)]
pub struct Image<T: Float> {
    /// Image data tensor with shape (C, H, W) or (H, W, C)
    /// 画像データテンソル (C, H, W) または (H, W, C) の形状
    pub data: Tensor<T>,
    /// Image height in pixels
    /// 画像の高さ（ピクセル）
    pub height: usize,
    /// Image width in pixels
    /// 画像の幅（ピクセル）
    pub width: usize,
    /// Number of channels (e.g., 1 for grayscale, 3 for RGB)
    /// チャンネル数 (グレースケール=1, RGB=3 など)
    pub channels: usize,
    /// Data format: CHW (channels first) or HWC (channels last)
    /// データ形式: CHW (チャンネル最初) または HWC (チャンネル最後)
    pub format: ImageFormat,
}

/// Image data format
/// 画像データ形式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// Channels first: (C, H, W)
    /// チャンネル最初: (C, H, W)
    CHW,
    /// Channels last: (H, W, C) 
    /// チャンネル最後: (H, W, C)
    HWC,
}

impl<T: Float + 'static> Image<T> {
    /// Create a new image from tensor data
    /// テンソルデータから新しい画像を作成
    pub fn new(data: Tensor<T>, format: ImageFormat) -> Result<Self, VisionError> {
        let shape = data.shape();
        
        let (height, width, channels) = match (format, shape.len()) {
            (ImageFormat::CHW, 3) => (shape[1], shape[2], shape[0]),
            (ImageFormat::HWC, 3) => (shape[0], shape[1], shape[2]),
            (ImageFormat::CHW, 4) => (shape[2], shape[3], shape[1]), // Batch dimension included
            (ImageFormat::HWC, 4) => (shape[1], shape[2], shape[3]), // Batch dimension included
            _ => return Err(VisionError::InvalidImageShape(format!("Expected 3D or 4D tensor, got {:?}", shape)))
        };
        
        Ok(Image {
            data,
            height,
            width, 
            channels,
            format,
        })
    }
    
    /// Convert image format (CHW <-> HWC)
    /// 画像形式を変換 (CHW <-> HWC)
    pub fn to_format(&self, target_format: ImageFormat) -> Result<Image<T>, VisionError> {
        if self.format == target_format {
            return Ok(self.clone());
        }
        
        // For now, return a simple clone - actual implementation would permute dimensions
        // 現在は簡単なクローンを返す - 実際の実装では次元を入れ替える
        let mut new_image = self.clone();
        new_image.format = target_format;
        Ok(new_image)
    }
    
    /// Get image size as (height, width)
    /// 画像サイズを (高さ, 幅) として取得
    pub fn size(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

/// Vision-related errors
/// ビジョン関連エラー
#[derive(Debug, Clone)]
pub enum VisionError {
    /// Invalid image shape
    /// 無効な画像形状
    InvalidImageShape(String),
    /// Invalid transform parameters
    /// 無効な変換パラメータ
    InvalidTransformParams(String),
    /// Dataset loading error
    /// データセット読み込みエラー
    DatasetError(String),
    /// I/O error
    /// I/Oエラー
    IoError(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisionError::InvalidImageShape(msg) => write!(f, "Invalid image shape: {}", msg),
            VisionError::InvalidTransformParams(msg) => write!(f, "Invalid transform parameters: {}", msg),
            VisionError::DatasetError(msg) => write!(f, "Dataset error: {}", msg),
            VisionError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for VisionError {}

/// Result type for vision operations
/// ビジョン操作の結果型
pub type VisionResult<T> = Result<T, VisionError>;