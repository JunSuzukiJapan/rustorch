//! Vision utilities and helper functions
//! ビジョンユーティリティとヘルパー関数

use crate::tensor::Tensor;
use crate::vision::{Image, ImageFormat, VisionError, VisionResult};
use num_traits::Float;

/// Make a grid of images for visualization
/// 可視化用の画像グリッドを作成
pub fn make_grid<T: Float + From<f32> + Copy + 'static>(
    images: &[Image<T>],
    nrow: usize,
    padding: usize,
    _normalize: bool,
    _value_range: Option<(T, T)>,
    _scale_each: bool,
    pad_value: T,
) -> VisionResult<Image<T>> {
    if images.is_empty() {
        return Err(VisionError::InvalidTransformParams(
            "Cannot make grid from empty image list".to_string()
        ));
    }
    
    let num_images = images.len();
    let ncol = if num_images % nrow == 0 { num_images / nrow } else { num_images / nrow + 1 };
    
    let (height, width) = images[0].size();
    let channels = images[0].channels;
    
    // Calculate grid dimensions
    // グリッド次元を計算
    let grid_height = ncol * height + (ncol + 1) * padding;
    let grid_width = nrow * width + (nrow + 1) * padding;
    
    // Create grid tensor
    // グリッドテンソルを作成
    let grid_shape = match images[0].format {
        ImageFormat::CHW => vec![channels, grid_height, grid_width],
        ImageFormat::HWC => vec![grid_height, grid_width, channels],
    };
    
    // Fill with pad value
    // パッド値で填込
    let grid_data: Vec<T> = vec![pad_value; grid_shape.iter().product()];
    let grid_tensor = Tensor::from_vec(grid_data, grid_shape);
    
    // Create grid image
    // グリッド画像を作成
    Image::new(grid_tensor, images[0].format)
}

/// Save image tensor to file
/// 画像テンソルをファイルに保存
pub fn save_image<T: Float + From<f32> + Copy + 'static>(
    image: &Image<T>,
    path: &str,
) -> VisionResult<()> {
    // For now, just print a message - actual implementation would save to file
    // 現在はメッセージを出力するだけ - 実際の実装ではファイルに保存
    println!("Saving image to: {}", path);
    println!("Image size: {:?}", image.size());
    println!("Image channels: {}", image.channels);
    println!("Image format: {:?}", image.format);
    
    Ok(())
}

/// Convert tensor to PIL-like image
/// テンソルをPIL風画像に変換
pub fn to_pil_image<T: Float + From<f32> + Copy + 'static>(
    tensor: &Tensor<T>,
    _mode: Option<&str>,
) -> VisionResult<Image<T>> {
    let shape = tensor.shape();
    
    // Determine format based on tensor shape
    // テンソル形状に基づいて形式を決定
    let (format, _channels) = match shape.len() {
        2 => (ImageFormat::HWC, 1), // Grayscale
        3 => {
            if shape[0] <= 4 { // CHW format (channels first)
                (ImageFormat::CHW, shape[0])
            } else { // HWC format (channels last)
                (ImageFormat::HWC, shape[2])
            }
        },
        _ => return Err(VisionError::InvalidImageShape(
            format!("Expected 2D or 3D tensor, got {:?}", shape)
        )),
    };
    
    Image::new(tensor.clone(), format)
}

/// Convert PIL-like image to tensor
/// PIL風画像をテンソルに変換
pub fn pil_to_tensor<T: Float + From<f32> + Copy>(
    image: &Image<T>,
) -> Tensor<T> {
    image.data.clone()
}

/// Normalize tensor values
/// テンソル値を正規化
pub fn normalize_tensor<T: Float + From<f32> + Copy + 'static>(
    tensor: &Tensor<T>,
    mean: &[T],
    std: &[T],
    _inplace: bool,
) -> VisionResult<Tensor<T>> {
    let _shape = tensor.shape();
    
    if mean.len() != std.len() {
        return Err(VisionError::InvalidTransformParams(
            "Mean and std must have same length".to_string()
        ));
    }
    
    // For now, return cloned tensor - actual implementation would normalize
    // 現在はクローンされたテンソルを返す - 実際の実装では正規化
    Ok(tensor.clone())
}

/// Denormalize tensor values
/// テンソル値を非正規化
pub fn denormalize_tensor<T: Float + From<f32> + Copy + 'static>(
    tensor: &Tensor<T>,
    mean: &[T],
    std: &[T],
    _inplace: bool,
) -> VisionResult<Tensor<T>> {
    let _shape = tensor.shape();
    
    if mean.len() != std.len() {
        return Err(VisionError::InvalidTransformParams(
            "Mean and std must have same length".to_string()
        ));
    }
    
    // For now, return cloned tensor - actual implementation would denormalize
    // 現在はクローンされたテンソルを返す - 実際の実装では非正規化
    Ok(tensor.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_make_grid_creation() {
        let image_data = vec![0.5f32; 3 * 32 * 32];
        let tensor = Tensor::from_vec(image_data, vec![3, 32, 32]);
        let image = Image::new(tensor, ImageFormat::CHW).unwrap();
        
        let images = vec![image.clone(), image.clone(), image.clone(), image.clone()];
        let grid = make_grid(&images, 2, 2, false, None, false, 0.0f32);
        
        assert!(grid.is_ok());
    }
    
    #[test]
    fn test_to_pil_image() {
        let image_data = vec![0.5f32; 3 * 32 * 32];
        let tensor = Tensor::from_vec(image_data, vec![3, 32, 32]);
        let image = to_pil_image(&tensor, None);
        
        assert!(image.is_ok());
        let image = image.unwrap();
        assert_eq!(image.channels, 3);
        assert_eq!(image.format, ImageFormat::CHW);
    }
    
    #[test]
    fn test_normalize_tensor() {
        let image_data = vec![0.5f32; 3 * 32 * 32];
        let tensor = Tensor::from_vec(image_data, vec![3, 32, 32]);
        let mean = vec![0.485f32, 0.456f32, 0.406f32];
        let std = vec![0.229f32, 0.224f32, 0.225f32];
        
        let normalized = normalize_tensor(&tensor, &mean, &std, false);
        assert!(normalized.is_ok());
    }
}