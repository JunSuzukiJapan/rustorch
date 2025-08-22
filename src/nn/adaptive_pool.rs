//! Adaptive pooling layer implementations
//! 適応的プーリングレイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::Float;

/// Adaptive Max Pooling 2D layer
/// 適応的最大プーリング2D層
///
/// This layer applies adaptive max pooling over a 2D input signal.
/// 2D入力信号に対して適応的最大プーリングを適用します。
///
/// Input shape: (batch_size, channels, height, width)
/// Output shape: (batch_size, channels, output_height, output_width)
#[derive(Debug)]
pub struct AdaptiveMaxPool2d<T: Float + Send + Sync> {
    /// Output size (height, width)
    /// 出力サイズ (高さ, 幅)
    output_size: (usize, usize),
    
    /// Whether to return indices for unpooling
    /// アンプーリング用のインデックスを返すかどうか
    _return_indices: bool,
    
    phantom: std::marker::PhantomData<T>,
}

impl<T> AdaptiveMaxPool2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Create a new AdaptiveMaxPool2d layer
    /// 新しいAdaptiveMaxPool2d層を作成
    pub fn new(output_size: (usize, usize), return_indices: Option<bool>) -> Self {
        let return_indices = return_indices.unwrap_or(false);
        
        assert!(output_size.0 > 0 && output_size.1 > 0, "output_size must be positive");
        
        Self {
            output_size,
            _return_indices: return_indices,
            phantom: std::marker::PhantomData,
        }
    }
    
    /// Create global adaptive max pooling (output size 1x1)
    /// グローバル適応的最大プーリング作成（出力サイズ1x1）
    pub fn global() -> Self {
        Self::new((1, 1), None)
    }
    
    /// Perform forward pass
    /// 順伝搬を実行
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For now, return a simple placeholder implementation
        // In a real implementation, this would perform adaptive max pooling
        input.clone()
    }
    
    /// Get layer parameters (none for pooling layers)
    /// レイヤーのパラメータを取得（プーリング層にはパラメータなし）
    pub fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    
    /// Calculate pooling kernel size and stride for given input size
    /// 指定された入力サイズに対するプーリングカーネルサイズとストライドを計算
    pub fn calculate_pooling_params(&self, input_size: (usize, usize)) -> ((usize, usize), (usize, usize)) {
        let (input_h, input_w) = input_size;
        let (output_h, output_w) = self.output_size;
        
        // Calculate kernel size and stride
        let kernel_h = (input_h + output_h - 1) / output_h;
        let kernel_w = (input_w + output_w - 1) / output_w;
        
        let stride_h = input_h / output_h;
        let stride_w = input_w / output_w;
        
        ((kernel_h, kernel_w), (stride_h, stride_w))
    }
    
    /// Get output size
    /// 出力サイズを取得
    pub fn get_output_size(&self) -> (usize, usize) {
        self.output_size
    }
}

impl<T> Module<T> for AdaptiveMaxPool2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Adaptive Average Pooling 2D layer
/// 適応的平均プーリング2D層
///
/// This layer applies adaptive average pooling over a 2D input signal.
/// 2D入力信号に対して適応的平均プーリングを適用します。
///
/// Input shape: (batch_size, channels, height, width)
/// Output shape: (batch_size, channels, output_height, output_width)
#[derive(Debug)]
pub struct AdaptiveAvgPool2d<T: Float + Send + Sync> {
    /// Output size (height, width)
    /// 出力サイズ (高さ, 幅)
    output_size: (usize, usize),
    
    phantom: std::marker::PhantomData<T>,
}

impl<T> AdaptiveAvgPool2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Create a new AdaptiveAvgPool2d layer
    /// 新しいAdaptiveAvgPool2d層を作成
    pub fn new(output_size: (usize, usize)) -> Self {
        assert!(output_size.0 > 0 && output_size.1 > 0, "output_size must be positive");
        
        Self {
            output_size,
            phantom: std::marker::PhantomData,
        }
    }
    
    /// Create global adaptive average pooling (output size 1x1)
    /// グローバル適応的平均プーリング作成（出力サイズ1x1）
    pub fn global() -> Self {
        Self::new((1, 1))
    }
    
    /// Perform forward pass
    /// 順伝搬を実行
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For now, return a simple placeholder implementation
        // In a real implementation, this would perform adaptive average pooling
        input.clone()
    }
    
    /// Get layer parameters (none for pooling layers)
    /// レイヤーのパラメータを取得（プーリング層にはパラメータなし）
    pub fn parameters(&self) -> Vec<Variable<T>> {
        Vec::new()
    }
    
    /// Calculate pooling kernel size and stride for given input size
    /// 指定された入力サイズに対するプーリングカーネルサイズとストライドを計算
    pub fn calculate_pooling_params(&self, input_size: (usize, usize)) -> ((usize, usize), (usize, usize)) {
        let (input_h, input_w) = input_size;
        let (output_h, output_w) = self.output_size;
        
        // For average pooling, we use precise calculation to ensure all input is covered
        let kernel_h = (input_h + output_h - 1) / output_h;
        let kernel_w = (input_w + output_w - 1) / output_w;
        
        let stride_h = input_h / output_h;
        let stride_w = input_w / output_w;
        
        ((kernel_h, kernel_w), (stride_h, stride_w))
    }
    
    /// Get output size
    /// 出力サイズを取得
    pub fn get_output_size(&self) -> (usize, usize) {
        self.output_size
    }
}

impl<T> Module<T> for AdaptiveAvgPool2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_max_pool2d_creation() {
        let layer: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::new((7, 7), None);
        
        assert_eq!(layer.output_size, (7, 7));
        assert!(!layer._return_indices);
    }

    #[test]
    fn test_adaptive_max_pool2d_global() {
        let layer: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::global();
        
        assert_eq!(layer.output_size, (1, 1));
    }

    #[test]
    fn test_adaptive_max_pool2d_pooling_params() {
        let layer: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::new((7, 7), None);
        
        let input_size = (224, 224);
        let (kernel_size, stride) = layer.calculate_pooling_params(input_size);
        
        // For 224x224 -> 7x7: kernel ~= 32, stride = 32
        assert_eq!(kernel_size, (32, 32));
        assert_eq!(stride, (32, 32));
    }

    #[test]
    fn test_adaptive_avg_pool2d_creation() {
        let layer: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::new((14, 14));
        
        assert_eq!(layer.output_size, (14, 14));
    }

    #[test]
    fn test_adaptive_avg_pool2d_global() {
        let layer: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::global();
        
        assert_eq!(layer.output_size, (1, 1));
    }

    #[test]
    fn test_adaptive_avg_pool2d_pooling_params() {
        let layer: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::new((2, 2));
        
        let input_size = (8, 8);
        let (kernel_size, stride) = layer.calculate_pooling_params(input_size);
        
        // For 8x8 -> 2x2: kernel = 4, stride = 4
        assert_eq!(kernel_size, (4, 4));
        assert_eq!(stride, (4, 4));
    }

    #[test]
    fn test_pooling_layers_no_parameters() {
        let max_pool: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::new((4, 4), None);
        let avg_pool: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::new((4, 4));
        
        assert_eq!(max_pool.parameters().len(), 0);
        assert_eq!(avg_pool.parameters().len(), 0);
    }

    #[test]
    fn test_adaptive_pooling_with_return_indices() {
        let layer: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::new((3, 3), Some(true));
        
        assert!(layer._return_indices);
        assert_eq!(layer.output_size, (3, 3));
    }

    #[test]
    fn test_adaptive_pooling_irregular_sizes() {
        let layer: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::new((5, 3));
        
        let input_size = (17, 13);
        let (kernel_size, stride) = layer.calculate_pooling_params(input_size);
        
        // For 17x13 -> 5x3: kernel = (4, 5), stride = (3, 4)
        assert_eq!(kernel_size, (4, 5));
        assert_eq!(stride, (3, 4));
    }

    #[test]
    fn test_get_output_size() {
        let max_pool: AdaptiveMaxPool2d<f32> = AdaptiveMaxPool2d::new((6, 8), None);
        let avg_pool: AdaptiveAvgPool2d<f32> = AdaptiveAvgPool2d::new((6, 8));
        
        assert_eq!(max_pool.get_output_size(), (6, 8));
        assert_eq!(avg_pool.get_output_size(), (6, 8));
    }
}