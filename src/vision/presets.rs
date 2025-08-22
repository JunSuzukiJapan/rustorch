//! Pre-configured transformation pipelines for common use cases
//! 一般的な用途のための事前設定された変換パイプライン
//!
//! This module provides ready-to-use transformation pipelines for popular
//! computer vision tasks such as image classification, object detection, and segmentation.
//!
//! このモジュールは、画像分類、物体検出、セグメンテーションなどの人気の
//! コンピュータビジョンタスクのためのすぐに使える変換パイプラインを提供します。

use crate::vision::pipeline::{Pipeline, PipelineBuilder, ExecutionMode, predicates};
use crate::vision::transforms::{
    Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, 
    Normalize, ToTensor, InterpolationMode
};
use num_traits::Float;

/// Image classification preprocessing pipeline
/// 画像分類前処理パイプライン
pub struct ImageNetPreprocessing;

impl ImageNetPreprocessing {
    /// Create ImageNet training pipeline
    /// ImageNet訓練パイプラインを作成
    pub fn training<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("imagenet_training".to_string())
            .transform(Box::new(Resize::new((256, 256))))
            .transform(Box::new(RandomCrop::new((224, 224)).with_padding((4, 4))))
            .transform(Box::new(RandomHorizontalFlip::new(0.5)))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(500)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
    
    /// Create ImageNet validation pipeline
    /// ImageNet検証パイプラインを作成
    pub fn validation<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("imagenet_validation".to_string())
            .transform(Box::new(Resize::new((256, 256))))
            .transform(Box::new(CenterCrop::new((224, 224))))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(200)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

/// CIFAR-10/100 preprocessing pipeline
/// CIFAR-10/100前処理パイプライン
pub struct CIFARPreprocessing;

impl CIFARPreprocessing {
    /// Create CIFAR training pipeline
    /// CIFAR訓練パイプラインを作成
    pub fn training<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("cifar_training".to_string())
            .transform(Box::new(RandomCrop::new((32, 32)).with_padding((4, 4))))
            .transform(Box::new(RandomHorizontalFlip::new(0.5)))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::new(
                vec![<T as From<f32>>::from(0.4914), <T as From<f32>>::from(0.4822), <T as From<f32>>::from(0.4465)],
                vec![<T as From<f32>>::from(0.2023), <T as From<f32>>::from(0.1994), <T as From<f32>>::from(0.2010)]
            ).unwrap()))
            .cache(1000)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
    
    /// Create CIFAR validation pipeline
    /// CIFAR検証パイプラインを作成
    pub fn validation<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("cifar_validation".to_string())
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::new(
                vec![<T as From<f32>>::from(0.4914), <T as From<f32>>::from(0.4822), <T as From<f32>>::from(0.4465)],
                vec![<T as From<f32>>::from(0.2023), <T as From<f32>>::from(0.1994), <T as From<f32>>::from(0.2010)]
            ).unwrap()))
            .cache(500)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

/// Object detection preprocessing pipeline
/// 物体検出前処理パイプライン
pub struct ObjectDetectionPreprocessing;

impl ObjectDetectionPreprocessing {
    /// Create COCO-style object detection pipeline
    /// COCO風物体検出パイプラインを作成
    pub fn coco_training<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("coco_detection_training".to_string())
            // Only resize large images to avoid upscaling small objects
            // 小さなオブジェクトのアップスケーリングを避けるため、大きな画像のみリサイズ
            .conditional_transform(
                Box::new(Resize::new((800, 800))),
                predicates::min_size(800, 800),
                "resize_large_images".to_string()
            )
            .transform(Box::new(RandomHorizontalFlip::new(0.5)))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(200)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

/// Segmentation preprocessing pipeline
/// セグメンテーション前処理パイプライン
pub struct SegmentationPreprocessing;

impl SegmentationPreprocessing {
    /// Create semantic segmentation pipeline
    /// セマンティックセグメンテーションパイプラインを作成
    pub fn semantic_training<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("segmentation_training".to_string())
            .transform(Box::new(Resize::new((512, 512))))
            .transform(Box::new(RandomHorizontalFlip::new(0.5)))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(100)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

/// Medical imaging preprocessing pipeline
/// 医用画像前処理パイプライン
pub struct MedicalImagingPreprocessing;

impl MedicalImagingPreprocessing {
    /// Create X-ray/CT scan preprocessing pipeline
    /// X線/CTスキャン前処理パイプラインを作成
    pub fn xray_preprocessing<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("medical_xray".to_string())
            .transform(Box::new(Resize::new((512, 512))))
            .transform(Box::new(CenterCrop::new((448, 448))))
            // Only apply to grayscale images
            // グレースケール画像にのみ適用
            .conditional_transform(
                Box::new(ToTensor::new()),
                predicates::channels_eq(1),
                "grayscale_tensor".to_string()
            )
            // Specific normalization for medical images
            // 医用画像用の特定の正規化
            .transform(Box::new(Normalize::new(
                vec![<T as From<f32>>::from(0.449)],
                vec![<T as From<f32>>::from(0.226)]
            ).unwrap()))
            .cache(50)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

/// Mobile/edge device optimized pipeline
/// モバイル/エッジデバイス最適化パイプライン
pub struct MobileOptimizedPreprocessing;

impl MobileOptimizedPreprocessing {
    /// Create mobile-optimized pipeline with smaller cache and batch processing
    /// 小さなキャッシュとバッチ処理を持つモバイル最適化パイプラインを作成
    pub fn mobile_inference<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>() -> Pipeline<T> {
        PipelineBuilder::new("mobile_inference".to_string())
            .transform(Box::new(Resize::new((224, 224))))
            .transform(Box::new(CenterCrop::new((224, 224))))
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(10) // Smaller cache for memory-constrained devices
            .execution_mode(ExecutionMode::Batch)
            .build()
    }
}

/// Custom pipeline factory for creating domain-specific pipelines
/// ドメイン固有パイプラインを作成するためのカスタムパイプラインファクトリ
pub struct CustomPipelineFactory;

impl CustomPipelineFactory {
    /// Create a pipeline for high-resolution image processing
    /// 高解像度画像処理用パイプラインを作成
    pub fn high_resolution<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>(
        target_size: (usize, usize),
        enable_augmentation: bool,
    ) -> Pipeline<T> {
        let mut builder = PipelineBuilder::new("high_resolution_custom".to_string())
            .transform(Box::new(Resize::new(target_size).with_interpolation(InterpolationMode::Bicubic)))
            .transform(Box::new(CenterCrop::new(target_size)));
            
        if enable_augmentation {
            builder = builder.transform(Box::new(RandomHorizontalFlip::new(0.3)));
        }
        
        builder
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(50)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
    
    /// Create a pipeline with probabilistic augmentations
    /// 確率的拡張を持つパイプラインを作成
    pub fn probabilistic_augmentation<T: Float + From<f32> + Clone + 'static + std::fmt::Debug>(
        base_size: (usize, usize),
        augment_probability: f64,
    ) -> Pipeline<T> {
        PipelineBuilder::new("probabilistic_augmentation".to_string())
            .transform(Box::new(Resize::new(base_size)))
            .conditional_transform(
                Box::new(RandomCrop::new(base_size).with_padding((8, 8))),
                predicates::probability(augment_probability),
                "random_crop_probabilistic".to_string()
            )
            .conditional_transform(
                Box::new(RandomHorizontalFlip::new(1.0)), // Always flip if condition is met
                predicates::probability(augment_probability),
                "random_flip_probabilistic".to_string()
            )
            .transform(Box::new(ToTensor::new()))
            .transform(Box::new(Normalize::imagenet()))
            .cache(200)
            .execution_mode(ExecutionMode::Sequential)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imagenet_preprocessing() {
        let training_pipeline = ImageNetPreprocessing::training::<f32>();
        let validation_pipeline = ImageNetPreprocessing::validation::<f32>();
        
        assert_eq!(training_pipeline.name(), "imagenet_training");
        assert_eq!(validation_pipeline.name(), "imagenet_validation");
        assert_eq!(training_pipeline.len(), 5);
        assert_eq!(validation_pipeline.len(), 4);
    }
    
    #[test]
    fn test_cifar_preprocessing() {
        let training_pipeline = CIFARPreprocessing::training::<f32>();
        let validation_pipeline = CIFARPreprocessing::validation::<f32>();
        
        assert_eq!(training_pipeline.name(), "cifar_training");
        assert_eq!(validation_pipeline.name(), "cifar_validation");
    }
    
    #[test]
    fn test_object_detection_preprocessing() {
        let coco_pipeline = ObjectDetectionPreprocessing::coco_training::<f32>();
        assert_eq!(coco_pipeline.name(), "coco_detection_training");
    }
    
    #[test]
    fn test_custom_pipeline_factory() {
        let high_res_pipeline = CustomPipelineFactory::high_resolution::<f32>((512, 512), true);
        let prob_aug_pipeline = CustomPipelineFactory::probabilistic_augmentation::<f32>((224, 224), 0.5);
        
        assert_eq!(high_res_pipeline.name(), "high_resolution_custom");
        assert_eq!(prob_aug_pipeline.name(), "probabilistic_augmentation");
    }
    
    #[test]
    fn test_medical_imaging_preprocessing() {
        let xray_pipeline = MedicalImagingPreprocessing::xray_preprocessing::<f32>();
        assert_eq!(xray_pipeline.name(), "medical_xray");
    }
    
    #[test]
    fn test_mobile_optimized_preprocessing() {
        let mobile_pipeline = MobileOptimizedPreprocessing::mobile_inference::<f32>();
        assert_eq!(mobile_pipeline.name(), "mobile_inference");
        
        // Check that cache size is small for mobile optimization
        // モバイル最適化のためにキャッシュサイズが小さいことを確認
        let (_, max_cache) = mobile_pipeline.cache_info();
        assert_eq!(max_cache, 10);
    }
}