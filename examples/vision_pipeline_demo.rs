//! Vision Pipeline Demo
//! ビジョンパイプラインデモ
//!
//! This example demonstrates the advanced data transformation pipeline functionality
//! for computer vision tasks, including conditional transforms, caching, and performance monitoring.
//!
//! この例では、条件付き変換、キャッシュ、パフォーマンス監視を含む
//! コンピュータビジョンタスク用の高度なデータ変換パイプライン機能を実演します。

use rustorch::prelude::*;
use rustorch::vision::{
    pipeline::{predicates, ExecutionMode, PipelineBuilder},
    Image, ImageFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 RusTorch Vision Pipeline Demo");
    println!("================================");

    // Create sample images for demonstration
    // デモンストレーション用のサンプル画像を作成
    let small_image = create_sample_image(64, 64)?;
    let large_image = create_sample_image(512, 512)?;

    // Demo 1: Basic Pipeline Usage
    // デモ1: 基本的なパイプライン使用
    println!("\n📋 Demo 1: Basic Pipeline");
    demo_basic_pipeline(&small_image)?;

    // Demo 2: Conditional Transformations
    // デモ2: 条件付き変換
    println!("\n🔀 Demo 2: Conditional Transformations");
    demo_conditional_transforms(&small_image, &large_image)?;

    // Demo 3: Preset Pipelines
    // デモ3: プリセットパイプライン
    println!("\n🏭 Demo 3: Preset Pipelines");
    demo_preset_pipelines(&small_image)?;

    // Demo 4: Pipeline Performance Monitoring
    // デモ4: パイプラインパフォーマンス監視
    println!("\n📊 Demo 4: Performance Monitoring");
    demo_performance_monitoring(&small_image)?;

    // Demo 5: Batch Processing
    // デモ5: バッチ処理
    println!("\n📦 Demo 5: Batch Processing");
    demo_batch_processing(vec![small_image.clone(), large_image.clone()])?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

fn create_sample_image(
    height: usize,
    width: usize,
) -> Result<Image<f32>, Box<dyn std::error::Error>> {
    // Create a sample image with gradient pattern
    // グラデーションパターンのサンプル画像を作成
    let mut image_data = Vec::new();
    for c in 0..3 {
        // RGB channels
        for h in 0..height {
            for w in 0..width {
                let value =
                    (h as f32 / height as f32 + w as f32 / width as f32 + c as f32 * 0.1) / 2.0;
                image_data.push(value);
            }
        }
    }

    let tensor = Tensor::from_vec(image_data, vec![3, height, width]);
    let image = Image::new(tensor, ImageFormat::CHW)?;
    Ok(image)
}

fn demo_basic_pipeline(image: &Image<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a basic pipeline with multiple transforms
    // 複数の変換を持つ基本的なパイプラインを作成
    let pipeline = PipelineBuilder::new("basic_demo".to_string())
        .transform(Box::new(Resize::new((224, 224))))
        .transform(Box::new(CenterCrop::new((200, 200))))
        .transform(Box::new(RandomHorizontalFlip::new(0.5)))
        .transform(Box::new(ToTensor::new()))
        .transform(Box::new(Normalize::imagenet()))
        .cache(10)
        .build();

    println!("  Pipeline: {}", pipeline.name());
    println!("  Number of transforms: {}", pipeline.len());

    // Apply the pipeline
    // パイプラインを適用
    let start_time = std::time::Instant::now();
    let result = pipeline.apply(image)?;
    let processing_time = start_time.elapsed();

    println!("  Input shape: {:?}", image.data.shape());
    println!("  Output shape: {:?}", result.data.shape());
    println!("  Processing time: {:?}", processing_time);

    // Get pipeline statistics
    // パイプライン統計を取得
    let stats = pipeline.get_stats();
    println!("  Images processed: {}", stats.total_processed);
    println!(
        "  Average processing time: {:.2}μs",
        stats.avg_processing_time_us
    );

    Ok(())
}

fn demo_conditional_transforms(
    small_image: &Image<f32>,
    large_image: &Image<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a pipeline with conditional transforms
    // 条件付き変換を持つパイプラインを作成
    let pipeline = PipelineBuilder::new("conditional_demo".to_string())
        .conditional_transform(
            Box::new(Resize::new((512, 512))),
            predicates::min_size(300, 300),
            "resize_large_images".to_string(),
        )
        .conditional_transform(
            Box::new(RandomCrop::new((224, 224))),
            predicates::probability(0.7),
            "probabilistic_crop".to_string(),
        )
        .transform(Box::new(ToTensor::new()))
        .build();

    println!("  Testing conditional transforms:");

    // Test on small image
    // 小さい画像でテスト
    println!(
        "    Small image ({}x{})",
        small_image.width, small_image.height
    );
    let small_result = pipeline.apply(small_image)?;
    println!("      Output: {:?}", small_result.data.shape());

    // Test on large image
    // 大きい画像でテスト
    println!(
        "    Large image ({}x{})",
        large_image.width, large_image.height
    );
    let large_result = pipeline.apply(large_image)?;
    println!("      Output: {:?}", large_result.data.shape());

    Ok(())
}

fn demo_preset_pipelines(image: &Image<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing preset pipelines:");

    // ImageNet training pipeline
    // ImageNet訓練パイプライン
    let imagenet_pipeline = ImageNetPreprocessing::training();
    let result = imagenet_pipeline.apply(image)?;
    println!(
        "    ImageNet Training: {} transforms → {:?}",
        imagenet_pipeline.len(),
        result.data.shape()
    );

    // CIFAR training pipeline
    // CIFAR訓練パイプライン
    let cifar_pipeline = CIFARPreprocessing::training();
    let result = cifar_pipeline.apply(image)?;
    println!(
        "    CIFAR Training: {} transforms → {:?}",
        cifar_pipeline.len(),
        result.data.shape()
    );

    // Mobile optimized pipeline
    // モバイル最適化パイプライン
    let mobile_pipeline = MobileOptimizedPreprocessing::mobile_inference();
    let result = mobile_pipeline.apply(image)?;
    let (cache_size, max_cache) = mobile_pipeline.cache_info();
    println!(
        "    Mobile Optimized: {} transforms → {:?} (cache: {}/{})",
        mobile_pipeline.len(),
        result.data.shape(),
        cache_size,
        max_cache
    );

    Ok(())
}

fn demo_performance_monitoring(image: &Image<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a pipeline with caching enabled
    // キャッシュを有効にしたパイプラインを作成
    let pipeline = PipelineBuilder::new("performance_demo".to_string())
        .transform(Box::new(Resize::new((256, 256))))
        .transform(Box::new(CenterCrop::new((224, 224))))
        .transform(Box::new(ToTensor::new()))
        .cache(5)
        .build();

    println!("  Performance monitoring test:");

    // Process the same image multiple times
    // 同じ画像を複数回処理
    for i in 1..=5 {
        let start = std::time::Instant::now();
        let _result = pipeline.apply(image)?;
        let elapsed = start.elapsed();

        let stats = pipeline.get_stats();
        let (cache_size, max_cache) = pipeline.cache_info();

        println!(
            "    Iteration {}: {:?} | Cache: {}/{} | Hits: {} | Misses: {}",
            i, elapsed, cache_size, max_cache, stats.cache_hits, stats.cache_misses
        );
    }

    let final_stats = pipeline.get_stats();
    println!("  Final statistics:");
    println!("    Total processed: {}", final_stats.total_processed);
    println!(
        "    Average time: {:.2}μs",
        final_stats.avg_processing_time_us
    );
    println!(
        "    Cache hit rate: {:.1}%",
        final_stats.cache_hits as f64 / (final_stats.cache_hits + final_stats.cache_misses) as f64
            * 100.0
    );

    Ok(())
}

fn demo_batch_processing(images: Vec<Image<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a pipeline optimized for batch processing
    // バッチ処理用に最適化されたパイプラインを作成
    let pipeline = PipelineBuilder::new("batch_demo".to_string())
        .transform(Box::new(Resize::new((224, 224))))
        .transform(Box::new(ToTensor::new()))
        .transform(Box::new(Normalize::imagenet()))
        .execution_mode(ExecutionMode::Batch)
        .build();

    println!("  Batch processing {} images:", images.len());

    let start_time = std::time::Instant::now();
    let results = pipeline.apply_batch(&images)?;
    let batch_time = start_time.elapsed();

    println!("    Batch processing time: {:?}", batch_time);
    println!(
        "    Average per image: {:?}",
        batch_time / images.len() as u32
    );
    println!("    Results: {} images processed", results.len());

    for (i, result) in results.iter().enumerate() {
        println!("      Image {}: {:?}", i + 1, result.data.shape());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_image_creation() {
        let image = create_sample_image(32, 32).unwrap();
        assert_eq!(image.width, 32);
        assert_eq!(image.height, 32);
        assert_eq!(image.channels, 3);
        assert_eq!(image.format, ImageFormat::CHW);
    }

    #[test]
    fn test_pipeline_demos() {
        let image = create_sample_image(64, 64).unwrap();

        // Test that all demo functions can run without errors
        // すべてのデモ関数がエラーなしで実行できることをテスト
        assert!(demo_basic_pipeline(&image).is_ok());

        let large_image = create_sample_image(256, 256).unwrap();
        assert!(demo_conditional_transforms(&image, &large_image).is_ok());

        assert!(demo_preset_pipelines(&image).is_ok());
        assert!(demo_performance_monitoring(&image).is_ok());
        assert!(demo_batch_processing(vec![image.clone(), large_image.clone()]).is_ok());
    }
}
