//! Data transformation pipeline for computer vision
//! コンピュータビジョン用データ変換パイプライン
//!
//! This module provides advanced pipeline functionality for chaining multiple
//! transformations with conditional logic, parallel processing, and caching.
//!
//! このモジュールは、条件付きロジック、並列処理、キャッシュ機能を持つ
//! 複数変換のチェーン化のための高度なパイプライン機能を提供します。

use crate::vision::{Image, ImageFormat};
use crate::error::RusTorchResult;
use crate::vision::transforms::Transform;
use num_traits::Float;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::fmt;

/// Pipeline statistics for monitoring performance
/// パフォーマンス監視用パイプライン統計
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total number of images processed
    /// 処理された画像の総数
    pub total_processed: usize,
    /// Average processing time per image (microseconds)
    /// 1画像あたりの平均処理時間（マイクロ秒）
    pub avg_processing_time_us: f64,
    /// Number of cache hits
    /// キャッシュヒット数
    pub cache_hits: usize,
    /// Number of cache misses
    /// キャッシュミス数
    pub cache_misses: usize,
    /// Memory usage estimate (bytes)
    /// メモリ使用量推定値（バイト）
    pub memory_usage_bytes: usize,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            total_processed: 0,
            avg_processing_time_us: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            memory_usage_bytes: 0,
        }
    }
}

/// Pipeline execution mode
/// パイプライン実行モード
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Sequential execution (default)
    /// 逐次実行（デフォルト）
    Sequential,
    /// Parallel execution where possible
    /// 可能な場合は並列実行
    Parallel,
    /// Batch processing mode
    /// バッチ処理モード
    Batch,
}

/// Conditional transformation that applies based on a predicate
/// 述語に基づいて適用される条件付き変換
pub struct ConditionalTransform<T: Float> {
    /// The transformation to apply
    /// 適用する変換
    pub transform: Box<dyn Transform<T>>,
    /// Predicate function that determines if transform should be applied
    /// 変換を適用するかどうかを決定する述語関数
    pub predicate: Box<dyn Fn(&Image<T>) -> bool + Send + Sync>,
    /// Human-readable name for this conditional transform
    /// この条件付き変換の人間が読める名前
    pub name: String,
}

impl<T: Float> fmt::Debug for ConditionalTransform<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalTransform")
            .field("name", &self.name)
            .finish()
    }
}

impl<T: Float + 'static + std::fmt::Debug> Transform<T> for ConditionalTransform<T> {
    fn apply(&self, image: &Image<T>) -> RusTorchResult<Image<T>> {
        if (self.predicate)(image) {
            self.transform.apply(image)
        } else {
            Ok(image.clone())
        }
    }
}

/// Advanced data transformation pipeline with caching and conditional logic
/// キャッシュと条件付きロジックを持つ高度なデータ変換パイプライン
pub struct Pipeline<T: Float> {
    /// List of transformations in the pipeline
    /// パイプライン内の変換リスト
    transforms: Vec<Box<dyn Transform<T>>>,
    /// Pipeline execution mode
    /// パイプライン実行モード
    execution_mode: ExecutionMode,
    /// Cache for transformed images
    /// 変換済み画像のキャッシュ
    cache: Arc<RwLock<HashMap<String, Image<T>>>>,
    /// Maximum cache size (number of images)
    /// 最大キャッシュサイズ（画像数）
    max_cache_size: usize,
    /// Enable caching
    /// キャッシュを有効にする
    cache_enabled: bool,
    /// Pipeline statistics
    /// パイプライン統計
    stats: Arc<RwLock<PipelineStats>>,
    /// Pipeline name for identification
    /// 識別用パイプライン名
    name: String,
}

impl<T: Float + Clone + 'static + std::fmt::Debug> Pipeline<T> {
    /// Create a new pipeline
    /// 新しいパイプラインを作成
    pub fn new(name: String) -> Self {
        Self {
            transforms: Vec::new(),
            execution_mode: ExecutionMode::Sequential,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: 1000,
            cache_enabled: false,
            stats: Arc::new(RwLock::new(PipelineStats::default())),
            name,
        }
    }
    
    /// Add a transformation to the pipeline
    /// パイプラインに変換を追加
    pub fn add_transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.transforms.push(transform);
        self
    }
    
    /// Add a conditional transformation
    /// 条件付き変換を追加
    pub fn add_conditional_transform<F>(mut self, 
        transform: Box<dyn Transform<T>>,
        predicate: F,
        name: String,
    ) -> Self 
    where 
        F: Fn(&Image<T>) -> bool + Send + Sync + 'static,
    {
        let conditional = ConditionalTransform {
            transform,
            predicate: Box::new(predicate),
            name,
        };
        self.transforms.push(Box::new(conditional));
        self
    }
    
    /// Set execution mode
    /// 実行モードを設定
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }
    
    /// Enable caching with specified max size
    /// 指定された最大サイズでキャッシュを有効にする
    pub fn with_cache(mut self, max_size: usize) -> Self {
        self.cache_enabled = true;
        self.max_cache_size = max_size;
        self
    }
    
    /// Apply all transformations in the pipeline
    /// パイプライン内のすべての変換を適用
    pub fn apply(&self, image: &Image<T>) -> RusTorchResult<Image<T>> {
        let start_time = std::time::Instant::now();
        
        // Generate cache key based on image properties
        // 画像プロパティに基づいてキャッシュキーを生成
        let cache_key = if self.cache_enabled {
            Some(self.generate_cache_key(image))
        } else {
            None
        };
        
        // Check cache first
        // 最初にキャッシュをチェック
        if let Some(ref key) = cache_key {
            if let Ok(cache) = self.cache.read() {
                if let Some(cached_image) = cache.get(key) {
                    // Update stats for cache hit
                    // キャッシュヒットの統計を更新
                    if let Ok(mut stats) = self.stats.write() {
                        stats.cache_hits += 1;
                    }
                    return Ok(cached_image.clone());
                }
            }
        }
        
        // Apply transformations
        // 変換を適用
        let result = match self.execution_mode {
            ExecutionMode::Sequential => self.apply_sequential(image),
            ExecutionMode::Parallel => self.apply_parallel(image),
            ExecutionMode::Batch => self.apply_batch(&[image.clone()]).map(|mut v| v.remove(0)),
        }?;
        
        // Update cache
        // キャッシュを更新
        if let Some(key) = cache_key {
            if let Ok(mut cache) = self.cache.write() {
                // Remove oldest entries if cache is full
                // キャッシュが満杯の場合、最古のエントリを削除
                if cache.len() >= self.max_cache_size {
                    // Simple eviction: remove first entry (FIFO-like behavior)
                    // シンプルな退去: 最初のエントリを削除（FIFO風動作）
                    if let Some(oldest_key) = cache.keys().next().cloned() {
                        cache.remove(&oldest_key);
                    }
                }
                cache.insert(key, result.clone());
                
                // Update stats for cache miss
                // キャッシュミスの統計を更新
                if let Ok(mut stats) = self.stats.write() {
                    stats.cache_misses += 1;
                }
            }
        }
        
        // Update performance stats
        // パフォーマンス統計を更新
        let processing_time = start_time.elapsed().as_micros() as f64;
        if let Ok(mut stats) = self.stats.write() {
            stats.total_processed += 1;
            let total = stats.total_processed as f64;
            stats.avg_processing_time_us = 
                (stats.avg_processing_time_us * (total - 1.0) + processing_time) / total;
        }
        
        Ok(result)
    }
    
    /// Apply transformations sequentially
    /// 変換を逐次的に適用
    fn apply_sequential(&self, image: &Image<T>) -> RusTorchResult<Image<T>> {
        let mut result = image.clone();
        
        for transform in &self.transforms {
            result = transform.apply(&result)?;
        }
        
        Ok(result)
    }
    
    /// Apply transformations with parallel processing where possible
    /// 可能な場合は並列処理で変換を適用
    fn apply_parallel(&self, image: &Image<T>) -> RusTorchResult<Image<T>> {
        // For now, fall back to sequential processing
        // 現在は逐次処理にフォールバック
        // Future implementation could use rayon for parallel processing
        // 将来の実装では並列処理にrayonを使用可能
        self.apply_sequential(image)
    }
    
    /// Apply transformations in batch mode
    /// バッチモードで変換を適用
    pub fn apply_batch(&self, images: &[Image<T>]) -> RusTorchResult<Vec<Image<T>>> {
        let mut results = Vec::with_capacity(images.len());
        
        for image in images {
            results.push(self.apply_sequential(image)?);
        }
        
        Ok(results)
    }
    
    /// Generate cache key for an image
    /// 画像のキャッシュキーを生成
    fn generate_cache_key(&self, image: &Image<T>) -> String {
        // Simple cache key based on image dimensions and format
        // 画像の次元と形式に基づく簡単なキャッシュキー
        format!("{}_{}_{}_{:?}_{}", 
                self.name,
                image.height, 
                image.width, 
                image.format,
                image.channels)
    }
    
    /// Get pipeline statistics
    /// パイプライン統計を取得
    pub fn get_stats(&self) -> PipelineStats {
        if let Ok(stats) = self.stats.read() {
            stats.clone()
        } else {
            PipelineStats::default()
        }
    }
    
    /// Reset pipeline statistics
    /// パイプライン統計をリセット
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.write() {
            *stats = PipelineStats::default();
        }
    }
    
    /// Clear the cache
    /// キャッシュをクリア
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
    
    /// Get cache statistics
    /// キャッシュ統計を取得
    pub fn cache_info(&self) -> (usize, usize) {
        if let Ok(cache) = self.cache.read() {
            (cache.len(), self.max_cache_size)
        } else {
            (0, self.max_cache_size)
        }
    }
    
    /// Get the number of transformations in the pipeline
    /// パイプライン内の変換数を取得
    pub fn len(&self) -> usize {
        self.transforms.len()
    }
    
    /// Check if pipeline is empty
    /// パイプラインが空かどうかをチェック
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
    
    /// Get pipeline name
    /// パイプライン名を取得
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<T: Float + 'static + std::fmt::Debug> fmt::Debug for Pipeline<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("num_transforms", &self.transforms.len())
            .field("execution_mode", &self.execution_mode)
            .field("cache_enabled", &self.cache_enabled)
            .field("max_cache_size", &self.max_cache_size)
            .finish()
    }
}

impl<T: Float + 'static + std::fmt::Debug> Transform<T> for Pipeline<T> {
    fn apply(&self, image: &Image<T>) -> RusTorchResult<Image<T>> {
        self.apply(image)
    }
}

/// Pipeline builder for fluent construction
/// 流暢な構築のためのパイプラインビルダー
pub struct PipelineBuilder<T: Float> {
    pipeline: Pipeline<T>,
}

impl<T: Float + Clone + 'static + std::fmt::Debug> PipelineBuilder<T> {
    /// Create a new pipeline builder
    /// 新しいパイプラインビルダーを作成
    pub fn new(name: String) -> Self {
        Self {
            pipeline: Pipeline::new(name),
        }
    }
    
    /// Add a transformation
    /// 変換を追加
    pub fn transform(mut self, transform: Box<dyn Transform<T>>) -> Self {
        self.pipeline = self.pipeline.add_transform(transform);
        self
    }
    
    /// Add a conditional transformation
    /// 条件付き変換を追加
    pub fn conditional_transform<F>(mut self,
        transform: Box<dyn Transform<T>>,
        predicate: F,
        name: String,
    ) -> Self 
    where 
        F: Fn(&Image<T>) -> bool + Send + Sync + 'static,
    {
        self.pipeline = self.pipeline.add_conditional_transform(transform, predicate, name);
        self
    }
    
    /// Set execution mode
    /// 実行モードを設定
    pub fn execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.pipeline = self.pipeline.with_execution_mode(mode);
        self
    }
    
    /// Enable caching
    /// キャッシュを有効にする
    pub fn cache(mut self, max_size: usize) -> Self {
        self.pipeline = self.pipeline.with_cache(max_size);
        self
    }
    
    /// Build the pipeline
    /// パイプラインを構築
    pub fn build(self) -> Pipeline<T> {
        self.pipeline
    }
}

/// Predefined common predicates for conditional transforms
/// 条件付き変換用の事前定義された一般的述語
pub mod predicates {
    use super::*;
    
    /// Only apply transform to images larger than specified dimensions
    /// 指定された次元より大きい画像にのみ変換を適用
    pub fn min_size<T: Float>(min_width: usize, min_height: usize) -> Box<dyn Fn(&Image<T>) -> bool + Send + Sync> {
        Box::new(move |image: &Image<T>| {
            image.width >= min_width && image.height >= min_height
        })
    }
    
    /// Only apply transform to images smaller than specified dimensions
    /// 指定された次元より小さい画像にのみ変換を適用
    pub fn max_size<T: Float>(max_width: usize, max_height: usize) -> Box<dyn Fn(&Image<T>) -> bool + Send + Sync> {
        Box::new(move |image: &Image<T>| {
            image.width <= max_width && image.height <= max_height
        })
    }
    
    /// Only apply transform to images with specific format
    /// 特定の形式の画像にのみ変換を適用
    pub fn format_is<T: Float>(target_format: ImageFormat) -> Box<dyn Fn(&Image<T>) -> bool + Send + Sync> {
        Box::new(move |image: &Image<T>| {
            image.format == target_format
        })
    }
    
    /// Only apply transform to images with specific number of channels
    /// 特定のチャンネル数の画像にのみ変換を適用
    pub fn channels_eq<T: Float>(target_channels: usize) -> Box<dyn Fn(&Image<T>) -> bool + Send + Sync> {
        Box::new(move |image: &Image<T>| {
            image.channels == target_channels
        })
    }
    
    /// Apply transform with specified probability
    /// 指定された確率で変換を適用
    pub fn probability<T: Float>(prob: f64) -> Box<dyn Fn(&Image<T>) -> bool + Send + Sync> {
        Box::new(move |_: &Image<T>| {
            use rand::Rng;
            rand::thread_rng().gen::<f64>() < prob
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::transforms::{Resize, ToTensor};
    use crate::tensor::Tensor;
    
    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::<f32>::new("test_pipeline".to_string());
        assert_eq!(pipeline.name(), "test_pipeline");
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }
    
    #[test]
    fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::<f32>::new("test".to_string())
            .transform(Box::new(Resize::new((224, 224))))
            .transform(Box::new(ToTensor::new()))
            .cache(100)
            .execution_mode(ExecutionMode::Sequential)
            .build();
            
        assert_eq!(pipeline.len(), 2);
        assert_eq!(pipeline.name(), "test");
        assert!(!pipeline.is_empty());
    }
    
    #[test]
    fn test_conditional_transform() {
        let predicate = predicates::min_size::<f32>(100, 100);
        
        // Create a small image that should not trigger the transform
        let small_image_data = vec![0.5f32; 3 * 50 * 50];
        let small_tensor = Tensor::from_vec(small_image_data, vec![3, 50, 50]);
        let small_image = Image::new(small_tensor, ImageFormat::CHW).unwrap();
        
        assert!(!predicate(&small_image));
        
        // Create a large image that should trigger the transform
        let large_image_data = vec![0.5f32; 3 * 200 * 200];
        let large_tensor = Tensor::from_vec(large_image_data, vec![3, 200, 200]);
        let large_image = Image::new(large_tensor, ImageFormat::CHW).unwrap();
        
        assert!(predicate(&large_image));
    }
    
    #[test]
    fn test_pipeline_stats() {
        let pipeline = Pipeline::<f32>::new("stats_test".to_string());
        let stats = pipeline.get_stats();
        
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
    
    #[test]
    fn test_cache_info() {
        let pipeline = Pipeline::<f32>::new("cache_test".to_string())
            .with_cache(50);
            
        let (current_size, max_size) = pipeline.cache_info();
        assert_eq!(current_size, 0);
        assert_eq!(max_size, 50);
    }
    
    #[test]
    fn test_execution_modes() {
        let modes = [
            ExecutionMode::Sequential,
            ExecutionMode::Parallel,
            ExecutionMode::Batch,
        ];
        
        for mode in modes.iter() {
            let _pipeline = Pipeline::<f32>::new("mode_test".to_string())
                .with_execution_mode(*mode);
            // Test that the pipeline can be created with different modes
            // 異なるモードでパイプラインが作成できることをテスト
        }
    }
}