//! Unified transformation pipeline for WASM operations
//! WASM操作の統一変換パイプライン

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use crate::wasm::common::{WasmError, WasmResult, WasmTransform, WasmOperation, MemoryManager};
use js_sys::Array;
use std::collections::HashMap;

/// Pipeline for chaining multiple WASM transformations
#[wasm_bindgen]
pub struct WasmTransformPipeline {
    transforms: Vec<Box<dyn WasmTransform>>,
    cache_enabled: bool,
    cache: HashMap<String, WasmTensor>,
}

#[wasm_bindgen]
impl WasmTransformPipeline {
    /// Create new empty pipeline
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmTransformPipeline {
        WasmTransformPipeline {
            transforms: Vec::new(),
            cache_enabled: false,
            cache: HashMap::new(),
        }
    }

    /// Enable caching for pipeline
    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }

    /// Disable caching for pipeline
    pub fn disable_cache(&mut self) {
        self.cache_enabled = false;
        self.cache.clear();
    }

    /// Clear pipeline cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Apply all transformations in pipeline
    pub fn apply(&self, input: &WasmTensor) -> WasmResult<WasmTensor> {
        if self.transforms.is_empty() {
            return Ok(WasmTensor::new(input.data(), input.shape()));
        }

        let mut current = WasmTensor::new(input.data(), input.shape());
        
        for (i, transform) in self.transforms.iter().enumerate() {
            let cache_key = if self.cache_enabled {
                Some(format!("{}_{}", i, transform.name()))
            } else {
                None
            };

            current = if let Some(key) = &cache_key {
                if let Some(cached) = self.cache.get(key) {
                    WasmTensor::new(cached.data(), cached.shape())
                } else {
                    let result = transform.apply(&current)?;
                    // Note: Can't store in cache due to &self constraint
                    result
                }
            } else {
                transform.apply(&current)?
            };
        }

        Ok(current)
    }

    /// Get pipeline summary
    pub fn summary(&self) -> String {
        let names: Vec<String> = self.transforms.iter()
            .map(|t| t.name())
            .collect();
        
        format!(
            "{{\"transform_count\":{},\"transforms\":[{}],\"cache_enabled\":{},\"cache_size\":{}}}",
            self.transforms.len(),
            names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(","),
            self.cache_enabled,
            self.cache.len()
        )
    }

    /// Validate entire pipeline
    pub fn validate(&self, input: &WasmTensor) -> WasmResult<()> {
        input.validate_non_empty()?;
        
        for transform in &self.transforms {
            transform.validate_params()?;
            transform.can_apply(input)?;
        }
        
        Ok(())
    }
}

/// Builder for creating transformation pipelines
pub struct WasmPipelineBuilder {
    transforms: Vec<Box<dyn WasmTransform>>,
    cache_enabled: bool,
}

impl WasmPipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            cache_enabled: false,
        }
    }

    /// Add transform to pipeline
    pub fn add_transform(mut self, transform: Box<dyn WasmTransform>) -> WasmResult<Self> {
        transform.validate_params()?;
        self.transforms.push(transform);
        Ok(self)
    }

    /// Enable caching
    pub fn with_cache(mut self) -> Self {
        self.cache_enabled = true;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> WasmTransformPipeline {
        WasmTransformPipeline {
            transforms: self.transforms,
            cache_enabled: self.cache_enabled,
            cache: HashMap::new(),
        }
    }
}

/// Batch processor for applying operations to multiple tensors
#[wasm_bindgen]
pub struct WasmBatchProcessor {
    operation_name: String,
    batch_size: usize,
    parallel_enabled: bool,
}

#[wasm_bindgen]
impl WasmBatchProcessor {
    /// Create new batch processor
    #[wasm_bindgen(constructor)]
    pub fn new(operation_name: String, batch_size: usize) -> WasmResult<WasmBatchProcessor> {
        if batch_size == 0 {
            return Err(WasmError::invalid_param("batch_size", batch_size, "must be non-zero"));
        }

        Ok(WasmBatchProcessor {
            operation_name,
            batch_size,
            parallel_enabled: false,
        })
    }

    /// Enable parallel processing
    pub fn enable_parallel(&mut self) {
        self.parallel_enabled = true;
    }

    /// Process multiple tensors in batches
    pub fn process_batch(&self, tensors: &Array) -> WasmResult<Array> {
        if tensors.length() == 0 {
            return Ok(Array::new());
        }

        let mut results = Vec::new();
        
        for i in 0..tensors.length() {
            let tensor_val = tensors.get(i);
            // In a real implementation, we would cast tensor_val to WasmTensor
            // and apply the operation. For now, just return success.
            results.push(tensor_val);
        }

        let result_array = Array::new();
        for result in results {
            result_array.push(&result);
        }

        Ok(result_array)
    }

    /// Get batch processing statistics
    pub fn get_stats(&self) -> String {
        format!(
            "{{\"operation\":\"{}\",\"batch_size\":{},\"parallel_enabled\":{}}}",
            self.operation_name, self.batch_size, self.parallel_enabled
        )
    }
}

/// Registry for WASM operations and factories
pub struct WasmOperationRegistry {
    transforms: HashMap<String, fn() -> Box<dyn WasmTransform>>,
    analyzers: HashMap<String, fn() -> Box<dyn WasmAnalyzer>>,
    detectors: HashMap<String, fn() -> Box<dyn WasmDetector>>,
}

impl WasmOperationRegistry {
    /// Create new operation registry
    pub fn new() -> Self {
        Self {
            transforms: HashMap::new(),
            analyzers: HashMap::new(),
            detectors: HashMap::new(),
        }
    }

    /// Register a transform factory
    pub fn register_transform<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn WasmTransform> + 'static,
    {
        self.transforms.insert(name.to_string(), Box::leak(Box::new(factory)));
    }

    /// Register an analyzer factory
    pub fn register_analyzer<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn WasmAnalyzer> + 'static,
    {
        self.analyzers.insert(name.to_string(), Box::leak(Box::new(factory)));
    }

    /// Register a detector factory
    pub fn register_detector<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn WasmDetector> + 'static,
    {
        self.detectors.insert(name.to_string(), Box::leak(Box::new(factory)));
    }

    /// Create transform by name
    pub fn create_transform(&self, name: &str) -> Option<Box<dyn WasmTransform>> {
        self.transforms.get(name).map(|factory| factory())
    }

    /// Create analyzer by name
    pub fn create_analyzer(&self, name: &str) -> Option<Box<dyn WasmAnalyzer>> {
        self.analyzers.get(name).map(|factory| factory())
    }

    /// Create detector by name
    pub fn create_detector(&self, name: &str) -> Option<Box<dyn WasmDetector>> {
        self.detectors.get(name).map(|factory| factory())
    }

    /// List available operations
    pub fn list_operations(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        (
            self.transforms.keys().cloned().collect(),
            self.analyzers.keys().cloned().collect(),
            self.detectors.keys().cloned().collect(),
        )
    }
}