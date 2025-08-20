//! Common benchmark utilities and helper functions
//! 
//! This module provides shared functionality for all RusTorch benchmarks,
//! including common test data generation, benchmark configuration, and utilities.

pub mod tensor_ops;
pub mod simd_ops;
pub mod memory_ops;

use criterion::Criterion;
use rustorch::tensor::Tensor;
use num_traits::Float;

/// Standard benchmark sizes for consistent testing across all benchmarks
pub const BENCHMARK_SIZES: &[(usize, &str)] = &[
    (1024, "1K elements"),
    (4096, "4K elements"), 
    (16384, "16K elements"),
    (65536, "64K elements"),
    (262144, "256K elements"),
];

/// Matrix benchmark sizes for 2D operations
pub const MATRIX_SIZES: &[(usize, &str)] = &[
    (100, "Small (10K elements)"),
    (500, "Medium (250K elements)"),
    (1000, "Large (1M elements)"),
    (2000, "XLarge (4M elements)"),
];

/// Small sizes for quick testing
pub const QUICK_SIZES: &[(usize, &str)] = &[
    (64, "Tiny (64 elements)"),
    (256, "Small (256 elements)"),
    (1024, "Medium (1K elements)"),
];

/// Generate test tensors with specified shape and fill value
pub fn create_test_tensors<T>(shape: &[usize], _fill_value: T) -> (Tensor<T>, Tensor<T>)
where
    T: Float + Clone + Default + 'static,
{
    // Create tensors filled with ones (ignoring fill_value for simplicity)
    let tensor1 = Tensor::<T>::ones(shape);
    let tensor2 = Tensor::<T>::ones(shape);
    
    (tensor1, tensor2)
}

/// Create a benchmark group with standard configuration
pub fn create_benchmark_group<'a>(c: &'a mut Criterion, name: &str) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(5));
    group
}

/// Helper macro for creating consistent benchmark functions
#[macro_export]
macro_rules! benchmark_operation {
    ($group:expr, $name:expr, $size_info:expr, $operation:expr) => {
        $group.bench_with_input(
            BenchmarkId::new($name, $size_info.1),
            $size_info,
            |b, (size, _)| {
                b.iter(|| {
                    let result = $operation(*size);
                    criterion::black_box(result)
                });
            },
        );
    };
}

/// Calculate elements count for throughput measurement
pub fn elements_count(shape: &[usize]) -> u64 {
    shape.iter().product::<usize>() as u64
}

/// Standard benchmark configuration for tensor operations
pub struct BenchmarkConfig {
    pub sample_size: usize,
    pub measurement_time_secs: u64,
    pub warm_up_time_secs: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sample_size: 100,
            measurement_time_secs: 5,
            warm_up_time_secs: 3,
        }
    }
}

impl BenchmarkConfig {
    pub fn quick() -> Self {
        Self {
            sample_size: 50,
            measurement_time_secs: 2,
            warm_up_time_secs: 1,
        }
    }
    
    pub fn thorough() -> Self {
        Self {
            sample_size: 200,
            measurement_time_secs: 10,
            warm_up_time_secs: 5,
        }
    }
}
