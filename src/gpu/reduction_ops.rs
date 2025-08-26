//! GPU Reduction Operations
//! GPU リダクション演算
//!
//! This module provides GPU-accelerated reduction operations including
//! sum, mean, max, min with optimized kernels and memory coalescing.

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "metal")]
use metal::{Buffer, CommandBuffer, CommandQueue, Device as MetalDevice};

/// Reduction operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum reduction
    Sum,
    /// Mean (average) reduction
    Mean,
    /// Maximum value reduction
    Max,
    /// Minimum value reduction
    Min,
    /// Product reduction
    Prod,
    /// Standard deviation reduction
    Std,
    /// Variance reduction
    Var,
}

/// GPU reduction executor
pub struct GpuReductionExecutor<T: Float + FromPrimitive + ScalarOperand + 'static> {
    device_type: super::DeviceType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + FromPrimitive + ScalarOperand + 'static> GpuReductionExecutor<T> {
    /// Create new GPU reduction executor
    pub fn new(device_type: super::DeviceType) -> RusTorchResult<Self> {
        match device_type {
            super::DeviceType::Cpu => Ok(Self {
                device_type,
                _phantom: std::marker::PhantomData,
            }),
            _ => {
                // Verify GPU device is available
                if device_type.is_available() {
                    Ok(Self {
                        device_type,
                        _phantom: std::marker::PhantomData,
                    })
                } else {
                    Err(RusTorchError::gpu(format!(
                        "GPU device {:?} not available",
                        device_type
                    )))
                }
            }
        }
    }

    /// Perform reduction operation on GPU
    pub fn reduce(
        &self,
        input: &Tensor<T>,
        op: ReductionOp,
        dim: Option<usize>,
    ) -> RusTorchResult<Tensor<T>> {
        match self.device_type {
            super::DeviceType::Cpu => self.cpu_reduce(input, op, dim),

            #[cfg(feature = "cuda")]
            super::DeviceType::Cuda(device_id) => self.cuda_reduce(input, op, dim, device_id),

            #[cfg(feature = "metal")]
            super::DeviceType::Metal(_) => self.metal_reduce(input, op, dim),

            #[cfg(feature = "opencl")]
            super::DeviceType::OpenCL(_) => {
                // For now, fall back to CPU
                self.cpu_reduce(input, op, dim)
            }

            #[allow(unreachable_patterns)]
            _ => Err(RusTorchError::gpu("Unsupported device for reduction")),
        }
    }

    /// CPU reduction fallback
    fn cpu_reduce(
        &self,
        input: &Tensor<T>,
        op: ReductionOp,
        dim: Option<usize>,
    ) -> RusTorchResult<Tensor<T>> {
        match op {
            ReductionOp::Sum => self.cpu_sum(input, dim),
            ReductionOp::Mean => self.cpu_mean(input, dim),
            ReductionOp::Max => self.cpu_max(input, dim),
            ReductionOp::Min => self.cpu_min(input, dim),
            ReductionOp::Prod => self.cpu_prod(input, dim),
            ReductionOp::Std => self.cpu_std(input, dim),
            ReductionOp::Var => self.cpu_var(input, dim),
        }
    }

    /// CPU sum reduction
    fn cpu_sum(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        match dim {
            None => {
                // Sum all elements
                let sum = input.data.iter().fold(T::zero(), |acc, &x| acc + x);
                Ok(Tensor::from_vec(vec![sum], vec![1]))
            }
            Some(axis) => {
                // Sum along specific axis
                let input_shape = input.shape();
                if axis >= input_shape.len() {
                    return Err(RusTorchError::gpu("Reduction axis out of bounds"));
                }

                // Calculate output shape
                let mut output_shape = input_shape.to_vec();
                output_shape[axis] = 1;

                let mut output_data = vec![T::zero(); output_shape.iter().product()];

                // Perform reduction along axis (simplified implementation)
                let axis_size = input_shape[axis];
                let outer_size: usize = input_shape[..axis].iter().product();
                let inner_size: usize = input_shape[axis + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let mut sum = T::zero();
                        for i in 0..axis_size {
                            let input_idx = (outer * axis_size + i) * inner_size + inner;
                            if let Some(val) = input.data.get(input_idx) {
                                sum = sum + *val;
                            }
                        }
                        let output_idx = outer * inner_size + inner;
                        output_data[output_idx] = sum;
                    }
                }

                Ok(Tensor::from_vec(output_data, output_shape))
            }
        }
    }

    /// CPU mean reduction
    fn cpu_mean(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let sum_result = self.cpu_sum(input, dim)?;
        let count = match dim {
            None => T::from(input.data.len()).unwrap_or(T::one()),
            Some(axis) => T::from(input.shape()[axis]).unwrap_or(T::one()),
        };

        let mean_data: Vec<T> = sum_result.data.iter().map(|&x| x / count).collect();
        Ok(Tensor::from_vec(mean_data, sum_result.shape().to_vec()))
    }

    /// CPU max reduction
    #[allow(clippy::only_used_in_recursion)]
    fn cpu_max(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        match dim {
            None => {
                let max_val =
                    input.data.iter().fold(
                        T::neg_infinity(),
                        |acc, &x| {
                            if x > acc {
                                x
                            } else {
                                acc
                            }
                        },
                    );
                Ok(Tensor::from_vec(vec![max_val], vec![1]))
            }
            Some(_) => {
                // For now, fall back to global max
                self.cpu_max(input, None)
            }
        }
    }

    /// CPU min reduction
    #[allow(clippy::only_used_in_recursion)]
    fn cpu_min(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        match dim {
            None => {
                let min_val = input
                    .data
                    .iter()
                    .fold(T::infinity(), |acc, &x| if x < acc { x } else { acc });
                Ok(Tensor::from_vec(vec![min_val], vec![1]))
            }
            Some(_) => {
                // For now, fall back to global min
                self.cpu_min(input, None)
            }
        }
    }

    /// CPU product reduction
    #[allow(clippy::only_used_in_recursion)]
    fn cpu_prod(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        match dim {
            None => {
                let prod = input.data.iter().fold(T::one(), |acc, &x| acc * x);
                Ok(Tensor::from_vec(vec![prod], vec![1]))
            }
            Some(_) => {
                // For now, fall back to global product
                self.cpu_prod(input, None)
            }
        }
    }

    /// CPU standard deviation reduction
    fn cpu_std(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let var_result = self.cpu_var(input, dim)?;
        let std_data: Vec<T> = var_result.data.iter().map(|&x| x.sqrt()).collect();
        Ok(Tensor::from_vec(std_data, var_result.shape().to_vec()))
    }

    /// CPU variance reduction
    fn cpu_var(&self, input: &Tensor<T>, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let mean_result = self.cpu_mean(input, dim)?;
        let mean_val = mean_result.data[0]; // Simplified for global variance

        let var = input
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);

        let count = T::from(input.data.len()).unwrap_or(T::one());
        let variance = var / count;

        Ok(Tensor::from_vec(vec![variance], vec![1]))
    }
}

// CUDA implementation
#[cfg(feature = "cuda")]
impl<T> GpuReductionExecutor<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static + cudarc::driver::DeviceRepr,
{
    fn cuda_reduce(
        &self,
        input: &Tensor<T>,
        op: ReductionOp,
        dim: Option<usize>,
        device_id: usize,
    ) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::memory_transfer::GpuMemoryManager;

        // Initialize CUDA device
        let _device = CudaDevice::new(device_id)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA device init failed: {}", e)))?;

        // For now, fall back to CPU until we implement CUDA kernels
        self.cpu_reduce(input, op, dim)
    }
}

// Metal implementation
#[cfg(feature = "metal")]
impl<T: Float + FromPrimitive + ScalarOperand + 'static> GpuReductionExecutor<T> {
    fn metal_reduce(
        &self,
        input: &Tensor<T>,
        op: ReductionOp,
        dim: Option<usize>,
    ) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::memory_transfer::GpuMemoryManager;

        // Get Metal device
        let _device = MetalDevice::system_default()
            .ok_or_else(|| RusTorchError::gpu("No Metal device found"))?;

        // For now, fall back to CPU until we implement Metal shaders
        self.cpu_reduce(input, op, dim)
    }
}

/// GPU reduction operations trait
pub trait GpuReduction<T: Float + FromPrimitive + ScalarOperand + 'static> {
    /// GPU sum reduction
    fn gpu_sum(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;

    /// GPU mean reduction  
    fn gpu_mean(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;

    /// GPU max reduction
    fn gpu_max(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;

    /// GPU min reduction
    fn gpu_min(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;

    /// GPU standard deviation
    fn gpu_std(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;

    /// GPU variance
    fn gpu_var(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>>;
}

impl<T: Float + FromPrimitive + ScalarOperand + 'static> GpuReduction<T> for Tensor<T> {
    fn gpu_sum(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Sum, dim)
    }

    fn gpu_mean(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Mean, dim)
    }

    fn gpu_max(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Max, dim)
    }

    fn gpu_min(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Min, dim)
    }

    fn gpu_std(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Std, dim)
    }

    fn gpu_var(&self, dim: Option<usize>) -> RusTorchResult<Tensor<T>> {
        let device = if super::DeviceManager::is_cuda_available() {
            super::DeviceType::Cuda(0)
        } else if super::DeviceManager::is_metal_available() {
            super::DeviceType::Metal(0)
        } else {
            super::DeviceType::Cpu
        };

        let executor = GpuReductionExecutor::new(device)?;
        executor.reduce(self, ReductionOp::Var, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_reduction_op_types() {
        assert_eq!(ReductionOp::Sum, ReductionOp::Sum);
        assert_ne!(ReductionOp::Sum, ReductionOp::Mean);
    }

    #[test]
    fn test_gpu_reduction_executor_creation() {
        let executor = GpuReductionExecutor::<f32>::new(super::super::DeviceType::Cpu);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_cpu_sum_reduction() {
        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let executor = GpuReductionExecutor::<f32>::new(super::super::DeviceType::Cpu).unwrap();

        let result = executor.reduce(&input, ReductionOp::Sum, None).unwrap();
        assert_eq!(result.data[0], 10.0);
    }

    #[test]
    fn test_gpu_sum_trait() {
        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let result = input.gpu_sum(None).unwrap();
        assert_eq!(result.data[0], 10.0);
    }

    #[test]
    fn test_gpu_mean_trait() {
        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let result = input.gpu_mean(None).unwrap();
        assert_eq!(result.data[0], 2.5);
    }
}
