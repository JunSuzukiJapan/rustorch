//! GPU Memory Buffer Abstraction
//! GPUメモリバッファの抽象化

use crate::error::RusTorchError;
use num_traits::Float;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice};

#[cfg(feature = "metal")]
use metal::{Buffer, Device as MetalDeviceType};

#[cfg(feature = "opencl")]
use opencl3::{context::Context as CLContext, memory::Buffer as CLBuffer};

/// GPU memory buffer abstraction
/// GPUメモリバッファの抽象化
pub enum GpuBuffer<T> {
    #[cfg(feature = "cuda")]
    Cuda {
        data: Arc<CudaSlice<T>>,
        device: Arc<CudarcDevice>,
    },

    #[cfg(feature = "metal")]
    /// Metal GPU buffer with device reference
    /// デバイス参照付きMetal GPUバッファ
    Metal {
        /// Metal buffer for GPU memory
        /// GPU メモリ用のMetalバッファ
        buffer: Arc<Buffer>,
        /// Metal device reference
        /// Metalデバイスの参照
        device: Arc<MetalDeviceType>,
    },

    #[cfg(feature = "opencl")]
    OpenCL {
        buffer: Arc<CLBuffer<T>>,
        context: Arc<CLContext>,
    },

    /// CPU fallback
    Cpu(Arc<Vec<T>>),
}

impl<T> GpuBuffer<T> {
    /// Check if buffer is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self, GpuBuffer::Cpu(_))
    }

    /// Check if buffer is on CUDA device
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, GpuBuffer::Cuda { .. })
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Check if buffer is on Metal device
    pub fn is_metal(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            matches!(self, GpuBuffer::Metal { .. })
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    /// Check if buffer is on OpenCL device
    pub fn is_opencl(&self) -> bool {
        #[cfg(feature = "opencl")]
        {
            matches!(self, GpuBuffer::OpenCL { .. })
        }
        #[cfg(not(feature = "opencl"))]
        {
            false
        }
    }

    /// Get buffer size (number of elements)
    pub fn len(&self) -> usize {
        match self {
            GpuBuffer::Cpu(data) => data.len(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, .. } => data.len(),
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, .. } => buffer.length() as usize / std::mem::size_of::<T>(),
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, .. } => buffer.size() / std::mem::size_of::<T>(),
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone> Clone for GpuBuffer<T> {
    fn clone(&self) -> Self {
        match self {
            GpuBuffer::Cpu(data) => GpuBuffer::Cpu(data.clone()),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device } => GpuBuffer::Cuda {
                data: data.clone(),
                device: device.clone(),
            },
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device } => GpuBuffer::Metal {
                buffer: buffer.clone(),
                device: device.clone(),
            },
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => GpuBuffer::OpenCL {
                buffer: buffer.clone(),
                context: context.clone(),
            },
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBuffer::Cpu(data) => f
                .debug_struct("GpuBuffer::Cpu")
                .field("len", &data.len())
                .finish(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device } => f
                .debug_struct("GpuBuffer::Cuda")
                .field("len", &data.len())
                .field("device_id", &device.device_id())
                .finish(),
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, .. } => f
                .debug_struct("GpuBuffer::Metal")
                .field(
                    "len",
                    &(buffer.length() as usize / std::mem::size_of::<T>()),
                )
                .finish(),
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, .. } => f
                .debug_struct("GpuBuffer::OpenCL")
                .field("len", &(buffer.size() / std::mem::size_of::<T>()))
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = GpuBuffer::Cpu(Arc::new(data));

        assert!(buffer.is_cpu());
        assert!(!buffer.is_cuda());
        assert!(!buffer.is_metal());
        assert!(!buffer.is_opencl());
        assert_eq!(buffer.len(), 4);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_empty_buffer() {
        let buffer = GpuBuffer::Cpu(Arc::new(Vec::<f32>::new()));
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }
}
