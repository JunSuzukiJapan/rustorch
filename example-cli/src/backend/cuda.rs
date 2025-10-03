// CUDA backend implementation using RusTorch GPU features

use super::Backend;
use anyhow::Result;
use rustorch::tensor::device::Device;
use rustorch::tensor::Tensor;

/// CUDA GPU backend for NVIDIA GPUs
pub struct CudaBackend {
    device: Device,
}

impl CudaBackend {
    /// Create new CUDA backend
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Check CUDA availability using RusTorch
            let device = Device::cuda(0);
            if !device.is_available() {
                anyhow::bail!("CUDA device not available");
            }
            Ok(Self { device })
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("CUDA feature not enabled")
        }
    }

    /// Create CUDA backend with specific device ID
    #[cfg(feature = "cuda")]
    pub fn with_device(device_id: usize) -> Result<Self> {
        let device = Device::cuda(device_id);
        if !device.is_available() {
            anyhow::bail!("CUDA device {} not available", device_id);
        }
        Ok(Self { device })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.device.is_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn to_device(&self, mut tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        tensor.device = self.device.clone();
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        let mut tensor = Tensor::zeros(shape);
        tensor.device = self.device.clone();
        Ok(tensor)
    }

    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>> {
        let mut tensor = Tensor::from_vec(data, shape.to_vec());
        tensor.device = self.device.clone();
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_backend_creation() {
        if let Ok(backend) = CudaBackend::new() {
            assert_eq!(backend.name(), "cuda");
            assert!(backend.is_available());
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_operations() {
        if let Ok(backend) = CudaBackend::new() {
            let tensor = backend.zeros(&[2, 2]).unwrap();
            assert_eq!(tensor.size(), vec![2, 2]);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_multi_device() {
        // Try device 0 and 1 if available
        if CudaBackend::new().is_ok() {
            let _ = CudaBackend::with_device(0);
            // Device 1 might not be available, so we don't assert
            let _ = CudaBackend::with_device(1);
        }
    }
}
