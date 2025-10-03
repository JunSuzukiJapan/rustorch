// OpenCL backend implementation using RusTorch GPU features

use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;
use rustorch::tensor::device::Device;

/// OpenCL GPU backend for cross-platform GPU acceleration
pub struct OpenCLBackend {
    device: Device,
}

impl OpenCLBackend {
    /// Create new OpenCL backend
    pub fn new() -> Result<Self> {
        #[cfg(feature = "opencl")]
        {
            // Check OpenCL availability using RusTorch
            let device = Device::opencl(0);
            if !device.is_available() {
                anyhow::bail!("OpenCL device not available");
            }
            Ok(Self { device })
        }
        #[cfg(not(feature = "opencl"))]
        {
            anyhow::bail!("OpenCL feature not enabled")
        }
    }

    /// Create OpenCL backend with specific device ID
    #[cfg(feature = "opencl")]
    pub fn with_device(device_id: usize) -> Result<Self> {
        let device = Device::opencl(device_id);
        if !device.is_available() {
            anyhow::bail!("OpenCL device {} not available", device_id);
        }
        Ok(Self { device })
    }
}

impl Backend for OpenCLBackend {
    fn name(&self) -> &'static str {
        "opencl"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "opencl")]
        {
            self.device.is_available()
        }
        #[cfg(not(feature = "opencl"))]
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
    #[cfg(feature = "opencl")]
    fn test_opencl_backend_creation() {
        if let Ok(backend) = OpenCLBackend::new() {
            assert_eq!(backend.name(), "opencl");
            assert!(backend.is_available());
        }
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn test_opencl_tensor_operations() {
        if let Ok(backend) = OpenCLBackend::new() {
            let tensor = backend.zeros(&[2, 2]).unwrap();
            assert_eq!(tensor.size(), vec![2, 2]);
        }
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn test_opencl_multi_device() {
        // Try device 0 and 1 if available
        if OpenCLBackend::new().is_ok() {
            let _ = OpenCLBackend::with_device(0);
            // Device 1 might not be available, so we don't assert
            let _ = OpenCLBackend::with_device(1);
        }
    }
}
