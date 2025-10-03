// Metal backend implementation using RusTorch GPU features

use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;
use rustorch::tensor::device::Device;

/// Metal GPU backend for macOS
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    /// Create new Metal backend
    pub fn new() -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            // Check Metal availability using RusTorch
            let device = Device::metal();
            if !device.is_available() {
                anyhow::bail!("Metal device not available");
            }
            Ok(Self { device })
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("Metal feature not enabled")
        }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.device.is_available()
        }
        #[cfg(not(feature = "metal"))]
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
    #[cfg(feature = "metal")]
    fn test_metal_backend_creation() {
        if let Ok(backend) = MetalBackend::new() {
            assert_eq!(backend.name(), "metal");
            assert!(backend.is_available());
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_tensor_operations() {
        if let Ok(backend) = MetalBackend::new() {
            let tensor = backend.zeros(&[2, 2]).unwrap();
            assert_eq!(tensor.size(), vec![2, 2]);
        }
    }
}
