// Metal backend implementation using RusTorch MPS (Metal Performance Shaders)

use super::Backend;
use anyhow::Result;
use rustorch::tensor::device::Device;
use rustorch::tensor::Tensor;

/// Metal GPU backend for macOS using Metal Performance Shaders
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    /// Create new Metal backend
    pub fn new() -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            // Use Metal Performance Shaders (MPS) device
            let device = Device::Mps;

            // TODO: Add runtime Metal availability check once implemented in rustorch
            // For now, assume Metal is available on macOS
            #[cfg(target_os = "macos")]
            {
                Ok(Self { device })
            }

            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("Metal is only available on macOS")
            }
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // Metal is available on macOS when feature is enabled
            true
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            false
        }
    }

    fn to_device(&self, mut tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        tensor.device = self.device;
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        let mut tensor = Tensor::zeros(shape);
        tensor.device = self.device;
        Ok(tensor)
    }

    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>> {
        let mut tensor = Tensor::from_vec(data, shape.to_vec());
        tensor.device = self.device;
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_backend_creation() {
        if let Ok(backend) = MetalBackend::new() {
            assert_eq!(backend.name(), "metal");
            assert!(backend.is_available());
        }
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn test_metal_tensor_operations() {
        if let Ok(backend) = MetalBackend::new() {
            let tensor = backend.zeros(&[2, 2]).unwrap();
            assert_eq!(tensor.size(), vec![2, 2]);
        }
    }
}
