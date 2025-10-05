use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;

/// Mac Hybrid Backend (Metal + CoreML)
pub struct HybridBackend {
    available: bool,
}

impl HybridBackend {
    pub fn new() -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
        {
            Ok(Self { available: true })
        }
        #[cfg(not(all(target_os = "macos", feature = "mac-hybrid")))]
        {
            anyhow::bail!("Mac Hybrid backend requires macOS and mac-hybrid feature")
        }
    }
}

impl Backend for HybridBackend {
    fn name(&self) -> &'static str {
        "Mac Hybrid (Metal + CoreML)"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn to_device(&self, tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        // For now, just return the CPU tensor
        // TODO: Implement actual device transfer when Metal/CoreML integration is ready
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        Ok(Tensor::zeros(shape))
    }

    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>> {
        Ok(Tensor::from_vec(data, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
    fn test_hybrid_backend_creation() {
        let backend = HybridBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
    fn test_hybrid_backend_name() {
        let backend = HybridBackend::new().unwrap();
        assert_eq!(backend.name(), "Mac Hybrid (Metal + CoreML)");
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
    fn test_hybrid_backend_is_available() {
        let backend = HybridBackend::new().unwrap();
        assert!(backend.is_available());
    }
}
