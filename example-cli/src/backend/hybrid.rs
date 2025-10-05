use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;

#[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
use rustorch::gpu::hybrid_executor::HybridExecutor;

/// Mac Hybrid Backend (Metal + CoreML)
///
/// Uses RusTorch's HybridExecutor for intelligent routing between:
/// - CPU for small operations (< 1MB)
/// - Metal GPU for medium compute (1MB-100MB)
/// - CoreML Neural Engine for large operations (> 100MB)
pub struct HybridBackend {
    #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
    executor: &'static HybridExecutor,

    available: bool,
}

impl HybridBackend {
    pub fn new() -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
        {
            let executor = HybridExecutor::global();
            let available = true;

            Ok(Self {
                executor,
                available,
            })
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
        // HybridExecutor handles device routing internally
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
    fn test_hybrid_backend_availability() {
        let backend = HybridBackend::new().unwrap();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "Mac Hybrid (Metal + CoreML)");
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "mac-hybrid"))]
    fn test_hybrid_backend_tensor_operations() {
        let backend = HybridBackend::new().unwrap();

        // Test zeros
        let zeros = backend.zeros(&[2, 3]).unwrap();
        assert_eq!(zeros.shape(), &[2, 3]);

        // Test from_vec
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = backend.from_vec(data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        // Test to_device
        let moved = backend.to_device(tensor).unwrap();
        assert_eq!(moved.shape(), &[2, 2]);
    }
}
