use super::Backend;
use anyhow::Result;
use rustorch::tensor::Tensor;

/// Hybrid F32 Backend
pub struct HybridF32Backend {
    available: bool,
}

impl HybridF32Backend {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "hybrid-f32")]
        {
            Ok(Self { available: true })
        }
        #[cfg(not(feature = "hybrid-f32"))]
        {
            anyhow::bail!("Hybrid F32 backend requires hybrid-f32 feature")
        }
    }
}

impl Backend for HybridF32Backend {
    fn name(&self) -> &'static str {
        "Hybrid F32"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn to_device(&self, tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        // For now, just return the CPU tensor
        // TODO: Implement actual F32 hybrid operations when ready
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
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_creation() {
        let backend = HybridF32Backend::new();
        assert!(backend.is_ok());
    }

    #[test]
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_name() {
        let backend = HybridF32Backend::new().unwrap();
        assert_eq!(backend.name(), "Hybrid F32");
    }

    #[test]
    #[cfg(feature = "hybrid-f32")]
    fn test_hybrid_f32_backend_is_available() {
        let backend = HybridF32Backend::new().unwrap();
        assert!(backend.is_available());
    }
}
