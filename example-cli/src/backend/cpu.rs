use anyhow::Result;
use rustorch::tensor::Tensor;

use super::Backend;

/// CPU backend implementation using RusTorch
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        tracing::debug!("Initializing CPU backend");
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn to_device(&self, tensor: Tensor<f64>) -> Result<Tensor<f64>> {
        // On CPU, tensor is already on the correct device
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        Ok(Tensor::zeros(shape))
    }

    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>> {
        Ok(Tensor::from_vec(data, shape.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_new() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_cpu_zeros() {
        let backend = CpuBackend::new();
        let tensor = backend.zeros(&[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_cpu_from_vec() {
        let backend = CpuBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = backend.from_vec(data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_cpu_to_device() {
        let backend = CpuBackend::new();
        let tensor = Tensor::<f64>::zeros(&[2, 2]);
        let result = backend.to_device(tensor).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }
}
