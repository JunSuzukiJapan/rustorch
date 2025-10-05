pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "mac-hybrid")]
pub mod hybrid;

#[cfg(feature = "hybrid-f32")]
pub mod hybrid_f32;

use anyhow::Result;
use rustorch::tensor::Tensor;

/// Backend trait for tensor operations
pub trait Backend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &'static str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Move tensor to this backend
    fn to_device(&self, tensor: Tensor<f64>) -> Result<Tensor<f64>>;

    /// Allocate tensor on this backend
    fn zeros(&self, shape: &[usize]) -> Result<Tensor<f64>>;

    /// Create tensor from data
    fn from_vec(&self, data: Vec<f64>, shape: &[usize]) -> Result<Tensor<f64>>;
}

/// Get backend by name
pub fn get_backend(name: &str) -> Result<Box<dyn Backend>> {
    match name.to_lowercase().as_str() {
        "cpu" => Ok(Box::new(cpu::CpuBackend::new())),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Box::new(cuda::CudaBackend::new()?)),
        #[cfg(feature = "metal")]
        "metal" => Ok(Box::new(metal::MetalBackend::new()?)),
        #[cfg(feature = "opencl")]
        "opencl" => Ok(Box::new(opencl::OpenCLBackend::new()?)),
        #[cfg(feature = "mac-hybrid")]
        "hybrid" => Ok(Box::new(hybrid::HybridBackend::new()?)),
        #[cfg(feature = "hybrid-f32")]
        "hybrid-f32" => Ok(Box::new(hybrid_f32::HybridF32Backend::new()?)),
        _ => anyhow::bail!("Unknown backend: {}", name),
    }
}

/// Auto-detect best available backend
pub fn auto_backend() -> Result<Box<dyn Backend>> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if let Ok(backend) = get_backend("metal") {
            if backend.is_available() {
                return Ok(backend);
            }
        }
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        if let Ok(backend) = get_backend("cuda") {
            if backend.is_available() {
                return Ok(backend);
            }
        }
    }

    #[cfg(feature = "opencl")]
    {
        if let Ok(backend) = get_backend("opencl") {
            if backend.is_available() {
                return Ok(backend);
            }
        }
    }

    // Fallback to CPU
    Ok(Box::new(cpu::CpuBackend::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = cpu::CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_get_backend_cpu() {
        let backend = get_backend("cpu").unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_auto_backend() {
        let backend = auto_backend().unwrap();
        assert!(backend.is_available());
    }

    #[test]
    fn test_unknown_backend() {
        let result = get_backend("unknown");
        assert!(result.is_err());
    }
}
