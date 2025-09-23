//! Core functionality - basic Python-Rust communication

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Basic communication functions
pub mod basic {
    use super::*;

    /// Simple function to test Rust->Python communication
    #[pyfunction]
    pub fn hello_from_rust() -> String {
        "Hello from RusTorch!".to_string()
    }

    /// Add two numbers (testing parameter passing)
    #[pyfunction]
    pub fn add_numbers(a: f32, b: f32) -> f32 {
        a + b
    }

    /// Process a list of numbers (testing list handling)
    #[pyfunction]
    pub fn sum_list(numbers: &PyList) -> PyResult<f32> {
        let mut total = 0.0;
        for item in numbers.iter() {
            total += item.extract::<f32>()?;
        }
        Ok(total)
    }

    /// Get RusTorch version
    #[pyfunction]
    pub fn get_version() -> String {
        "0.3.3".to_string()
    }

    /// Test creating and returning a simple tensor-like structure
    #[pyfunction]
    pub fn create_simple_tensor(data: Vec<f32>) -> Vec<f32> {
        data
    }
}

/// Register core functions with Python module
pub fn register_core_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(basic::hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(basic::add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(basic::sum_list, m)?)?;
    m.add_function(wrap_pyfunction!(basic::get_version, m)?)?;
    m.add_function(wrap_pyfunction!(basic::create_simple_tensor, m)?)?;
    Ok(())
}