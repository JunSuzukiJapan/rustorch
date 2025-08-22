//! Minimal Python bindings for RusTorch - basic call interface only

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Simple function to test Rust->Python communication
#[pyfunction]
fn hello_from_rust() -> String {
    "Hello from RusTorch!".to_string()
}

/// Add two numbers (testing parameter passing)
#[pyfunction]
fn add_numbers(a: f32, b: f32) -> f32 {
    a + b
}

/// Process a list of numbers (testing list handling)
#[pyfunction]
fn sum_list(numbers: &PyList) -> PyResult<f32> {
    let mut total = 0.0;
    for item in numbers.iter() {
        total += item.extract::<f32>()?;
    }
    Ok(total)
}

/// Get RusTorch version
#[pyfunction]
fn get_version() -> String {
    "0.3.3".to_string()
}

/// Test creating and returning a simple tensor-like structure
#[pyfunction]
fn create_simple_tensor(data: Vec<f32>) -> Vec<f32> {
    // Just return the same data for now
    data
}

/// Minimal Python module
#[pymodule]
fn _rustorch_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(sum_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_tensor, m)?)?;
    
    m.add("__version__", "0.3.3")?;
    
    Ok(())
}