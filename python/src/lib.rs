//! Enhanced Python bindings for RusTorch with Rust→Python communication

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError};
use std::collections::HashMap;

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
    "0.3.4".to_string()
}

/// Test creating and returning a simple tensor-like structure
#[pyfunction]
fn create_simple_tensor(data: Vec<f32>) -> Vec<f32> {
    // Just return the same data for now
    data
}

// Rust→Python Communication System

/// Python callback registry for Rust→Python communication
#[pyclass(name = "CallbackRegistry")]
pub struct PyCallbackRegistry {
    callbacks: HashMap<String, PyObject>,
}

#[pymethods]
impl PyCallbackRegistry {
    #[new]
    fn new() -> Self {
        Self {
            callbacks: HashMap::new(),
        }
    }
    
    /// Register a Python function that can be called from Rust
    fn register_callback(&mut self, name: String, callback: PyObject, py: Python) -> PyResult<()> {
        // Verify it's callable
        if !callback.as_ref(py).is_callable() {
            return Err(PyErr::new::<PyTypeError, _>(
                format!("Object '{}' is not callable", name)
            ));
        }
        
        self.callbacks.insert(name, callback);
        Ok(())
    }
    
    /// List all registered callbacks
    fn list_callbacks(&self) -> Vec<String> {
        self.callbacks.keys().cloned().collect()
    }
    
    /// Call a Python function from Rust with arguments
    fn call_python_function(&self, name: &str, args: &PyList, py: Python) -> PyResult<PyObject> {
        match self.callbacks.get(name) {
            Some(callback) => {
                let args_tuple = PyTuple::new(py, args);
                callback.call1(py, args_tuple)
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Callback '{}' not found", name)
            ))
        }
    }
    
    fn __repr__(&self) -> String {
        format!("CallbackRegistry(callbacks={})", self.callbacks.len())
    }
}

/// Initialize the callback system
#[pyfunction]
fn init_callback_system() -> PyCallbackRegistry {
    PyCallbackRegistry::new()
}

/// Example function that demonstrates calling Python from Rust
#[pyfunction]
fn call_python_from_rust(
    registry: &PyCallbackRegistry, 
    callback_name: String, 
    message: String,
    py: Python
) -> PyResult<String> {
    let args = PyList::new(py, &[message.into_py(py)]);
    
    match registry.call_python_function(&callback_name, args, py) {
        Ok(result) => {
            match result.extract::<String>(py) {
                Ok(result_str) => Ok(result_str),
                Err(_) => Ok(format!("Python function returned: {:?}", result))
            }
        }
        Err(e) => Err(e)
    }
}

/// Progress callback example
#[pyfunction]
fn progress_callback_example(
    registry: &PyCallbackRegistry,
    total_steps: usize,
    py: Python
) -> PyResult<Vec<String>> {
    let mut results = Vec::new();
    
    for step in 0..total_steps {
        std::thread::sleep(std::time::Duration::from_millis(50));
        
        let progress = (step as f64 / total_steps as f64) * 100.0;
        let args = PyList::new(py, &[
            step.into_py(py),
            total_steps.into_py(py), 
            progress.into_py(py)
        ]);
        
        if let Ok(result) = registry.call_python_function("progress", args, py) {
            if let Ok(msg) = result.extract::<String>(py) {
                results.push(msg);
            }
        }
    }
    
    let args = PyList::new(py, &[results.len().into_py(py)]);
    if let Ok(_) = registry.call_python_function("completed", args, py) {
        results.push("Operation completed".to_string());
    }
    
    Ok(results)
}

// Error Handling System

/// Custom RusTorch exception for Python
#[pyclass(extends=PyException)]
pub struct RusTorchError {
    pub message: String,
    pub error_code: i32,
}

#[pymethods]
impl RusTorchError {
    #[new]
    fn new(message: String, error_code: Option<i32>) -> Self {
        Self {
            message,
            error_code: error_code.unwrap_or(1000),
        }
    }
    
    #[getter]
    fn message(&self) -> &str {
        &self.message
    }
    
    #[getter]
    fn error_code(&self) -> i32 {
        self.error_code
    }
    
    fn __str__(&self) -> String {
        format!("[{}] {}", self.error_code, self.message)
    }
}

/// Result type for operations that can fail
#[pyclass(name = "Result")]
pub struct PyResult_ {
    success: bool,
    value: Option<PyObject>,
    error: Option<String>,
}

#[pymethods]
impl PyResult_ {
    #[new]
    fn new(success: bool, value: Option<PyObject>, error: Option<String>) -> Self {
        Self { success, value, error }
    }
    
    /// Create a successful result
    #[staticmethod]
    fn ok(value: PyObject) -> Self {
        Self {
            success: true,
            value: Some(value),
            error: None,
        }
    }
    
    /// Create an error result
    #[staticmethod]
    fn err(error: String) -> Self {
        Self {
            success: false,
            value: None,
            error: Some(error),
        }
    }
    
    #[getter]
    fn is_ok(&self) -> bool {
        self.success
    }
    
    #[getter]
    fn is_err(&self) -> bool {
        !self.success
    }
    
    /// Get the value, raising an exception if this is an error
    fn unwrap(&self, py: Python) -> PyResult<PyObject> {
        if self.success {
            Ok(self.value.as_ref().unwrap().clone_ref(py))
        } else {
            let error_msg = self.error.as_ref()
                .map(|s| s.as_str())
                .unwrap_or("Unknown error");
            Err(PyRuntimeError::new_err(error_msg.to_string()))
        }
    }
    
    /// Get the value or a default
    fn unwrap_or(&self, default: PyObject, py: Python) -> PyObject {
        if self.success {
            self.value.as_ref().unwrap().clone_ref(py)
        } else {
            default
        }
    }
    
    fn __repr__(&self) -> String {
        if self.success {
            "Result.Ok(<value>)".to_string()
        } else {
            format!("Result.Err('{}')", self.error.as_ref().unwrap_or(&"Unknown".to_string()))
        }
    }
}

/// Try-catch style error handling
#[pyfunction]
fn try_catch(operation: PyObject, error_handler: Option<PyObject>, py: Python) -> PyResult<PyObject> {
    match operation.call0(py) {
        Ok(result) => Ok(result),
        Err(e) => {
            if let Some(handler) = error_handler {
                let args = (e.value(py),);
                handler.call1(py, args)
            } else {
                Err(e)
            }
        }
    }
}

/// Enhanced Python module with Rust→Python communication
#[pymodule]
fn _rustorch_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Basic functions
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(sum_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_tensor, m)?)?;
    
    // Rust→Python communication
    m.add_class::<PyCallbackRegistry>()?;
    m.add_function(wrap_pyfunction!(init_callback_system, m)?)?;
    m.add_function(wrap_pyfunction!(call_python_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(progress_callback_example, m)?)?;
    
    // Error handling
    m.add_class::<RusTorchError>()?;
    m.add_class::<PyResult_>()?;
    m.add_function(wrap_pyfunction!(try_catch, m)?)?;
    
    m.add("__version__", "0.3.4")?;
    
    Ok(())
}