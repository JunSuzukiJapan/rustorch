//! Rust→Python callback system

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::exceptions::PyTypeError;
use std::collections::HashMap;

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
    
    /// Check if a callback is registered
    fn has_callback(&self, name: &str) -> bool {
        self.callbacks.contains_key(name)
    }
    
    /// Remove a callback
    fn remove_callback(&mut self, name: &str) -> bool {
        self.callbacks.remove(name).is_some()
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
    
    /// Call Python function with error handling (returns None on error)
    fn safe_call_python_function(&self, name: &str, args: &PyList, py: Python) -> PyResult<Option<PyObject>> {
        match self.call_python_function(name, args, py) {
            Ok(result) => Ok(Some(result)),
            Err(_) => Ok(None) // Suppress errors
        }
    }
    
    fn __repr__(&self) -> String {
        format!("CallbackRegistry(callbacks={})", self.callbacks.len())
    }
}

/// Callback-related functions
pub mod functions {
    use super::*;

    /// Initialize the callback system
    #[pyfunction]
    pub fn init_callback_system() -> PyCallbackRegistry {
        PyCallbackRegistry::new()
    }

    /// Example function that demonstrates calling Python from Rust
    #[pyfunction]
    pub fn call_python_from_rust(
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
    pub fn progress_callback_example(
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
            
            if let Ok(Some(result)) = registry.safe_call_python_function("progress", args, py) {
                if let Ok(msg) = result.extract::<String>(py) {
                    results.push(msg);
                }
            }
        }
        
        let args = PyList::new(py, &[results.len().into_py(py)]);
        if let Ok(Some(_)) = registry.safe_call_python_function("completed", args, py) {
            results.push("Operation completed".to_string());
        }
        
        Ok(results)
    }
}

/// Register callback system with Python module
pub fn register_callback_system(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCallbackRegistry>()?;
    m.add_function(wrap_pyfunction!(functions::init_callback_system, m)?)?;
    m.add_function(wrap_pyfunction!(functions::call_python_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(functions::progress_callback_example, m)?)?;
    Ok(())
}