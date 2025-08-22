//! Comprehensive error handling and exception integration

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyRuntimeError};

/// Error types and codes
pub mod types {
    /// Error categories with associated codes
    pub enum ErrorCategory {
        General = 1000,
        Tensor = 2000,
        Shape = 2001,
        Device = 2002,
        Callback = 3000,
        Runtime = 4000,
    }
}

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
            error_code: error_code.unwrap_or(types::ErrorCategory::General as i32),
        }
    }
    
    /// Create error with specific category
    #[staticmethod]
    fn tensor_error(message: String) -> Self {
        Self {
            message,
            error_code: types::ErrorCategory::Tensor as i32,
        }
    }
    
    #[staticmethod]
    fn shape_error(message: String) -> Self {
        Self {
            message,
            error_code: types::ErrorCategory::Shape as i32,
        }
    }
    
    #[staticmethod]
    fn device_error(message: String) -> Self {
        Self {
            message,
            error_code: types::ErrorCategory::Device as i32,
        }
    }
    
    #[staticmethod]
    fn callback_error(message: String) -> Self {
        Self {
            message,
            error_code: types::ErrorCategory::Callback as i32,
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
    
    #[getter]
    fn category(&self) -> String {
        match self.error_code {
            1000..=1999 => "General".to_string(),
            2000..=2999 => "Tensor".to_string(),
            3000..=3999 => "Callback".to_string(),
            4000..=4999 => "Runtime".to_string(),
            _ => "Unknown".to_string(),
        }
    }
    
    fn __str__(&self) -> String {
        format!("[{}:{}] {}", self.category(), self.error_code, self.message)
    }
    
    fn __repr__(&self) -> String {
        format!("RusTorchError(message='{}', code={}, category='{}')", 
                self.message, self.error_code, self.category())
    }
}

/// Result type for operations that can fail
#[pyclass(name = "Result")]
#[derive(Clone)]
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
    
    /// Get the error message
    fn error_message(&self) -> Option<String> {
        self.error.clone()
    }
    
    /// Chain operations (monadic bind)
    fn and_then(&self, func: PyObject, py: Python) -> PyResult<PyResult_> {
        if self.success {
            match func.call1(py, (self.value.as_ref().unwrap().clone_ref(py),)) {
                Ok(result) => {
                    if let Ok(result_obj) = result.extract::<PyResult_>(py) {
                        Ok(result_obj)
                    } else {
                        Ok(PyResult_ {
                            success: true,
                            value: Some(result),
                            error: None,
                        })
                    }
                }
                Err(e) => Ok(PyResult_ {
                    success: false,
                    value: None,
                    error: Some(format!("Chain operation failed: {}", e)),
                }),
            }
        } else {
            Ok(PyResult_ {
                success: false,
                value: None,
                error: self.error.clone(),
            })
        }
    }
    
    fn __repr__(&self) -> String {
        if self.success {
            "Result.Ok(<value>)".to_string()
        } else {
            format!("Result.Err('{}')", 
                   self.error.as_ref().unwrap_or(&"Unknown".to_string()))
        }
    }
}

/// Error handling utilities
pub mod functions {
    use super::*;

    /// Try-catch style error handling
    #[pyfunction]
    pub fn try_catch(operation: PyObject, error_handler: Option<PyObject>, py: Python) -> PyResult<PyObject> {
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

    /// Safe execution wrapper
    #[pyfunction]
    pub fn safe_execute(operation: PyObject, py: Python) -> PyResult_ {
        match operation.call0(py) {
            Ok(result) => PyResult_ {
                success: true,
                value: Some(result),
                error: None,
            },
            Err(e) => PyResult_ {
                success: false,
                value: None,
                error: Some(format!("Operation failed: {}", e)),
            },
        }
    }
}

/// Register error handling system with Python module
pub fn register_error_system(m: &PyModule) -> PyResult<()> {
    m.add_class::<RusTorchError>()?;
    m.add_class::<PyResult_>()?;
    m.add_function(wrap_pyfunction!(functions::try_catch, m)?)?;
    m.add_function(wrap_pyfunction!(functions::safe_execute, m)?)?;
    Ok(())
}