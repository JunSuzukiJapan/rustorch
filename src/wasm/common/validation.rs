//! Validation utilities for WASM tensors
//! WASMテンサー用検証ユーティリティ

use crate::wasm::common::error::{WasmError, WasmResult};
use crate::wasm::tensor::WasmTensor;
use wasm_bindgen::prelude::*;

/// Trait for tensor validation
pub trait WasmValidation {
    /// Validate tensor is at least 2D
    fn validate_2d(&self) -> WasmResult<()>;

    /// Validate tensor shapes match
    fn validate_shape_match(&self, other: &Self) -> WasmResult<()>;

    /// Validate tensor is non-empty
    fn validate_non_empty(&self) -> WasmResult<()>;

    /// Validate tensor dimensions
    fn validate_dimensions(&self, min_dims: usize) -> WasmResult<()>;

    /// Validate tensor size
    fn validate_min_size(&self, min_size: usize) -> WasmResult<()>;
}

impl WasmValidation for WasmTensor {
    fn validate_2d(&self) -> WasmResult<()> {
        if self.shape().len() < 2 {
            Err(WasmError::dimension_error(2, self.shape().len()))
        } else {
            Ok(())
        }
    }

    fn validate_shape_match(&self, other: &Self) -> WasmResult<()> {
        if self.shape() != other.shape() {
            Err(WasmError::shape_mismatch())
        } else {
            Ok(())
        }
    }

    fn validate_non_empty(&self) -> WasmResult<()> {
        if self.data().is_empty() {
            Err(WasmError::empty_tensor())
        } else {
            Ok(())
        }
    }

    fn validate_dimensions(&self, min_dims: usize) -> WasmResult<()> {
        if self.shape().len() < min_dims {
            Err(WasmError::dimension_error(min_dims, self.shape().len()))
        } else {
            Ok(())
        }
    }

    fn validate_min_size(&self, min_size: usize) -> WasmResult<()> {
        if self.data().len() < min_size {
            Err(WasmError::insufficient_data(
                "operation",
                min_size,
                self.data().len(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Validation utilities for common operations
pub struct WasmValidator;

impl WasmValidator {
    /// Validate image tensor for transformation
    pub fn validate_image_tensor(tensor: &WasmTensor) -> WasmResult<(usize, usize)> {
        tensor.validate_2d()?;
        let shape = tensor.shape();
        let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);

        if h == 0 || w == 0 {
            return Err(WasmError::invalid_param(
                "image_dimensions",
                format!("{}x{}", h, w),
                "must be non-zero",
            ));
        }

        Ok((h, w))
    }

    /// Validate crop parameters
    pub fn validate_crop_params(
        tensor: &WasmTensor,
        crop_h: usize,
        crop_w: usize,
    ) -> WasmResult<(usize, usize)> {
        let (img_h, img_w) = Self::validate_image_tensor(tensor)?;

        if crop_h > img_h || crop_w > img_w {
            return Err(WasmError::invalid_param(
                "crop_size",
                format!("{}x{}", crop_h, crop_w),
                &format!("larger than image {}x{}", img_h, img_w),
            ));
        }

        Ok((img_h, img_w))
    }

    /// Validate threshold parameter
    pub fn validate_threshold(threshold: f32, min: f32, max: f32) -> WasmResult<()> {
        if threshold < min || threshold > max {
            Err(WasmError::invalid_range("threshold", threshold, min, max))
        } else {
            Ok(())
        }
    }

    /// Validate percentage parameter (0-100)
    pub fn validate_percentage(value: f32, param_name: &str) -> WasmResult<()> {
        if value < 0.0 || value > 100.0 {
            Err(WasmError::invalid_range(param_name, value, 0.0, 100.0))
        } else {
            Ok(())
        }
    }

    /// Validate positive parameter
    pub fn validate_positive(value: f32, param_name: &str) -> WasmResult<()> {
        if value <= 0.0 {
            Err(WasmError::invalid_param(
                param_name,
                value,
                "must be positive",
            ))
        } else {
            Ok(())
        }
    }
}
