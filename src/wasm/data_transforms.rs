//! WASM bindings for data transformations - Refactored
//! データ変換のWASMバインディング - リファクタリング版

use crate::wasm::common::{
    MemoryManager, PooledBuffer, WasmError, WasmImageOperation, WasmNaming, WasmOperation,
    WasmParamValidator, WasmResult, WasmTransform, WasmValidation, WasmVersion,
};
use crate::wasm::tensor::WasmTensor;
use wasm_bindgen::prelude::*;

/// WASM wrapper for Normalize transformation
#[wasm_bindgen]
pub struct WasmNormalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[wasm_bindgen]
impl WasmNormalize {
    /// Create new normalization transform
    #[wasm_bindgen(constructor)]
    pub fn new(mean: &[f32], std: &[f32]) -> WasmResult<WasmNormalize> {
        if mean.len() != std.len() {
            return Err(WasmError::size_mismatch(mean.len(), std.len()));
        }

        if mean.is_empty() {
            return Err(WasmError::empty_tensor());
        }

        Ok(WasmNormalize {
            mean: mean.to_vec(),
            std: std.to_vec(),
        })
    }

    /// Apply normalization to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;

        let data = tensor.data();
        let shape = tensor.shape();

        let mut result_buffer = MemoryManager::get_buffer(data.len());
        result_buffer.extend(data.iter().enumerate().map(|(i, &x)| {
            let channel = i % self.mean.len();
            (x - self.mean[channel]) / self.std[channel]
        }));

        Ok(WasmTensor::new(result_buffer, shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        "Normalize".to_string()
    }
}

/// WASM wrapper for Resize transformation
#[wasm_bindgen]
pub struct WasmResize {
    height: usize,
    width: usize,
    interpolation: String,
}

#[wasm_bindgen]
impl WasmResize {
    /// Create new resize transform
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize, interpolation: &str) -> WasmResult<WasmResize> {
        WasmParamValidator::validate_dimensions(width, height, "dimensions")?;

        match interpolation {
            "nearest" | "bilinear" | "bicubic" => {}
            _ => {
                return Err(WasmError::invalid_param(
                    "interpolation",
                    interpolation,
                    "must be nearest, bilinear, or bicubic",
                ))
            }
        }

        Ok(WasmResize {
            height,
            width,
            interpolation: interpolation.to_string(),
        })
    }

    /// Apply resize to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        let (old_h, old_w) = crate::wasm::common::WasmValidator::validate_image_tensor(tensor)?;
        let (new_h, new_w) = (self.height, self.width);

        let data = tensor.data();
        let mut result_buffer = MemoryManager::get_buffer(new_h * new_w);

        // Optimized nearest neighbor interpolation
        for i in 0..new_h {
            for j in 0..new_w {
                let old_i = (i * old_h) / new_h;
                let old_j = (j * old_w) / new_w;
                let idx = old_i * old_w + old_j;
                result_buffer.push(data[idx]);
            }
        }

        let mut new_shape = tensor.shape();
        let shape_len = new_shape.len();
        new_shape[shape_len - 2] = new_h;
        new_shape[shape_len - 1] = new_w;

        Ok(WasmTensor::new(result_buffer, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        WasmNaming::transform_with_dims("Resize", self.width, self.height)
    }
}

/// WASM wrapper for CenterCrop transformation
#[wasm_bindgen]
pub struct WasmCenterCrop {
    height: usize,
    width: usize,
}

#[wasm_bindgen]
impl WasmCenterCrop {
    /// Create new center crop transform
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize) -> WasmResult<WasmCenterCrop> {
        WasmParamValidator::validate_dimensions(width, height, "dimensions")?;

        Ok(WasmCenterCrop { height, width })
    }

    /// Apply center crop to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        let (img_h, img_w) = crate::wasm::common::WasmValidator::validate_crop_params(
            tensor,
            self.height,
            self.width,
        )?;

        let start_h = (img_h - self.height) / 2;
        let start_w = (img_w - self.width) / 2;

        let data = tensor.data();
        let mut result_buffer = MemoryManager::get_buffer(self.height * self.width);

        for i in 0..self.height {
            for j in 0..self.width {
                let idx = (start_h + i) * img_w + (start_w + j);
                result_buffer.push(data[idx]);
            }
        }

        let mut new_shape = tensor.shape();
        let shape_len = new_shape.len();
        new_shape[shape_len - 2] = self.height;
        new_shape[shape_len - 1] = self.width;

        Ok(WasmTensor::new(result_buffer, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        WasmNaming::transform_with_dims("CenterCrop", self.width, self.height)
    }
}

/// WASM wrapper for RandomCrop transformation
#[wasm_bindgen]
pub struct WasmRandomCrop {
    height: usize,
    width: usize,
    padding: Option<usize>,
}

#[wasm_bindgen]
impl WasmRandomCrop {
    /// Create new random crop transform
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize, padding: Option<usize>) -> WasmResult<WasmRandomCrop> {
        WasmParamValidator::validate_dimensions(width, height, "dimensions")?;

        Ok(WasmRandomCrop {
            height,
            width,
            padding,
        })
    }

    /// Apply random crop to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        let (img_h, img_w) = crate::wasm::common::WasmValidator::validate_crop_params(
            tensor,
            self.height,
            self.width,
        )?;

        let max_start_h = img_h - self.height;
        let max_start_w = img_w - self.width;
        let start_h = (js_sys::Math::random() * max_start_h as f64) as usize;
        let start_w = (js_sys::Math::random() * max_start_w as f64) as usize;

        let data = tensor.data();
        let mut result_buffer = MemoryManager::get_buffer(self.height * self.width);

        for i in 0..self.height {
            for j in 0..self.width {
                let idx = (start_h + i) * img_w + (start_w + j);
                result_buffer.push(data[idx]);
            }
        }

        let mut new_shape = tensor.shape();
        let shape_len = new_shape.len();
        new_shape[shape_len - 2] = self.height;
        new_shape[shape_len - 1] = self.width;

        Ok(WasmTensor::new(result_buffer, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        WasmNaming::transform_with_dims("RandomCrop", self.width, self.height)
    }
}

/// WASM wrapper for ColorJitter transformation
#[wasm_bindgen]
pub struct WasmColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
}

#[wasm_bindgen]
impl WasmColorJitter {
    /// Create new color jitter transform
    #[wasm_bindgen(constructor)]
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    ) -> WasmResult<WasmColorJitter> {
        // Validate parameters are within reasonable ranges
        if brightness < 0.0 || contrast < 0.0 || saturation < 0.0 || hue < 0.0 {
            return Err(WasmError::invalid_param(
                "jitter_params",
                "negative values",
                "all parameters must be non-negative",
            ));
        }

        Ok(WasmColorJitter {
            brightness,
            contrast,
            saturation,
            hue,
        })
    }

    /// Apply color jitter to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;

        let brightness_factor = 1.0 + (js_sys::Math::random() as f32 - 0.5) * self.brightness;
        let contrast_factor = 1.0 + (js_sys::Math::random() as f32 - 0.5) * self.contrast;

        let data = tensor.data();
        let mut result_buffer = MemoryManager::get_buffer(data.len());

        result_buffer.extend(
            data.iter()
                .map(|&x| ((x * brightness_factor) * contrast_factor).clamp(0.0, 1.0)),
        );

        Ok(WasmTensor::new(result_buffer, tensor.shape()))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        "ColorJitter".to_string()
    }
}

/// WASM wrapper for ToTensor transformation
#[wasm_bindgen]
pub struct WasmToTensor;

#[wasm_bindgen]
impl WasmToTensor {
    /// Create new to tensor transform
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmToTensor {
        WasmToTensor
    }

    /// Apply to tensor transformation (identity operation)
    pub fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        tensor.validate_non_empty()?;
        Ok(WasmTensor::new(tensor.data(), tensor.shape()))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        "ToTensor".to_string()
    }
}

/// Version information
#[wasm_bindgen]
pub fn wasm_transforms_version() -> String {
    WasmVersion::module_version("Data Transforms")
}

/// Create ImageNet preprocessing pipeline
#[wasm_bindgen]
pub fn create_imagenet_preprocessing() -> WasmResult<WasmNormalize> {
    WasmNormalize::new(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])
}

/// Create CIFAR preprocessing pipeline
#[wasm_bindgen]
pub fn create_cifar_preprocessing() -> WasmResult<WasmNormalize> {
    WasmNormalize::new(&[0.4914, 0.4822, 0.4465], &[0.2023, 0.1994, 0.2010])
}

// Trait implementations for WasmNormalize
impl WasmOperation for WasmNormalize {
    fn name(&self) -> String {
        "Normalize".to_string()
    }
}

impl WasmTransform for WasmNormalize {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}

// Trait implementations for WasmResize
impl WasmOperation for WasmResize {
    fn name(&self) -> String {
        self.name()
    }
}

impl WasmTransform for WasmResize {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}

impl WasmImageOperation for WasmResize {
    fn output_dimensions(&self, _input_dims: (usize, usize)) -> (usize, usize) {
        (self.height, self.width)
    }
}

// Trait implementations for WasmCenterCrop
impl WasmOperation for WasmCenterCrop {
    fn name(&self) -> String {
        self.name()
    }
}

impl WasmTransform for WasmCenterCrop {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}

impl WasmImageOperation for WasmCenterCrop {
    fn output_dimensions(&self, _input_dims: (usize, usize)) -> (usize, usize) {
        (self.height, self.width)
    }
}

// Trait implementations for WasmRandomCrop
impl WasmOperation for WasmRandomCrop {
    fn name(&self) -> String {
        self.name()
    }
}

impl WasmTransform for WasmRandomCrop {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}

impl WasmImageOperation for WasmRandomCrop {
    fn output_dimensions(&self, _input_dims: (usize, usize)) -> (usize, usize) {
        (self.height, self.width)
    }
}

// Trait implementations for WasmColorJitter
impl WasmOperation for WasmColorJitter {
    fn name(&self) -> String {
        self.name()
    }
}

impl WasmTransform for WasmColorJitter {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}

impl WasmImageOperation for WasmColorJitter {
    fn output_dimensions(&self, input_dims: (usize, usize)) -> (usize, usize) {
        input_dims
    }
}

// Trait implementations for WasmToTensor
impl WasmOperation for WasmToTensor {
    fn name(&self) -> String {
        self.name()
    }
}

impl WasmTransform for WasmToTensor {
    fn apply(&self, tensor: &WasmTensor) -> WasmResult<WasmTensor> {
        self.apply(tensor)
    }
}
