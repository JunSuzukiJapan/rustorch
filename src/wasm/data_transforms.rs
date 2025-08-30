//! WASM bindings for data transformations
//! データ変換のWASMバインディング

use wasm_bindgen::prelude::*;
use crate::wasm::tensor::WasmTensor;
use js_sys::Array;

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
    pub fn new(mean: &[f32], std: &[f32]) -> WasmNormalize {
        WasmNormalize {
            mean: mean.to_vec(),
            std: std.to_vec(),
        }
    }

    /// Apply normalization to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let data = tensor.data();
        let shape = tensor.shape();
        
        if self.mean.len() != self.std.len() {
            return Err(JsValue::from_str("Mean and std must have same length"));
        }
        
        let result: Vec<f32> = data.iter().enumerate().map(|(i, &x)| {
            let channel = i % self.mean.len();
            (x - self.mean[channel]) / self.std[channel]
        }).collect();
        
        Ok(WasmTensor::new(result, shape))
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
    pub fn new(height: usize, width: usize, interpolation: &str) -> WasmResize {
        WasmResize {
            height,
            width,
            interpolation: interpolation.to_string(),
        }
    }

    /// Apply resize to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(JsValue::from_str("Tensor must be at least 2D for resize"));
        }
        
        let (old_h, old_w) = (shape[shape.len()-2], shape[shape.len()-1]);
        let (new_h, new_w) = (self.height, self.width);
        
        let data = tensor.data();
        let mut new_data = Vec::with_capacity(new_h * new_w);
        
        // Simple nearest neighbor interpolation
        for i in 0..new_h {
            for j in 0..new_w {
                let old_i = (i * old_h) / new_h;
                let old_j = (j * old_w) / new_w;
                let idx = old_i * old_w + old_j;
                new_data.push(data[idx]);
            }
        }
        
        let mut new_shape = shape.clone();
        new_shape[shape.len()-2] = new_h;
        new_shape[shape.len()-1] = new_w;
        
        Ok(WasmTensor::new(new_data, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        format!("Resize({}, {})", self.height, self.width)
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
    pub fn new(height: usize, width: usize) -> WasmCenterCrop {
        WasmCenterCrop { height, width }
    }

    /// Apply center crop to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(JsValue::from_str("Tensor must be at least 2D for crop"));
        }
        
        let (img_h, img_w) = (shape[shape.len()-2], shape[shape.len()-1]);
        if self.height > img_h || self.width > img_w {
            return Err(JsValue::from_str("Crop size larger than image"));
        }
        
        let start_h = (img_h - self.height) / 2;
        let start_w = (img_w - self.width) / 2;
        
        let data = tensor.data();
        let mut new_data = Vec::with_capacity(self.height * self.width);
        
        for i in 0..self.height {
            for j in 0..self.width {
                let idx = (start_h + i) * img_w + (start_w + j);
                new_data.push(data[idx]);
            }
        }
        
        let mut new_shape = shape.clone();
        new_shape[shape.len()-2] = self.height;
        new_shape[shape.len()-1] = self.width;
        
        Ok(WasmTensor::new(new_data, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        format!("CenterCrop({}, {})", self.height, self.width)
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
    pub fn new(height: usize, width: usize, padding: Option<usize>) -> WasmRandomCrop {
        WasmRandomCrop { height, width, padding }
    }

    /// Apply random crop to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(JsValue::from_str("Tensor must be at least 2D for crop"));
        }
        
        let (img_h, img_w) = (shape[shape.len()-2], shape[shape.len()-1]);
        if self.height > img_h || self.width > img_w {
            return Err(JsValue::from_str("Crop size larger than image"));
        }
        
        let max_start_h = img_h - self.height;
        let max_start_w = img_w - self.width;
        let start_h = (js_sys::Math::random() * max_start_h as f64) as usize;
        let start_w = (js_sys::Math::random() * max_start_w as f64) as usize;
        
        let data = tensor.data();
        let mut new_data = Vec::with_capacity(self.height * self.width);
        
        for i in 0..self.height {
            for j in 0..self.width {
                let idx = (start_h + i) * img_w + (start_w + j);
                new_data.push(data[idx]);
            }
        }
        
        let mut new_shape = shape.clone();
        new_shape[shape.len()-2] = self.height;
        new_shape[shape.len()-1] = self.width;
        
        Ok(WasmTensor::new(new_data, new_shape))
    }

    /// Get transformation name
    pub fn name(&self) -> String {
        format!("RandomCrop({}, {})", self.height, self.width)
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
    pub fn new(brightness: f32, contrast: f32, saturation: f32, hue: f32) -> WasmColorJitter {
        WasmColorJitter { brightness, contrast, saturation, hue }
    }

    /// Apply color jitter to tensor
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let brightness_factor = 1.0 + (js_sys::Math::random() as f32 - 0.5) * self.brightness;
        let contrast_factor = 1.0 + (js_sys::Math::random() as f32 - 0.5) * self.contrast;
        
        let result: Vec<f32> = tensor.data().iter()
            .map(|&x| ((x * brightness_factor) * contrast_factor).clamp(0.0, 1.0))
            .collect();
        
        Ok(WasmTensor::new(result, tensor.shape()))
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

    /// Apply to tensor transformation
    pub fn apply(&self, tensor: &WasmTensor) -> Result<WasmTensor, JsValue> {
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
    "RusTorch WASM Data Transforms v0.5.2".to_string()
}

/// Create ImageNet preprocessing pipeline
#[wasm_bindgen]
pub fn create_imagenet_preprocessing() -> WasmNormalize {
    WasmNormalize::new(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])
}

/// Create CIFAR preprocessing pipeline
#[wasm_bindgen]
pub fn create_cifar_preprocessing() -> WasmNormalize {
    WasmNormalize::new(&[0.4914, 0.4822, 0.4465], &[0.2023, 0.1994, 0.2010])
}