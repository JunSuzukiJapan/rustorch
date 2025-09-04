//! Browser-specific WASM utilities
//! ブラウザ専用WASM機能

#[cfg(feature = "wasm")]
use crate::wasm::tensor::WasmTensor;
#[cfg(feature = "wasm")]
use js_sys::Array;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Browser storage utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct BrowserStorage;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl BrowserStorage {
    /// Create new browser storage utility
    #[wasm_bindgen(constructor)]
    pub fn new() -> BrowserStorage {
        BrowserStorage
    }

    /// Save tensor to localStorage
    #[wasm_bindgen]
    pub fn save_tensor(&self, key: &str, tensor: &WasmTensor) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window.local_storage()?.ok_or("No localStorage")?;

        let data = serde_json::to_string(&(tensor.data(), tensor.shape()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        storage.set_item(key, &data)?;
        Ok(())
    }

    /// Load tensor from localStorage
    #[wasm_bindgen]
    pub fn load_tensor(&self, key: &str) -> Result<WasmTensor, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window.local_storage()?.ok_or("No localStorage")?;

        let data = storage.get_item(key)?.ok_or("Key not found")?;

        let (tensor_data, tensor_shape): (Vec<f32>, Vec<usize>) =
            serde_json::from_str(&data).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmTensor::new(tensor_data, tensor_shape))
    }

    /// List all saved tensor keys
    #[wasm_bindgen]
    pub fn list_tensor_keys(&self) -> Result<Array, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window.local_storage()?.ok_or("No localStorage")?;

        let result = Array::new();
        let length = storage.length()?;

        for i in 0..length {
            if let Some(key) = storage.key(i)? {
                if key.starts_with("rustorch_tensor_") {
                    result.push(&JsValue::from_str(&key));
                }
            }
        }

        Ok(result)
    }

    /// Clear all saved tensors
    #[wasm_bindgen]
    pub fn clear_tensors(&self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window.local_storage()?.ok_or("No localStorage")?;

        let keys_to_remove: Vec<String> = (0..storage.length()?)
            .filter_map(|i| storage.key(i).ok().flatten())
            .filter(|key| key.starts_with("rustorch_tensor_"))
            .collect();

        for key in keys_to_remove {
            storage.remove_item(&key)?;
        }

        Ok(())
    }
}

/// File API utilities for loading tensors from files
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct FileLoader;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl FileLoader {
    /// Create new file loader utility
    #[wasm_bindgen(constructor)]
    pub fn new() -> FileLoader {
        FileLoader
    }

    /// Create file input element for tensor loading
    #[wasm_bindgen]
    pub fn create_file_input(&self) -> Result<web_sys::HtmlInputElement, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let document = window.document().ok_or("No document")?;

        let input = document
            .create_element("input")?
            .dyn_into::<web_sys::HtmlInputElement>()?;

        input.set_type("file");
        input.set_accept(".json,.txt,.csv");
        input.set_multiple(false);

        Ok(input)
    }
}

/// Canvas utilities for tensor visualization
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct CanvasRenderer {
    canvas: web_sys::HtmlCanvasElement,
    context: web_sys::CanvasRenderingContext2d,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl CanvasRenderer {
    /// Create new canvas renderer for the specified canvas element
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<CanvasRenderer, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let document = window.document().ok_or("No document")?;

        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;

        let context = canvas
            .get_context("2d")?
            .ok_or("Failed to get 2D context")?
            .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

        Ok(CanvasRenderer { canvas, context })
    }

    /// Render 2D tensor as heatmap
    #[wasm_bindgen]
    pub fn render_heatmap(&self, tensor: &WasmTensor) -> Result<(), JsValue> {
        let shape = tensor.shape();
        let data = tensor.data();

        if shape.len() != 2 {
            return Err(JsValue::from_str("Only 2D tensors supported"));
        }

        let (rows, cols) = (shape[0], shape[1]);
        self.canvas.set_width(cols as u32 * 10);
        self.canvas.set_height(rows as u32 * 10);

        // Find min/max for normalization
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        for i in 0..rows {
            for j in 0..cols {
                let val = data[i * cols + j];
                let normalized = if range > 0.0 {
                    (val - min_val) / range
                } else {
                    0.0
                };

                // Simple grayscale mapping
                let intensity = (normalized * 255.0) as u8;
                let color = format!("rgb({}, {}, {})", intensity, intensity, intensity);

                self.context.set_fill_style_str(&color);
                self.context
                    .fill_rect((j * 10) as f64, (i * 10) as f64, 10.0, 10.0);
            }
        }

        Ok(())
    }

    /// Clear canvas
    #[wasm_bindgen]
    pub fn clear(&self) {
        let width = self.canvas.width() as f64;
        let height = self.canvas.height() as f64;
        self.context.clear_rect(0.0, 0.0, width, height);
    }
}

/// Web Worker utilities for background computation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WorkerManager {
    worker: Option<web_sys::Worker>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WorkerManager {
    /// Create new web worker manager
    #[wasm_bindgen(constructor)]
    pub fn new() -> WorkerManager {
        WorkerManager { worker: None }
    }

    /// Create and start a web worker
    #[wasm_bindgen]
    pub fn create_worker(&mut self, script_url: &str) -> Result<(), JsValue> {
        let worker = web_sys::Worker::new(script_url)?;
        self.worker = Some(worker);
        Ok(())
    }

    /// Send tensor data to worker
    #[wasm_bindgen]
    pub fn send_tensor(&self, tensor: &WasmTensor) -> Result<(), JsValue> {
        if let Some(worker) = &self.worker {
            let message = js_sys::Object::new();
            let shape_array = Array::new();
            for dim in tensor.shape() {
                shape_array.push(&JsValue::from(dim));
            }
            js_sys::Reflect::set(&message, &"data".into(), &tensor.data().into())?;
            js_sys::Reflect::set(&message, &"shape".into(), &shape_array.into())?;
            worker.post_message(&message)?;
        }
        Ok(())
    }

    /// Terminate worker
    #[wasm_bindgen]
    pub fn terminate(&mut self) {
        if let Some(worker) = &self.worker {
            worker.terminate();
        }
        self.worker = None;
    }
}

/// Performance monitoring utilities
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct PerformanceMonitor;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl PerformanceMonitor {
    /// Get memory usage information
    #[wasm_bindgen]
    pub fn get_memory_info() -> Result<js_sys::Object, JsValue> {
        let info = js_sys::Object::new();

        if let Some(window) = web_sys::window() {
            if let Some(performance) = window.performance() {
                if let Ok(memory) = js_sys::Reflect::get(&performance, &"memory".into()) {
                    if !memory.is_undefined() {
                        return Ok(memory.into());
                    }
                }
            }
        }

        // Fallback info
        js_sys::Reflect::set(
            &info,
            &"message".into(),
            &"Memory info not available".into(),
        )?;
        Ok(info)
    }

    /// Measure function execution time
    #[wasm_bindgen]
    pub fn time_function(name: &str) {
        web_sys::console::time_with_label(name);
    }

    /// End timing measurement
    #[wasm_bindgen]
    pub fn time_end(name: &str) {
        web_sys::console::time_end_with_label(name);
    }
}
