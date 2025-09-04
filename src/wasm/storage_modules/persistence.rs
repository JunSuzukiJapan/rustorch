//! Browser model storage and persistence for WebAssembly
//! WebAssembly向けブラウザモデル保存・永続化

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

/// WASM model storage for browser persistence
/// ブラウザ永続化用WASMモデルストレージ
#[wasm_bindgen]
pub struct WasmModelStorage {
    use_indexeddb: bool,
    chunk_size: usize,
}

#[wasm_bindgen]
impl WasmModelStorage {
    /// Create new model storage
    /// 新しいモデルストレージを作成
    #[wasm_bindgen(constructor)]
    pub fn new(use_indexeddb: bool, chunk_size: usize) -> WasmModelStorage {
        WasmModelStorage {
            use_indexeddb,
            chunk_size: if chunk_size > 0 {
                chunk_size
            } else {
                1_000_000
            },
        }
    }

    /// Save model data to browser storage
    /// ブラウザストレージにモデルデータを保存
    #[wasm_bindgen]
    pub async fn save_model(&self, model_name: &str, model_data: Vec<u8>) -> Result<(), JsValue> {
        if self.use_indexeddb {
            self.save_to_indexeddb(model_name, &model_data).await
        } else {
            self.save_to_localstorage(model_name, &model_data)
        }
    }

    /// Load model data from browser storage
    /// ブラウザストレージからモデルデータを読み込み
    #[wasm_bindgen]
    pub async fn load_model(&self, model_name: &str) -> Result<Vec<u8>, JsValue> {
        if self.use_indexeddb {
            self.load_from_indexeddb(model_name).await
        } else {
            self.load_from_localstorage(model_name)
        }
    }

    /// Save large model in chunks to avoid memory limits
    /// メモリ制限を避けるために大きなモデルをチャンク単位で保存
    #[wasm_bindgen]
    pub async fn save_large_model(
        &self,
        model_name: &str,
        model_data: Vec<u8>,
    ) -> Result<(), JsValue> {
        let chunk_count = (model_data.len() + self.chunk_size - 1) / self.chunk_size;

        // Save metadata
        let metadata = js_sys::Object::new();
        js_sys::Reflect::set(
            &metadata,
            &"total_size".into(),
            &JsValue::from(model_data.len()),
        )?;
        js_sys::Reflect::set(
            &metadata,
            &"chunk_count".into(),
            &JsValue::from(chunk_count),
        )?;
        js_sys::Reflect::set(
            &metadata,
            &"chunk_size".into(),
            &JsValue::from(self.chunk_size),
        )?;

        let metadata_key = format!("{}_metadata", model_name);
        self.save_object(&metadata_key, &metadata).await?;

        // Save chunks
        for i in 0..chunk_count {
            let start = i * self.chunk_size;
            let end = ((i + 1) * self.chunk_size).min(model_data.len());
            let chunk = model_data[start..end].to_vec();

            let chunk_key = format!("{}_{}", model_name, i);
            self.save_model(&chunk_key, chunk).await?;
        }

        Ok(())
    }

    /// Load large model progressively from chunks
    /// チャンクから大きなモデルを段階的に読み込み
    #[wasm_bindgen]
    pub async fn load_large_model(&self, model_name: &str) -> Result<Vec<u8>, JsValue> {
        let metadata_key = format!("{}_metadata", model_name);
        let metadata = self.load_object(&metadata_key).await?;

        let total_size = js_sys::Reflect::get(&metadata, &"total_size".into())?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Invalid total_size"))?
            as usize;
        let chunk_count = js_sys::Reflect::get(&metadata, &"chunk_count".into())?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Invalid chunk_count"))?
            as usize;

        let mut model_data = Vec::with_capacity(total_size);

        for i in 0..chunk_count {
            let chunk_key = format!("{}_{}", model_name, i);
            let chunk = self.load_model(&chunk_key).await?;
            model_data.extend(chunk);
        }

        Ok(model_data)
    }

    /// Delete model from storage
    /// ストレージからモデルを削除
    #[wasm_bindgen]
    pub async fn delete_model(&self, model_name: &str) -> Result<(), JsValue> {
        if self.use_indexeddb {
            self.delete_from_indexeddb(model_name).await
        } else {
            self.delete_from_localstorage(model_name)
        }
    }

    /// List all stored models
    /// 保存されたすべてのモデルをリスト表示
    #[wasm_bindgen]
    pub async fn list_models(&self) -> Result<Vec<String>, JsValue> {
        if self.use_indexeddb {
            self.list_from_indexeddb().await
        } else {
            self.list_from_localstorage()
        }
    }

    /// Get storage size estimate in bytes
    /// ストレージサイズの推定（バイト単位）
    #[wasm_bindgen]
    pub async fn get_storage_size(&self) -> Result<f64, JsValue> {
        if self.use_indexeddb {
            // IndexedDB size estimation
            let navigator = window()?.navigator();
            if let Ok(storage) = js_sys::Reflect::get(&navigator, &"storage".into()) {
                if let Ok(estimate_func) = js_sys::Reflect::get(&storage, &"estimate".into()) {
                    let estimate_promise = js_sys::Function::from(estimate_func).call0(&storage)?;
                    let estimate_result =
                        JsFuture::from(js_sys::Promise::from(estimate_promise)).await?;

                    if let Ok(usage) = js_sys::Reflect::get(&estimate_result, &"usage".into()) {
                        return Ok(usage.as_f64().unwrap_or(0.0));
                    }
                }
            }
            Ok(0.0)
        } else {
            // LocalStorage size estimation
            let storage = window()?
                .local_storage()?
                .ok_or("LocalStorage not available")?;
            let mut total_size = 0.0;

            for i in 0..storage.length()? {
                if let Some(key) = storage.key(i)? {
                    if let Some(value) = storage.get_item(&key)? {
                        total_size += (key.len() + value.len()) as f64;
                    }
                }
            }

            Ok(total_size)
        }
    }

    /// Check available storage space
    /// 利用可能なストレージ容量をチェック
    #[wasm_bindgen]
    pub async fn get_available_storage(&self) -> Result<f64, JsValue> {
        let navigator = window()?.navigator();
        if let Ok(storage) = js_sys::Reflect::get(&navigator, &"storage".into()) {
            if let Ok(estimate_func) = js_sys::Reflect::get(&storage, &"estimate".into()) {
                let estimate_promise = js_sys::Function::from(estimate_func).call0(&storage)?;
                let estimate_result =
                    JsFuture::from(js_sys::Promise::from(estimate_promise)).await?;

                let quota = js_sys::Reflect::get(&estimate_result, &"quota".into())?
                    .as_f64()
                    .unwrap_or(0.0);
                let usage = js_sys::Reflect::get(&estimate_result, &"usage".into())?
                    .as_f64()
                    .unwrap_or(0.0);

                return Ok(quota - usage);
            }
        }

        // Fallback: estimate based on LocalStorage
        Ok(10_000_000.0) // ~10MB typical LocalStorage limit
    }

    // Private implementation methods

    async fn save_to_indexeddb(&self, key: &str, data: &[u8]) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let idb_factory = window.indexed_db()?.ok_or("IndexedDB not supported")?;

        // Open database
        let db_request = idb_factory
            .open("rustorch_models")
            .map_err(|_| "Failed to open database")?;
        let db_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onupgradeneeded = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        let db = web_sys::IdbDatabase::from(result);
                        let _object_store = db.create_object_store("models");
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Database error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            db_request.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
            db_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            db_request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onupgradeneeded.forget();
            onsuccess.forget();
            onerror.forget();
        });

        let db_result = JsFuture::from(db_promise).await?;
        let db = web_sys::IdbDatabase::from(db_result);

        // Store data
        let transaction =
            db.transaction_with_str_and_mode("models", web_sys::IdbTransactionMode::Readwrite)?;
        let store = transaction.object_store("models")?;

        let uint8_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        uint8_array.copy_from(data);

        let _request = store.put_with_key(&uint8_array, &JsValue::from_str(key))?;

        Ok(())
    }

    async fn load_from_indexeddb(&self, key: &str) -> Result<Vec<u8>, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let idb_factory = window.indexed_db()?.ok_or("IndexedDB not supported")?;

        let db_request = idb_factory
            .open("rustorch_models")
            .map_err(|_| "Failed to open database")?;
        let db_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Database error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            db_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            db_request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onsuccess.forget();
            onerror.forget();
        });

        let db_result = JsFuture::from(db_promise).await?;
        let db = web_sys::IdbDatabase::from(db_result);

        let transaction = db.transaction_with_str("models")?;
        let store = transaction.object_store("models")?;
        let request = store.get(&JsValue::from_str(key))?;

        let get_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Get error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onsuccess.forget();
            onerror.forget();
        });

        let data_result = JsFuture::from(get_promise).await?;
        let uint8_array = js_sys::Uint8Array::from(data_result);
        Ok(uint8_array.to_vec())
    }

    async fn delete_from_indexeddb(&self, key: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let idb_factory = window.indexed_db()?.ok_or("IndexedDB not supported")?;

        let db_request = idb_factory
            .open("rustorch_models")
            .map_err(|_| "Failed to open database")?;
        let db_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Database error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            db_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            db_request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onsuccess.forget();
            onerror.forget();
        });

        let db_result = JsFuture::from(db_promise).await?;
        let db = web_sys::IdbDatabase::from(db_result);

        let transaction =
            db.transaction_with_str_and_mode("models", web_sys::IdbTransactionMode::Readwrite)?;
        let store = transaction.object_store("models")?;
        let _request = store.delete(&JsValue::from_str(key))?;

        Ok(())
    }

    async fn list_from_indexeddb(&self) -> Result<Vec<String>, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let idb_factory = window.indexed_db()?.ok_or("IndexedDB not supported")?;

        let db_request = idb_factory
            .open("rustorch_models")
            .map_err(|_| "Failed to open database")?;
        let db_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Database error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            db_request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            db_request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onsuccess.forget();
            onerror.forget();
        });

        let db_result = JsFuture::from(db_promise).await?;
        let db = web_sys::IdbDatabase::from(db_result);

        let transaction = db.transaction_with_str("models")?;
        let store = transaction.object_store("models")?;
        let request = store.get_all_keys()?;

        let keys_promise = js_sys::Promise::new(&mut |resolve, reject| {
            let onsuccess = Closure::wrap(Box::new(move |event: web_sys::Event| {
                if let Some(target) = event.target() {
                    if let Ok(result) = js_sys::Reflect::get(&target, &"result".into()) {
                        resolve.call1(&JsValue::UNDEFINED, &result).unwrap();
                    }
                }
            }) as Box<dyn FnMut(_)>);

            let onerror = Closure::wrap(Box::new(move |_: web_sys::Event| {
                reject
                    .call1(&JsValue::UNDEFINED, &JsValue::from_str("Keys error"))
                    .unwrap();
            }) as Box<dyn FnMut(_)>);

            request.set_onsuccess(Some(onsuccess.as_ref().unchecked_ref()));
            request.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onsuccess.forget();
            onerror.forget();
        });

        let keys_result = JsFuture::from(keys_promise).await?;
        let keys_array = js_sys::Array::from(&keys_result);

        let mut model_names = Vec::new();
        for i in 0..keys_array.length() {
            if let Some(key) = keys_array.get(i).as_string() {
                if !key.ends_with("_metadata") && !key.contains('_') {
                    model_names.push(key);
                }
            }
        }

        Ok(model_names)
    }

    fn save_to_localstorage(&self, key: &str, data: &[u8]) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window
            .local_storage()?
            .ok_or("LocalStorage not available")?;

        // Convert bytes to base64 for storage
        let base64_data = base64_encode(data);
        storage.set_item(key, &base64_data)?;

        Ok(())
    }

    fn load_from_localstorage(&self, key: &str) -> Result<Vec<u8>, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window
            .local_storage()?
            .ok_or("LocalStorage not available")?;

        let base64_data = storage.get_item(key)?.ok_or("Model not found")?;
        base64_decode(&base64_data)
    }

    fn delete_from_localstorage(&self, key: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window
            .local_storage()?
            .ok_or("LocalStorage not available")?;

        storage.remove_item(key)?;
        Ok(())
    }

    fn list_from_localstorage(&self) -> Result<Vec<String>, JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let storage = window
            .local_storage()?
            .ok_or("LocalStorage not available")?;

        let mut model_names = Vec::new();
        for i in 0..storage.length()? {
            if let Some(key) = storage.key(i)? {
                if !key.ends_with("_metadata") && !key.contains('_') {
                    model_names.push(key);
                }
            }
        }

        Ok(model_names)
    }

    async fn save_object(&self, key: &str, obj: &js_sys::Object) -> Result<(), JsValue> {
        let json_string = js_sys::JSON::stringify(obj)?;
        let json_str = json_string.as_string().ok_or("Failed to stringify")?;
        let data = json_str.as_bytes();

        self.save_to_localstorage(key, data)
    }

    async fn load_object(&self, key: &str) -> Result<js_sys::Object, JsValue> {
        let data = self.load_from_localstorage(key)?;
        let json_str = String::from_utf8(data).map_err(|_| JsValue::from_str("Invalid UTF-8"))?;
        let obj = js_sys::JSON::parse(&json_str)?;
        Ok(js_sys::Object::from(obj))
    }
}

/// Model compression utilities for storage
/// ストレージ用モデル圧縮ユーティリティ
#[wasm_bindgen]
pub struct WasmModelCompression;

#[wasm_bindgen]
impl WasmModelCompression {
    /// Simple RLE compression for model weights
    /// モデル重みの簡単なRLE圧縮
    #[wasm_bindgen]
    pub fn compress_weights(weights: Vec<f32>) -> Vec<u8> {
        let mut compressed = Vec::new();
        let mut i = 0;

        while i < weights.len() {
            let current = weights[i];
            let mut count = 1u8;

            // Count consecutive equal values (up to 255)
            while i + (count as usize) < weights.len()
                && weights[i + (count as usize)] == current
                && count < 255
            {
                count += 1;
            }

            // Store count and value
            compressed.push(count);
            compressed.extend_from_slice(&current.to_ne_bytes());

            i += count as usize;
        }

        compressed
    }

    /// Decompress RLE compressed weights
    /// RLE圧縮された重みを展開
    #[wasm_bindgen]
    pub fn decompress_weights(compressed: Vec<u8>) -> Vec<f32> {
        let mut weights = Vec::new();
        let mut i = 0;

        while i + 4 < compressed.len() {
            let count = compressed[i];
            let value_bytes = [
                compressed[i + 1],
                compressed[i + 2],
                compressed[i + 3],
                compressed[i + 4],
            ];
            let value = f32::from_ne_bytes(value_bytes);

            for _ in 0..count {
                weights.push(value);
            }

            i += 5; // 1 byte count + 4 bytes f32
        }

        weights
    }

    /// Quantize weights to reduce storage size
    /// ストレージサイズ削減のための重み量子化
    #[wasm_bindgen]
    pub fn quantize_weights(weights: Vec<f32>, bits: u8) -> Vec<u8> {
        if bits > 16 {
            return Vec::new(); // Unsupported
        }

        let max_val = weights.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = (1 << (bits - 1)) as f32 / max_val;

        let mut quantized = Vec::new();

        // Store scale factor
        quantized.extend_from_slice(&scale.to_ne_bytes());
        quantized.push(bits);

        for &weight in &weights {
            let quantized_val = (weight * scale).round() as i16;
            match bits {
                8 => quantized.push(quantized_val.max(i8::MIN as i16).min(i8::MAX as i16) as u8),
                16 => quantized.extend_from_slice(&(quantized_val as u16).to_ne_bytes()),
                _ => quantized.push((quantized_val as i8) as u8),
            }
        }

        quantized
    }

    /// Dequantize weights back to f32
    /// 重みをf32に逆量子化
    #[wasm_bindgen]
    pub fn dequantize_weights(quantized: Vec<u8>) -> Vec<f32> {
        if quantized.len() < 5 {
            return Vec::new();
        }

        let scale_bytes = [quantized[0], quantized[1], quantized[2], quantized[3]];
        let scale = f32::from_ne_bytes(scale_bytes);
        let bits = quantized[4];

        let mut weights = Vec::new();
        let data_start = 5;

        match bits {
            8 => {
                for i in data_start..quantized.len() {
                    let quantized_val = quantized[i] as i8;
                    weights.push(quantized_val as f32 / scale);
                }
            }
            16 => {
                let mut i = data_start;
                while i + 1 < quantized.len() {
                    let value_bytes = [quantized[i], quantized[i + 1]];
                    let quantized_val = u16::from_ne_bytes(value_bytes) as i16;
                    weights.push(quantized_val as f32 / scale);
                    i += 2;
                }
            }
            _ => {
                // Default to 8-bit
                for i in data_start..quantized.len() {
                    let quantized_val = quantized[i] as i8;
                    weights.push(quantized_val as f32 / scale);
                }
            }
        }

        weights
    }
}

/// Progress tracker for large model operations
/// 大きなモデル操作の進捗トラッカー
#[wasm_bindgen]
pub struct WasmProgressTracker {
    current_step: usize,
    total_steps: usize,
    operation: String,
}

#[wasm_bindgen]
impl WasmProgressTracker {
    #[wasm_bindgen(constructor)]
    pub fn new(total_steps: usize, operation: String) -> WasmProgressTracker {
        WasmProgressTracker {
            current_step: 0,
            total_steps,
            operation,
        }
    }

    #[wasm_bindgen]
    pub fn update(&mut self, step: usize) {
        self.current_step = step.min(self.total_steps);
    }

    #[wasm_bindgen]
    pub fn progress_percent(&self) -> f32 {
        if self.total_steps == 0 {
            100.0
        } else {
            (self.current_step as f32 / self.total_steps as f32) * 100.0
        }
    }

    #[wasm_bindgen]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.total_steps
    }

    #[wasm_bindgen]
    pub fn get_status(&self) -> String {
        format!(
            "{}: {}/{} ({:.1}%)",
            self.operation,
            self.current_step,
            self.total_steps,
            self.progress_percent()
        )
    }
}

// Helper functions for base64 encoding/decoding
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &byte) in chunk.iter().enumerate() {
            buf[i] = byte;
        }

        let b1 = buf[0] >> 2;
        let b2 = ((buf[0] & 0x03) << 4) | (buf[1] >> 4);
        let b3 = ((buf[1] & 0x0f) << 2) | (buf[2] >> 6);
        let b4 = buf[2] & 0x3f;

        result.push(CHARS[b1 as usize] as char);
        result.push(CHARS[b2 as usize] as char);
        result.push(if chunk.len() > 1 {
            CHARS[b3 as usize] as char
        } else {
            '='
        });
        result.push(if chunk.len() > 2 {
            CHARS[b4 as usize] as char
        } else {
            '='
        });
    }

    result
}

fn base64_decode(encoded: &str) -> Result<Vec<u8>, JsValue> {
    let mut lookup = [255u8; 256];
    for (i, &c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .iter()
        .enumerate()
    {
        lookup[c as usize] = i as u8;
    }

    let mut decoded = Vec::new();
    let chars: Vec<u8> = encoded
        .bytes()
        .filter(|&c| c != b'=' && c != b'\n' && c != b'\r')
        .collect();

    for chunk in chars.chunks(4) {
        if chunk.len() < 4 {
            break;
        }

        let b1 = lookup[chunk[0] as usize];
        let b2 = lookup[chunk[1] as usize];
        let b3 = lookup[chunk[2] as usize];
        let b4 = lookup[chunk[3] as usize];

        if b1 == 255 || b2 == 255 {
            return Err(JsValue::from_str("Invalid base64"));
        }

        decoded.push((b1 << 2) | (b2 >> 4));

        if b3 != 255 {
            decoded.push((b2 << 4) | (b3 >> 2));
            if b4 != 255 {
                decoded.push((b3 << 6) | b4);
            }
        }
    }

    Ok(decoded)
}

fn window() -> Result<web_sys::Window, JsValue> {
    web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))
}

#[cfg(test)]
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_model_compression() {
        let weights = vec![1.0, 1.0, 1.0, 2.0, 3.0];
        let compressed = WasmModelCompression::compress_weights(weights.clone());
        let decompressed = WasmModelCompression::decompress_weights(compressed);

        assert_eq!(weights, decompressed);
    }

    #[wasm_bindgen_test]
    fn test_quantization() {
        let weights = vec![1.0, -1.0, 0.5, -0.5];
        let quantized = WasmModelCompression::quantize_weights(weights.clone(), 8);
        let dequantized = WasmModelCompression::dequantize_weights(quantized);

        assert_eq!(dequantized.len(), weights.len());
        // Check that values are approximately equal (quantization introduces error)
        for (orig, deq) in weights.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[wasm_bindgen_test]
    fn test_base64() {
        let data = b"Hello, WASM!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    #[wasm_bindgen_test]
    fn test_progress_tracker() {
        let mut tracker = WasmProgressTracker::new(100, "Loading model".to_string());

        assert_eq!(tracker.progress_percent(), 0.0);
        assert!(!tracker.is_complete());

        tracker.update(50);
        assert_eq!(tracker.progress_percent(), 50.0);

        tracker.update(100);
        assert_eq!(tracker.progress_percent(), 100.0);
        assert!(tracker.is_complete());
    }
}
