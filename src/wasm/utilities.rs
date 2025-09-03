//! Phase 8 Tensor Utilities WASM Bindings  
//! フェーズ８テンソルユーティリティWASMバインディング

use crate::wasm::tensor::WasmTensor;
use js_sys::{Array, Int32Array, Uint8Array};
use wasm_bindgen::prelude::*;

/// WASM bindings for Phase 8 tensor utilities
/// フェーズ８テンソルユーティリティのWASMバインディング
#[wasm_bindgen]
pub struct WasmTensorUtils;

#[wasm_bindgen]
impl WasmTensorUtils {
    /// Constructor
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmTensorUtils
    }

    // === Conditional Operations ===

    /// Select elements from x or y based on boolean condition (using Uint8Array for booleans)
    /// ブール条件に基づいてxまたはyから要素を選択（boolean用にUint8Arrayを使用）
    #[wasm_bindgen]
    pub fn where_op(
        &self,
        condition: &Uint8Array,
        x: &WasmTensor,
        y: &WasmTensor,
    ) -> Result<WasmTensor, JsValue> {
        if x.shape() != y.shape() {
            return Err(JsValue::from_str("X and Y tensors must have same shape"));
        }

        let condition_vec = condition.to_vec();
        let x_data = x.data();
        let y_data = y.data();
        let mut result_data = Vec::with_capacity(x_data.len());

        // Simple element-wise selection
        for i in 0..x_data.len() {
            let cond_idx = i % condition_vec.len();
            if condition_vec[cond_idx] != 0 {
                result_data.push(x_data[i]);
            } else {
                result_data.push(y_data[i]);
            }
        }

        Ok(WasmTensor::new(result_data, x.shape()))
    }

    /// Select elements where mask is true (using Uint8Array for boolean mask)
    /// マスクがtrueの位置の要素を選択（boolean mask用にUint8Arrayを使用）
    #[wasm_bindgen]
    pub fn masked_select(
        &self,
        input: &WasmTensor,
        mask: &Uint8Array,
    ) -> Result<WasmTensor, JsValue> {
        let input_data = input.data();
        let mask_vec = mask.to_vec();

        if input_data.len() != mask_vec.len() {
            return Err(JsValue::from_str("Input and mask must have same size"));
        }

        let selected: Vec<f32> = input_data
            .iter()
            .zip(mask_vec.iter())
            .filter_map(|(&value, &mask_val)| if mask_val != 0 { Some(value) } else { None })
            .collect();

        let selected_len = selected.len();
        Ok(WasmTensor::new(selected, vec![selected_len]))
    }

    /// Fill elements where mask is true with specified value
    /// マスクがtrueの位置を指定値で埋める
    #[wasm_bindgen]
    pub fn masked_fill(
        &self,
        input: &WasmTensor,
        mask: &Uint8Array,
        value: f32,
    ) -> Result<WasmTensor, JsValue> {
        let mut result_data = input.data();
        let mask_vec = mask.to_vec();

        if result_data.len() != mask_vec.len() {
            return Err(JsValue::from_str("Input and mask must have same size"));
        }

        for (elem, &mask_val) in result_data.iter_mut().zip(mask_vec.iter()) {
            if mask_val != 0 {
                *elem = value;
            }
        }

        Ok(WasmTensor::new(result_data, input.shape()))
    }

    // === Index Operations ===

    /// Gather values using index (1D version for WASM)
    /// インデックスを使って値を収集（WASM用1D版）
    #[wasm_bindgen]
    pub fn gather_1d(&self, input: &WasmTensor, index: &Int32Array) -> Result<WasmTensor, JsValue> {
        let input_data = input.data();
        let index_vec = index.to_vec();
        let input_len = input_data.len();

        let mut result_data = Vec::with_capacity(index_vec.len());

        for &idx in index_vec.iter() {
            if idx < 0 || idx as usize >= input_len {
                return Err(JsValue::from_str(&format!(
                    "Index {} out of bounds for size {}",
                    idx, input_len
                )));
            }
            result_data.push(input_data[idx as usize]);
        }

        Ok(WasmTensor::new(result_data, vec![index_vec.len()]))
    }

    /// Select specific indices from tensor
    /// テンソルから特定のインデックスを選択
    #[wasm_bindgen]
    pub fn index_select_1d(
        &self,
        input: &WasmTensor,
        index: &Int32Array,
    ) -> Result<WasmTensor, JsValue> {
        self.gather_1d(input, index)
    }

    // === Statistical Operations ===

    /// Find top-k largest elements
    /// 上位k個の最大要素を検索
    #[wasm_bindgen]
    pub fn topk_1d(&self, input: &WasmTensor, k: usize, largest: bool) -> Result<Array, JsValue> {
        let input_data = input.data();

        if k > input_data.len() {
            return Err(JsValue::from_str("k cannot be larger than tensor size"));
        }

        let mut indexed_values: Vec<(f32, usize)> = input_data
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        if largest {
            indexed_values
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed_values
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        let values: Vec<f32> = indexed_values.iter().take(k).map(|(val, _)| *val).collect();
        let indices: Vec<i32> = indexed_values
            .iter()
            .take(k)
            .map(|(_, idx)| *idx as i32)
            .collect();

        let result = Array::new();
        result.push(&WasmTensor::new(values, vec![k]).into());
        result.push(&Int32Array::from(indices.as_slice()).into());
        Ok(result)
    }

    /// Find k-th smallest value
    /// k番目に小さい値を検索
    #[wasm_bindgen]
    pub fn kthvalue_1d(&self, input: &WasmTensor, k: usize) -> Result<Array, JsValue> {
        let input_data = input.data();

        if k >= input_data.len() {
            return Err(JsValue::from_str("k must be less than tensor size"));
        }

        let mut indexed_values: Vec<(f32, usize)> = input_data
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();

        indexed_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let (value, original_idx) = indexed_values[k];

        let result = Array::new();
        result.push(&WasmTensor::new(vec![value], vec![1]).into());
        result.push(&Int32Array::from(&[original_idx as i32][..]).into());
        Ok(result)
    }

    /// Find unique elements
    /// 一意要素を検索
    #[wasm_bindgen]
    pub fn unique_1d(&self, input: &WasmTensor, sorted: bool) -> Result<WasmTensor, JsValue> {
        let mut input_data = input.data();

        if sorted {
            input_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut unique_values = Vec::new();

        for &value in input_data.iter() {
            if unique_values.is_empty()
                || unique_values
                    .last()
                    .map_or(true, |last: &f32| (value - *last).abs() > 1e-6_f32)
            {
                unique_values.push(value);
            }
        }

        Ok(WasmTensor::new(
            unique_values.clone(),
            vec![unique_values.len()],
        ))
    }

    /// Compute histogram
    /// ヒストグラム計算
    #[wasm_bindgen]
    pub fn histogram_simple(
        &self,
        input: &WasmTensor,
        bins: usize,
        min_val: f32,
        max_val: f32,
    ) -> Result<Array, JsValue> {
        if bins == 0 {
            return Err(JsValue::from_str("Number of bins must be positive"));
        }

        if min_val >= max_val {
            return Err(JsValue::from_str("min_val must be less than max_val"));
        }

        let input_data = input.data();
        let bin_width = (max_val - min_val) / bins as f32;
        let mut bin_counts = vec![0i32; bins];

        for &value in input_data.iter() {
            if value >= min_val && value <= max_val {
                let bin_idx = if value == max_val {
                    bins - 1
                } else {
                    ((value - min_val) / bin_width).floor() as usize
                };
                if bin_idx < bins {
                    bin_counts[bin_idx] += 1;
                }
            }
        }

        // Create bin edges
        let mut bin_edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            bin_edges.push(min_val + i as f32 * bin_width);
        }

        let result = Array::new();
        result.push(&Int32Array::from(bin_counts.as_slice()).into());
        result.push(&WasmTensor::new(bin_edges, vec![bins + 1]).into());
        Ok(result)
    }
}

/// Convenience functions for browser usage
/// ブラウザ使用のための便利関数

/// Simple where operation for browser ML
/// ブラウザML用シンプルwhere操作
#[wasm_bindgen]
pub fn tensor_where_simple(
    condition: &Uint8Array,
    x: &WasmTensor,
    y: &WasmTensor,
) -> Result<WasmTensor, JsValue> {
    let utils = WasmTensorUtils::new();
    utils.where_op(condition, x, y)
}

/// Simple masked select for browser ML
/// ブラウザML用シンプルマスク選択
#[wasm_bindgen]
pub fn tensor_masked_select_simple(
    input: &WasmTensor,
    mask: &Uint8Array,
) -> Result<WasmTensor, JsValue> {
    let utils = WasmTensorUtils::new();
    utils.masked_select(input, mask)
}

/// Simple topk for browser ML
/// ブラウザML用シンプルトップk
#[wasm_bindgen]
pub fn tensor_topk_simple(input: &WasmTensor, k: usize) -> Result<Array, JsValue> {
    let utils = WasmTensorUtils::new();
    utils.topk_1d(input, k, true)
}

/// Simple unique for browser ML
/// ブラウザML用シンプル一意
#[wasm_bindgen]
pub fn tensor_unique_simple(input: &WasmTensor) -> Result<WasmTensor, JsValue> {
    let utils = WasmTensorUtils::new();
    utils.unique_1d(input, true)
}
