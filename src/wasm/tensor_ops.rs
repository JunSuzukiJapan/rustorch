//! Advanced tensor operations for WASM
//! WASM用の高度なテンソル操作

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Advanced tensor operations for neural networks
/// ニューラルネットワーク用の高度なテンソル操作
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensorOps;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensorOps {
    /// Matrix multiplication: A @ B
    /// 行列積: A @ B
    #[wasm_bindgen]
    pub fn matmul(
        a: Vec<f32>,
        a_rows: usize,
        a_cols: usize,
        b: Vec<f32>,
        b_rows: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        if a_cols != b_rows {
            panic!(
                "Matrix dimensions mismatch: {}x{} @ {}x{}",
                a_rows, a_cols, b_rows, b_cols
            );
        }

        let mut result = vec![0.0; a_rows * b_cols];

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[i * a_cols + k] * b[k * b_cols + j];
                }
                result[i * b_cols + j] = sum;
            }
        }

        result
    }

    /// Transpose a 2D matrix
    /// 2D行列の転置
    #[wasm_bindgen]
    pub fn transpose(matrix: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }

        result
    }

    /// Reshape tensor while preserving total elements
    /// 総要素数を保持しながらテンソルをリシェイプ
    #[wasm_bindgen]
    pub fn reshape(data: Vec<f32>, new_shape: Vec<usize>) -> Vec<f32> {
        let total_elements = data.len();
        let new_total: usize = new_shape.iter().product();

        if total_elements != new_total {
            panic!(
                "Cannot reshape: {} elements to {} elements",
                total_elements, new_total
            );
        }

        data // Data layout remains the same, only shape metadata changes
    }

    /// Concatenate tensors along specified axis
    /// 指定軸でテンソルを連結
    #[wasm_bindgen]
    pub fn concatenate(
        tensors: js_sys::Array,
        shapes: js_sys::Array,
        axis: usize,
    ) -> js_sys::Object {
        let mut all_data = Vec::new();
        let mut all_shapes = Vec::new();

        // Extract data and shapes
        for i in 0..tensors.length() {
            let tensor = tensors.get(i);
            let shape = shapes.get(i);

            if let Ok(tensor_array) = tensor.dyn_into::<js_sys::Array>() {
                let data: Vec<f32> = (0..tensor_array.length())
                    .map(|j| tensor_array.get(j).as_f64().unwrap_or(0.0) as f32)
                    .collect();
                all_data.push(data);
            }

            if let Ok(shape_array) = shape.dyn_into::<js_sys::Array>() {
                let shape_vec: Vec<usize> = (0..shape_array.length())
                    .map(|j| shape_array.get(j).as_f64().unwrap_or(0.0) as usize)
                    .collect();
                all_shapes.push(shape_vec);
            }
        }

        if all_data.is_empty() {
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"data".into(), &js_sys::Array::new()).unwrap();
            js_sys::Reflect::set(&result, &"shape".into(), &js_sys::Array::new()).unwrap();
            return result;
        }

        // Validate shapes (all dimensions except concat axis must match)
        let first_shape = &all_shapes[0];
        for shape in &all_shapes[1..] {
            for (dim, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if dim != axis && s1 != s2 {
                    panic!("Shapes don't match along non-concat dimensions");
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.clone();
        output_shape[axis] = all_shapes.iter().map(|s| s[axis]).sum();

        // Calculate strides for concatenation
        let mut output_data = Vec::new();

        // Simple concatenation for 2D case (most common)
        if first_shape.len() == 2 {
            if axis == 0 {
                // Concatenate along rows
                for data in all_data {
                    output_data.extend(data);
                }
            } else {
                // Concatenate along columns
                let cols = first_shape[1];
                let total_rows = first_shape[0];

                output_data = vec![0.0; output_shape.iter().product()];
                let mut col_offset = 0;

                for (tensor_idx, data) in all_data.iter().enumerate() {
                    let tensor_cols = all_shapes[tensor_idx][1];

                    for row in 0..total_rows {
                        for col in 0..tensor_cols {
                            let src_idx = row * tensor_cols + col;
                            let dst_idx = row * output_shape[1] + col_offset + col;
                            output_data[dst_idx] = data[src_idx];
                        }
                    }

                    col_offset += tensor_cols;
                }
            }
        } else {
            // For higher dimensional tensors, simple flattened concatenation
            for data in all_data {
                output_data.extend(data);
            }
        }

        let result = js_sys::Object::new();
        let data_array =
            js_sys::Array::from_iter(output_data.iter().map(|&x| JsValue::from_f64(x as f64)));
        let shape_array =
            js_sys::Array::from_iter(output_shape.iter().map(|&x| JsValue::from_f64(x as f64)));

        js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
        js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();

        result
    }

    /// Split tensor along specified axis
    /// 指定軸でテンソルを分割
    #[wasm_bindgen]
    pub fn split(
        data: Vec<f32>,
        shape: Vec<usize>,
        axis: usize,
        split_sizes: Vec<usize>,
    ) -> js_sys::Array {
        if axis >= shape.len() {
            panic!(
                "Axis {} out of bounds for shape with {} dimensions",
                axis,
                shape.len()
            );
        }

        let total_axis_size: usize = split_sizes.iter().sum();
        if total_axis_size != shape[axis] {
            panic!(
                "Split sizes sum ({}) doesn't match axis size ({})",
                total_axis_size, shape[axis]
            );
        }

        let results = js_sys::Array::new();
        let mut axis_offset = 0;

        for &split_size in &split_sizes {
            let mut split_shape = shape.clone();
            split_shape[axis] = split_size;

            let split_data = if axis == 0 && shape.len() == 2 {
                // Split along rows (simple case)
                let cols = shape[1];
                let start_idx = axis_offset * cols;
                let end_idx = start_idx + split_size * cols;
                data[start_idx..end_idx].to_vec()
            } else if axis == 1 && shape.len() == 2 {
                // Split along columns
                let rows = shape[0];
                let cols = shape[1];
                let mut split_data = Vec::with_capacity(rows * split_size);

                for row in 0..rows {
                    let row_start = row * cols + axis_offset;
                    let row_end = row_start + split_size;
                    split_data.extend_from_slice(&data[row_start..row_end]);
                }

                split_data
            } else {
                // For other cases, use simple slicing
                let elements_per_split = split_shape.iter().product::<usize>();
                let start_idx = axis_offset * elements_per_split / shape[axis] * split_size;
                let end_idx = start_idx + elements_per_split;
                data[start_idx..end_idx].to_vec()
            };

            let split_result = js_sys::Object::new();
            let data_array =
                js_sys::Array::from_iter(split_data.iter().map(|&x| JsValue::from_f64(x as f64)));
            let shape_array =
                js_sys::Array::from_iter(split_shape.iter().map(|&x| JsValue::from_f64(x as f64)));

            js_sys::Reflect::set(&split_result, &"data".into(), &data_array).unwrap();
            js_sys::Reflect::set(&split_result, &"shape".into(), &shape_array).unwrap();

            results.push(&split_result);
            axis_offset += split_size;
        }

        results
    }

    /// Compute tensor dot product (Einstein summation)
    /// テンソル内積の計算（アインシュタイン記法）
    #[wasm_bindgen]
    pub fn dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
        if a.len() != b.len() {
            panic!("Vectors must have same length for dot product");
        }

        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Element-wise operations
    /// 要素ごとの操作
    #[wasm_bindgen]
    pub fn element_wise_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Tensors must have same size for element-wise addition");
        }
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Element-wise multiplication
    /// 要素ごとの乗算
    #[wasm_bindgen]
    pub fn element_wise_mul(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Tensors must have same size for element-wise multiplication");
        }
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    /// Element-wise subtraction
    /// 要素ごとの減算
    #[wasm_bindgen]
    pub fn element_wise_sub(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Tensors must have same size for element-wise subtraction");
        }
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    /// Element-wise division
    /// 要素ごとの除算
    #[wasm_bindgen]
    pub fn element_wise_div(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
        if a.len() != b.len() {
            panic!("Tensors must have same size for element-wise division");
        }
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if y.abs() < f32::EPSILON {
                    if x >= 0.0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    }
                } else {
                    x / y
                }
            })
            .collect()
    }

    /// Reduce operations
    /// リダクション操作
    #[wasm_bindgen]
    pub fn reduce_sum(data: Vec<f32>, axis: Option<usize>, shape: Vec<usize>) -> js_sys::Object {
        let result = js_sys::Object::new();

        match axis {
            Some(ax) if ax < shape.len() => {
                // Sum along specific axis
                let mut new_shape = shape.clone();
                new_shape.remove(ax);
                let output_size = new_shape.iter().product::<usize>();
                let mut output = vec![0.0; output_size];

                if shape.len() == 2 && ax == 0 {
                    // Sum along rows
                    let cols = shape[1];
                    for col in 0..cols {
                        for row in 0..shape[0] {
                            output[col] += data[row * cols + col];
                        }
                    }
                } else if shape.len() == 2 && ax == 1 {
                    // Sum along columns
                    let cols = shape[1];
                    for row in 0..shape[0] {
                        let mut sum = 0.0;
                        for col in 0..cols {
                            sum += data[row * cols + col];
                        }
                        output[row] = sum;
                    }
                } else {
                    // General case - simplified
                    output = vec![data.iter().sum()];
                    new_shape = vec![1];
                }

                let data_array =
                    js_sys::Array::from_iter(output.iter().map(|&x| JsValue::from_f64(x as f64)));
                let shape_array = js_sys::Array::from_iter(
                    new_shape.iter().map(|&x| JsValue::from_f64(x as f64)),
                );

                js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
                js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();
            }
            _ => {
                // Sum all elements
                let sum = data.iter().sum::<f32>();
                let data_array = js_sys::Array::from_iter([JsValue::from_f64(sum as f64)]);
                let shape_array = js_sys::Array::new(); // Scalar

                js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
                js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();
            }
        }

        result
    }

    /// Reduce mean
    /// 平均値の計算
    #[wasm_bindgen]
    pub fn reduce_mean(data: Vec<f32>, axis: Option<usize>, shape: Vec<usize>) -> js_sys::Object {
        let sum_result = Self::reduce_sum(data.clone(), axis, shape.clone());

        if let Ok(sum_data) = js_sys::Reflect::get(&sum_result, &"data".into()) {
            if let Ok(sum_array) = sum_data.dyn_into::<js_sys::Array>() {
                let count = match axis {
                    Some(ax) if ax < shape.len() => shape[ax] as f32,
                    _ => data.len() as f32,
                };

                let mean_array = js_sys::Array::new();
                for i in 0..sum_array.length() {
                    let val = sum_array.get(i).as_f64().unwrap_or(0.0) as f32;
                    mean_array.push(&JsValue::from_f64((val / count) as f64));
                }

                js_sys::Reflect::set(&sum_result, &"data".into(), &mean_array).unwrap();
            }
        }

        sum_result
    }

    /// Broadcasting addition for tensors of different shapes
    /// 異なる形状のテンソルのブロードキャスト加算
    #[wasm_bindgen]
    pub fn broadcast_add(
        a: Vec<f32>,
        a_shape: Vec<usize>,
        b: Vec<f32>,
        b_shape: Vec<usize>,
    ) -> js_sys::Object {
        // Simple broadcasting for common cases
        let result = js_sys::Object::new();

        if a_shape == b_shape {
            // Same shape - direct element-wise addition
            let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
            let data_array =
                js_sys::Array::from_iter(sum.iter().map(|&x| JsValue::from_f64(x as f64)));
            let shape_array =
                js_sys::Array::from_iter(a_shape.iter().map(|&x| JsValue::from_f64(x as f64)));

            js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
            js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();
        } else if b.len() == 1 {
            // Scalar addition
            let scalar = b[0];
            let sum: Vec<f32> = a.iter().map(|&x| x + scalar).collect();
            let data_array =
                js_sys::Array::from_iter(sum.iter().map(|&x| JsValue::from_f64(x as f64)));
            let shape_array =
                js_sys::Array::from_iter(a_shape.iter().map(|&x| JsValue::from_f64(x as f64)));

            js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
            js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();
        } else if a.len() == 1 {
            // Scalar addition (reversed)
            let scalar = a[0];
            let sum: Vec<f32> = b.iter().map(|&x| scalar + x).collect();
            let data_array =
                js_sys::Array::from_iter(sum.iter().map(|&x| JsValue::from_f64(x as f64)));
            let shape_array =
                js_sys::Array::from_iter(b_shape.iter().map(|&x| JsValue::from_f64(x as f64)));

            js_sys::Reflect::set(&result, &"data".into(), &data_array).unwrap();
            js_sys::Reflect::set(&result, &"shape".into(), &shape_array).unwrap();
        } else {
            panic!("Complex broadcasting not implemented for arbitrary shapes");
        }

        result
    }

    /// Compute gradient clipping (useful for training)
    /// 勾配クリッピングを計算（訓練に有用）
    #[wasm_bindgen]
    pub fn clip_gradients(gradients: Vec<f32>, max_norm: f32) -> Vec<f32> {
        let grad_norm: f32 = gradients.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if grad_norm <= max_norm {
            gradients
        } else {
            let scale = max_norm / grad_norm;
            gradients.iter().map(|&x| x * scale).collect()
        }
    }

    /// Apply dropout during training (sets random elements to zero)
    /// 訓練中のドロップアウトを適用（ランダム要素をゼロに設定）
    #[wasm_bindgen]
    pub fn dropout(input: Vec<f32>, dropout_rate: f32, training: bool, seed: u32) -> Vec<f32> {
        if !training || dropout_rate <= 0.0 {
            return input;
        }

        let keep_prob = 1.0 - dropout_rate;
        let scale = 1.0 / keep_prob;
        let mut rng_state = seed as u64;

        input
            .into_iter()
            .map(|x| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let random_val = (rng_state % 2147483647) as f32 / 2147483647.0;

                if random_val < keep_prob {
                    x * scale
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = WasmTensorOps::matmul(a, 2, 2, b, 2, 2);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[wasm_bindgen_test]
    fn test_transpose() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let transposed = WasmTensorOps::transpose(matrix, 2, 3);

        // Expected: [[1, 4], [2, 5], [3, 6]] flattened = [1, 4, 2, 5, 3, 6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[wasm_bindgen_test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = WasmTensorOps::dot_product(a, b);

        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(dot, 32.0);
    }

    #[wasm_bindgen_test]
    fn test_element_wise_ops() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert_eq!(
            WasmTensorOps::element_wise_add(a.clone(), b.clone()),
            vec![5.0, 7.0, 9.0]
        );
        assert_eq!(
            WasmTensorOps::element_wise_mul(a.clone(), b.clone()),
            vec![4.0, 10.0, 18.0]
        );
        assert_eq!(
            WasmTensorOps::element_wise_sub(a.clone(), b.clone()),
            vec![-3.0, -3.0, -3.0]
        );
        assert_eq!(WasmTensorOps::element_wise_div(a, b), vec![0.25, 0.4, 0.5]);
    }

    #[wasm_bindgen_test]
    fn test_clip_gradients() {
        let gradients = vec![3.0, 4.0]; // Norm = 5.0
        let clipped = WasmTensorOps::clip_gradients(gradients, 2.0);

        // Should be scaled down to norm 2.0: [1.2, 1.6]
        assert!((clipped[0] - 1.2).abs() < 1e-5);
        assert!((clipped[1] - 1.6).abs() < 1e-5);
    }
}
