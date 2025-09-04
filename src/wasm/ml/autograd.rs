//! Simplified automatic differentiation for WebAssembly
//! WebAssembly向け簡素化自動微分

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct VariableWasm {
    data: Vec<f64>,
    shape: Vec<usize>,
    grad: Option<Vec<f64>>,
    requires_grad: bool,
    grad_fn_type: Option<String>,
    grad_fn_inputs: Vec<String>, // Input variable IDs for gradient computation
}

#[wasm_bindgen]
impl VariableWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> VariableWasm {
        VariableWasm {
            data,
            shape,
            grad: None,
            requires_grad,
            grad_fn_type: None,
            grad_fn_inputs: Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }

    #[wasm_bindgen]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[wasm_bindgen]
    pub fn grad(&self) -> Option<Vec<f64>> {
        self.grad.clone()
    }

    #[wasm_bindgen]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[wasm_bindgen]
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = None;
        }
    }

    #[wasm_bindgen]
    pub fn backward(&mut self) {
        if !self.requires_grad {
            return;
        }

        // Initialize gradient with ones (scalar case)
        if self.grad.is_none() {
            self.grad = Some(vec![1.0; self.data.len()]);
        }
    }

    #[wasm_bindgen]
    pub fn sum(&self) -> VariableWasm {
        let sum_value = self.data.iter().sum::<f64>();
        let mut result = VariableWasm::new(vec![sum_value], vec![1], self.requires_grad);

        if self.requires_grad {
            result.grad_fn_type = Some("Sum".to_string());
        }

        result
    }

    #[wasm_bindgen]
    pub fn mean(&self) -> VariableWasm {
        let mean_value = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let mut result = VariableWasm::new(vec![mean_value], vec![1], self.requires_grad);

        if self.requires_grad {
            result.grad_fn_type = Some("Mean".to_string());
        }

        result
    }

    #[wasm_bindgen]
    pub fn pow(&self, exponent: f64) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter().map(|&x| x.powf(exponent)).collect();
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);

        if self.requires_grad {
            result.grad_fn_type = Some("Pow".to_string());
        }

        result
    }
}

#[wasm_bindgen]
pub struct ComputationGraphWasm {
    variables: HashMap<String, VariableWasm>,
    variable_counter: u32,
}

#[wasm_bindgen]
impl ComputationGraphWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ComputationGraphWasm {
        ComputationGraphWasm {
            variables: HashMap::new(),
            variable_counter: 0,
        }
    }

    #[wasm_bindgen]
    pub fn create_variable(
        &mut self,
        data: Vec<f64>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> String {
        let id = format!("var_{}", self.variable_counter);
        self.variable_counter += 1;

        let variable = VariableWasm::new(data, shape, requires_grad);
        self.variables.insert(id.clone(), variable);

        id
    }

    #[wasm_bindgen]
    pub fn get_variable_data(&self, id: &str) -> Option<Vec<f64>> {
        self.variables.get(id).map(|v| v.data())
    }

    #[wasm_bindgen]
    pub fn get_variable_grad(&self, id: &str) -> Option<Vec<f64>> {
        self.variables.get(id).and_then(|v| v.grad())
    }

    #[wasm_bindgen]
    pub fn add_variables(&mut self, id1: &str, id2: &str) -> Option<String> {
        let var1 = self.variables.get(id1)?;
        let var2 = self.variables.get(id2)?;

        if var1.data.len() != var2.data.len() {
            return None;
        }

        let result_data: Vec<f64> = var1
            .data
            .iter()
            .zip(var2.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let requires_grad = var1.requires_grad || var2.requires_grad;
        let mut result = VariableWasm::new(result_data, var1.shape.clone(), requires_grad);

        if requires_grad {
            result.grad_fn_type = Some("Add".to_string());
            result.grad_fn_inputs = vec![id1.to_string(), id2.to_string()];
        }

        let result_id = format!("var_{}", self.variable_counter);
        self.variable_counter += 1;
        self.variables.insert(result_id.clone(), result);

        Some(result_id)
    }

    #[wasm_bindgen]
    pub fn mul_variables(&mut self, id1: &str, id2: &str) -> Option<String> {
        let var1 = self.variables.get(id1)?;
        let var2 = self.variables.get(id2)?;

        if var1.data.len() != var2.data.len() {
            return None;
        }

        let result_data: Vec<f64> = var1
            .data
            .iter()
            .zip(var2.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        let requires_grad = var1.requires_grad || var2.requires_grad;
        let mut result = VariableWasm::new(result_data, var1.shape.clone(), requires_grad);

        if requires_grad {
            result.grad_fn_type = Some("Mul".to_string());
            result.grad_fn_inputs = vec![id1.to_string(), id2.to_string()];
        }

        let result_id = format!("var_{}", self.variable_counter);
        self.variable_counter += 1;
        self.variables.insert(result_id.clone(), result);

        Some(result_id)
    }

    #[wasm_bindgen]
    pub fn backward(&mut self, id: &str) {
        if let Some(variable) = self.variables.get_mut(id) {
            variable.backward();
        }
    }

    #[wasm_bindgen]
    pub fn zero_grad_all(&mut self) {
        for variable in self.variables.values_mut() {
            variable.zero_grad();
        }
    }

    #[wasm_bindgen]
    pub fn clear_graph(&mut self) {
        self.variables.clear();
        self.variable_counter = 0;
    }

    #[wasm_bindgen]
    pub fn variable_count(&self) -> u32 {
        self.variables.len() as u32
    }
}

// Simple neural network layer for WASM / WASM向け簡単なニューラルネットワーク層
#[wasm_bindgen]
pub struct LinearLayerWasm {
    weights: Vec<f64>,
    bias: Vec<f64>,
    input_size: usize,
    output_size: usize,
}

#[wasm_bindgen]
impl LinearLayerWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(input_size: usize, output_size: usize) -> LinearLayerWasm {
        // Xavier initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights: Vec<f64> = (0..input_size * output_size)
            .map(|_| {
                // Simple random initialization for WASM
                let random = (js_sys::Math::random() - 0.5) * 2.0;
                random * scale
            })
            .collect();

        let bias = vec![0.0; output_size];

        LinearLayerWasm {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    #[wasm_bindgen]
    pub fn forward(&self, input: &[f64]) -> Option<Vec<f64>> {
        if input.len() != self.input_size {
            return None;
        }

        let mut output = vec![0.0; self.output_size];

        for i in 0..self.output_size {
            let mut sum = self.bias[i];
            for j in 0..self.input_size {
                sum += input[j] * self.weights[i * self.input_size + j];
            }
            output[i] = sum;
        }

        Some(output)
    }

    #[wasm_bindgen]
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    #[wasm_bindgen]
    pub fn get_bias(&self) -> Vec<f64> {
        self.bias.clone()
    }

    #[wasm_bindgen]
    pub fn update_weights(&mut self, new_weights: Vec<f64>) -> bool {
        if new_weights.len() == self.weights.len() {
            self.weights = new_weights;
            true
        } else {
            false
        }
    }

    #[wasm_bindgen]
    pub fn update_bias(&mut self, new_bias: Vec<f64>) -> bool {
        if new_bias.len() == self.bias.len() {
            self.bias = new_bias;
            true
        } else {
            false
        }
    }
}

// Activation functions for WASM / WASM向け活性化関数
#[wasm_bindgen]
pub fn relu_wasm(x: f64) -> f64 {
    x.max(0.0)
}

#[wasm_bindgen]
pub fn relu_array_wasm(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| relu_wasm(x)).collect()
}

#[wasm_bindgen]
pub fn sigmoid_wasm(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[wasm_bindgen]
pub fn sigmoid_array_wasm(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| sigmoid_wasm(x)).collect()
}

#[wasm_bindgen]
pub fn tanh_wasm(x: f64) -> f64 {
    x.tanh()
}

#[wasm_bindgen]
pub fn tanh_array_wasm(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| tanh_wasm(x)).collect()
}

#[wasm_bindgen]
pub fn softmax_wasm(values: &[f64]) -> Vec<f64> {
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum_exp).collect()
}

/// Enhanced WASM autograd operations
/// 拡張WASM自動微分演算
#[wasm_bindgen]
impl VariableWasm {
    /// Element-wise addition with another variable
    /// 他の変数との要素ごと加算
    #[wasm_bindgen]
    pub fn add(&self, other: &VariableWasm) -> Option<VariableWasm> {
        if self.data.len() != other.data.len() {
            return None;
        }
        
        let result_data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = VariableWasm::new(result_data, self.shape.clone(), requires_grad);
        
        if requires_grad {
            result.grad_fn_type = Some("Add".to_string());
        }
        
        Some(result)
    }
    
    /// Element-wise subtraction
    /// 要素ごと減算
    #[wasm_bindgen]
    pub fn sub(&self, other: &VariableWasm) -> Option<VariableWasm> {
        if self.data.len() != other.data.len() {
            return None;
        }
        
        let result_data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = VariableWasm::new(result_data, self.shape.clone(), requires_grad);
        
        if requires_grad {
            result.grad_fn_type = Some("Sub".to_string());
        }
        
        Some(result)
    }
    
    /// Element-wise multiplication
    /// 要素ごと乗算
    #[wasm_bindgen]
    pub fn mul(&self, other: &VariableWasm) -> Option<VariableWasm> {
        if self.data.len() != other.data.len() {
            return None;
        }
        
        let result_data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = VariableWasm::new(result_data, self.shape.clone(), requires_grad);
        
        if requires_grad {
            result.grad_fn_type = Some("Mul".to_string());
        }
        
        Some(result)
    }
    
    /// Matrix multiplication (2D tensors only)
    /// 行列乗算（2次元テンソルのみ）
    #[wasm_bindgen]
    pub fn matmul(&self, other: &VariableWasm) -> Option<VariableWasm> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return None;
        }
        
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k1 != k2 {
            return None;
        }
        
        let mut result_data = vec![0.0; m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += self.data[i * k1 + k] * other.data[k * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = VariableWasm::new(result_data, vec![m, n], requires_grad);
        
        if requires_grad {
            result.grad_fn_type = Some("MatMul".to_string());
        }
        
        Some(result)
    }
    
    /// Apply ReLU activation
    /// ReLU活性化関数を適用
    #[wasm_bindgen]
    pub fn relu(&self) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| x.max(0.0))
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("ReLU".to_string());
        }
        
        result
    }
    
    /// Apply Sigmoid activation
    /// Sigmoid活性化関数を適用
    #[wasm_bindgen]
    pub fn sigmoid(&self) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| sigmoid_wasm(x))
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Sigmoid".to_string());
        }
        
        result
    }
    
    /// Apply Tanh activation
    /// Tanh活性化関数を適用
    #[wasm_bindgen]
    pub fn tanh_activation(&self) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| tanh_wasm(x))
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Tanh".to_string());
        }
        
        result
    }
    
    /// Compute exponential
    /// 指数計算
    #[wasm_bindgen]
    pub fn exp(&self) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| x.exp())
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Exp".to_string());
        }
        
        result
    }
    
    /// Compute natural logarithm
    /// 自然対数を計算
    #[wasm_bindgen]
    pub fn log(&self) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Log".to_string());
        }
        
        result
    }
    
    /// Scale by scalar
    /// スカラーによる倍率
    #[wasm_bindgen]
    pub fn scale(&self, factor: f64) -> VariableWasm {
        let result_data: Vec<f64> = self.data.iter()
            .map(|&x| x * factor)
            .collect();
        
        let mut result = VariableWasm::new(result_data, self.shape.clone(), self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Scale".to_string());
        }
        
        result
    }
    
    /// Reshape variable
    /// 変数をリシェイプ
    #[wasm_bindgen]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<VariableWasm> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.data.len() {
            return None;
        }
        
        Some(VariableWasm {
            data: self.data.clone(),
            shape: new_shape,
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn_type: None,
            grad_fn_inputs: Vec::new(),
        })
    }
    
    /// Transpose 2D matrix
    /// 2次元行列の転置
    #[wasm_bindgen]
    pub fn transpose(&self) -> Option<VariableWasm> {
        if self.shape.len() != 2 {
            return None;
        }
        
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut result_data = vec![0.0; self.data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                result_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        
        let mut result = VariableWasm::new(result_data, vec![cols, rows], self.requires_grad);
        
        if self.requires_grad {
            result.grad_fn_type = Some("Transpose".to_string());
        }
        
        Some(result)
    }
}

/// Simple optimizer for WASM
/// WASM用簡単なオプティマイザー
#[wasm_bindgen]
pub struct WasmOptimizer {
    learning_rate: f64,
    optimizer_type: String,
}

#[wasm_bindgen]
impl WasmOptimizer {
    /// Create SGD optimizer
    /// SGDオプティマイザーを作成
    #[wasm_bindgen]
    pub fn sgd(learning_rate: f64) -> WasmOptimizer {
        WasmOptimizer {
            learning_rate,
            optimizer_type: "SGD".to_string(),
        }
    }
    
    /// Update variable parameters using computed gradients
    /// 計算された勾配を使用して変数パラメータを更新
    #[wasm_bindgen]
    pub fn step(&self, variable: &mut VariableWasm) {
        if let Some(ref grad) = variable.grad.clone() {
            for i in 0..variable.data.len() {
                variable.data[i] -= self.learning_rate * grad[i];
            }
        }
    }
    
    /// Update linear layer weights and biases
    /// 線形層の重みとバイアスを更新
    #[wasm_bindgen]
    pub fn update_layer(&self, layer: &mut LinearLayerWasm, weight_grads: Vec<f64>, bias_grads: Vec<f64>) {
        // Update weights
        for i in 0..layer.weights.len() {
            if i < weight_grads.len() {
                layer.weights[i] -= self.learning_rate * weight_grads[i];
            }
        }
        
        // Update biases
        for i in 0..layer.bias.len() {
            if i < bias_grads.len() {
                layer.bias[i] -= self.learning_rate * bias_grads[i];
            }
        }
    }
    
    /// Set learning rate
    /// 学習率を設定
    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
    
    /// Get current learning rate
    /// 現在の学習率を取得
    #[wasm_bindgen]
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}
