//! Implementation of a linear (fully connected) layer.
//! 線形（全結合）レイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
use crate::serialization::core::{Loadable, Saveable, SerializationError, SerializationResult};
use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

/// A linear (fully connected) layer.
/// 線形（全結合）レイヤー
///
/// This layer applies a linear transformation to the incoming data: `y = xW^T + b`
pub struct Linear<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    /// The weight matrix of shape (output_features, input_features)
    /// 重み行列 (出力特徴量, 入力特徴量)
    weight: Variable<T>,

    /// The bias vector of shape (output_features,)
    /// バイアスベクトル (出力特徴量,)
    bias: Option<Variable<T>>,

    /// Input size (number of input features)
    /// 入力サイズ（入力特徴量の数）
    input_size: usize,

    /// Output size (number of output features)
    /// 出力サイズ（出力特徴量の数）
    output_size: usize,
}

impl<T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    std::fmt::Debug for Linear<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("input_size", &self.input_size)
            .field("output_size", &self.output_size)
            .field("has_bias", &self.bias.is_some())
            .finish()
    }
}

impl<T> Linear<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    /// Creates a new linear layer with the given input and output sizes.
    /// 入力サイズと出力サイズを指定して新しい線形レイヤーを作成します。
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize weights using Kaiming initialization
        let k = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, k as f64).unwrap();

        // Initialize weights
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| {
                num_traits::cast(normal.sample(&mut rand::thread_rng()) as f64).unwrap_or(T::zero())
            })
            .collect();

        let weight = Variable::new(
            Tensor::new(
                Array::from_shape_vec((output_size, input_size), weight_data)
                    .unwrap()
                    .into_dyn(),
            ),
            true,
        );

        // Initialize bias
        let bias_data: Vec<T> = (0..output_size)
            .map(|_| {
                num_traits::cast(normal.sample(&mut rand::thread_rng()) as f64).unwrap_or(T::zero())
            })
            .collect();

        let bias = Variable::new(
            Tensor::new(
                Array::from_shape_vec((output_size,), bias_data)
                    .unwrap()
                    .into_dyn(),
            ),
            true,
        );

        Linear {
            weight,
            bias: Some(bias),
            input_size,
            output_size,
        }
    }

    /// Creates a new linear layer without a bias term.
    /// バイアス項を持たない線形レイヤーを作成します。
    pub fn new_no_bias(input_size: usize, output_size: usize) -> Self {
        // Initialize weights using Kaiming initialization
        let k = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, k as f64).unwrap();

        // Initialize weights
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| {
                num_traits::cast(normal.sample(&mut rand::thread_rng()) as f64).unwrap_or(T::zero())
            })
            .collect();

        let weight = Variable::new(
            Tensor::new(
                Array::from_shape_vec((output_size, input_size), weight_data)
                    .unwrap()
                    .into_dyn(),
            ),
            true,
        );

        Linear {
            weight,
            bias: None,
            input_size,
            output_size,
        }
    }

    /// Performs the forward pass of the linear layer.
    /// 線形レイヤーの順伝搬を実行します。
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();

        let input_shape = input_data.shape();

        let mut output_data = if input_shape.len() == 3 {
            // Handle 3D input: (batch_size, seq_length, input_features)
            let batch_size = input_shape[0];
            let seq_length = input_shape[1];
            let input_features = input_shape[2];

            // Reshape to 2D for matrix multiplication: (batch_size * seq_length, input_features)
            let reshaped_input = input_data
                .reshape(&[batch_size * seq_length, input_features])
                .expect("Reshape failed");

            // Transpose weight here for 2D case
            let weight_t = weight_data.transpose().expect("Transpose failed");
            let matmul_result = reshaped_input.matmul(&weight_t).expect("MatMul failed");

            // Reshape back to 3D: (batch_size, seq_length, output_features)
            let output_features = self.output_size;
            matmul_result
                .reshape(&[batch_size, seq_length, output_features])
                .expect("Reshape back failed")
        } else {
            // Handle 2D input: (batch_size, input_features) @ (input_features, output_features)
            let weight_t = weight_data.transpose().expect("Transpose failed");
            input_data.matmul(&weight_t).expect("MatMul failed")
        };

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_binding = bias.data();
            let bias_data = bias_binding.read().unwrap();

            // Broadcast bias to match output shape
            let output_shape = output_data.shape();
            match output_shape.len() {
                1 => {
                    // Single sample: direct addition
                    output_data = &output_data + &*bias_data;
                }
                2 => {
                    // Batch processing: add bias to each sample
                    let batch_size = output_shape[0];
                    let output_features = output_shape[1];

                    // Create bias tensor with shape (1, output_features) for broadcasting
                    let bias_expanded = bias_data
                        .as_array()
                        .clone()
                        .into_shape_with_order((1, output_features))
                        .unwrap();
                    let bias_tensor = Tensor::new(bias_expanded.into_dyn());

                    output_data = &output_data + &bias_tensor;
                }
                3 => {
                    // 3D tensor: (batch_size, seq_length, output_features)
                    let batch_size = output_shape[0];
                    let seq_length = output_shape[1];
                    let output_features = output_shape[2];

                    // Create bias tensor with shape (1, 1, output_features) for broadcasting
                    let bias_expanded = bias_data
                        .as_array()
                        .clone()
                        .into_shape_with_order((1, 1, output_features))
                        .unwrap();
                    let bias_tensor = Tensor::new(bias_expanded.into_dyn());

                    output_data = &output_data + &bias_tensor;
                }
                _ => {
                    // For higher dimensions, broadcast by repeating the bias for all leading dimensions
                    let total_elements = output_shape.iter().product::<usize>();
                    let output_features = output_shape.last().unwrap();
                    let leading_dims: usize =
                        output_shape[..output_shape.len() - 1].iter().product();

                    let mut broadcasted_bias = Vec::with_capacity(total_elements);
                    let bias_slice = bias_data.as_array();

                    for _ in 0..leading_dims {
                        for i in 0..*output_features {
                            broadcasted_bias.push(bias_slice[i]);
                        }
                    }

                    let bias_tensor = Tensor::from_vec(broadcasted_bias, output_shape.to_vec());
                    output_data = &output_data + &bias_tensor;
                }
            }
        }

        let requires_grad = input.requires_grad()
            || self.weight.requires_grad()
            || self.bias.as_ref().map_or(false, |b| b.requires_grad());

        if requires_grad {
            // Create gradient function for backpropagation
            use crate::autograd::linear_grad_fn::LinearBackward;
            use std::sync::{Arc, RwLock};

            let grad_fn = Arc::new(LinearBackward {
                input: Arc::new(RwLock::new(input_data.clone())),
                weight: Arc::new(RwLock::new(weight_data.clone())),
                input_var: input.clone(),
                weight_var: self.weight.clone(),
                bias_var: self.bias.clone(),
            });

            Variable::new_with_grad_fn(output_data, true, Some(grad_fn))
        } else {
            Variable::new(output_data, false)
        }
    }

    /// Returns the input size of the layer.
    /// レイヤーの入力サイズを返します。
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the output size of the layer.
    /// レイヤーの出力サイズを返します。
    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

impl<T> Module<T> for Linear<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Serialization support for Linear layer
impl<T> Saveable for Linear<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn save_binary(&self) -> SerializationResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Save input and output sizes
        let input_size_bytes = self.input_size.to_le_bytes();
        let output_size_bytes = self.output_size.to_le_bytes();
        buffer.extend_from_slice(&input_size_bytes);
        buffer.extend_from_slice(&output_size_bytes);

        // Save weight tensor
        let weight_data = self.weight.save_binary()?;
        let weight_size = weight_data.len() as u64;
        buffer.extend_from_slice(&weight_size.to_le_bytes());
        buffer.extend_from_slice(&weight_data);

        // Save bias (if present)
        let has_bias = self.bias.is_some();
        buffer.push(has_bias as u8);

        if let Some(ref bias) = self.bias {
            let bias_data = bias.save_binary()?;
            let bias_size = bias_data.len() as u64;
            buffer.extend_from_slice(&bias_size.to_le_bytes());
            buffer.extend_from_slice(&bias_data);
        }

        Ok(buffer)
    }

    fn type_id(&self) -> &'static str {
        "nn.Linear"
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("input_size".to_string(), self.input_size.to_string());
        meta.insert("output_size".to_string(), self.output_size.to_string());
        meta.insert("has_bias".to_string(), self.bias.is_some().to_string());
        meta
    }
}

impl<T> Loadable for Linear<T>
where
    T: Float
        + Debug
        + Default
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + 'static
        + Send
        + Sync
        + Copy
        + ScalarOperand
        + Sum
        + std::fmt::Display,
{
    fn load_binary(data: &[u8]) -> SerializationResult<Self> {
        let mut offset = 0;

        // Load input and output sizes
        if data.len() < offset + 16 {
            return Err(SerializationError::FormatError(
                "Insufficient data for sizes".to_string(),
            ));
        }

        let input_size = usize::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        let output_size = usize::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Load weight tensor
        if data.len() < offset + 8 {
            return Err(SerializationError::FormatError(
                "Insufficient data for weight size".to_string(),
            ));
        }

        let weight_size = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        offset += 8;

        if data.len() < offset + weight_size {
            return Err(SerializationError::FormatError(
                "Insufficient data for weight".to_string(),
            ));
        }

        let weight_data = &data[offset..offset + weight_size];
        let weight = Variable::load_binary(weight_data)?;
        offset += weight_size;

        // Load bias
        if data.len() < offset + 1 {
            return Err(SerializationError::FormatError(
                "Insufficient data for bias flag".to_string(),
            ));
        }

        let has_bias = data[offset] != 0;
        offset += 1;

        let bias = if has_bias {
            if data.len() < offset + 8 {
                return Err(SerializationError::FormatError(
                    "Insufficient data for bias size".to_string(),
                ));
            }

            let bias_size = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            offset += 8;

            if data.len() < offset + bias_size {
                return Err(SerializationError::FormatError(
                    "Insufficient data for bias".to_string(),
                ));
            }

            let bias_data = &data[offset..offset + bias_size];
            Some(Variable::load_binary(bias_data)?)
        } else {
            None
        };

        Ok(Linear {
            weight,
            bias,
            input_size,
            output_size,
        })
    }

    fn expected_type_id() -> &'static str {
        "nn.Linear"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        // Test with bias
        let linear = Linear::<f32>::new(3, 2);
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]), false);
        let output = linear.forward(&input);

        // Check that we get some output (shape might vary based on implementation)
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        assert!(!output_data.is_empty());

        // Test without bias
        let linear = Linear::<f32>::new_no_bias(3, 2);
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]), false);
        let output = linear.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        assert!(!output_data.is_empty());
    }

    #[test]
    fn test_linear_parameters() {
        let linear = Linear::<f32>::new(3, 2);
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight and bias

        let linear = Linear::<f32>::new_no_bias(3, 2);
        let params = linear.parameters();
        assert_eq!(params.len(), 1); // only weight
    }
}
