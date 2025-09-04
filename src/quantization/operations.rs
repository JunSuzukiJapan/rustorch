//! Quantized tensor operations and arithmetic
//! 量子化テンソル演算と算術

use super::types::{QuantizableInteger, QuantizedTensor};
use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, Zip};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use std::ops::{Add, Div, Mul, Sub};

/// Trait for quantized tensor operations
/// 量子化テンソル演算のトレイト
pub trait QuantizedOps<Q: QuantizableInteger> {
    /// Add two quantized tensors
    /// 二つの量子化テンソルを加算
    fn qadd(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Subtract two quantized tensors
    /// 二つの量子化テンソルを減算
    fn qsub(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Multiply two quantized tensors
    /// 二つの量子化テンソルを乗算
    fn qmul(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Matrix multiplication for quantized tensors
    /// 量子化テンソルの行列乗算
    fn qmatmul(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Quantized ReLU activation
    /// 量子化ReLU活性化関数
    fn qrelu(&self) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Add scalar to quantized tensor
    /// 量子化テンソルにスカラーを加算
    fn qadd_scalar(&self, scalar: f32) -> RusTorchResult<QuantizedTensor<Q>>;

    /// Multiply quantized tensor by scalar
    /// 量子化テンソルをスカラーで乗算
    fn qmul_scalar(&self, scalar: f32) -> RusTorchResult<QuantizedTensor<Q>>;
}

impl<Q: QuantizableInteger> QuantizedOps<Q> for QuantizedTensor<Q> {
    fn qadd(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>> {
        // Check shape compatibility
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        // For different quantization parameters, we need to requantize
        // 異なる量子化パラメータの場合、再量子化が必要
        let (result_scale, result_zero_point) = if self.is_compatible_with(other) {
            // Same quantization parameters - direct addition
            // 同じ量子化パラメータ - 直接加算
            (self.scale, self.zero_point)
        } else {
            // Different parameters - compute new scale
            // 異なるパラメータ - 新しいスケールを計算
            compute_output_quantization_params(
                (self.scale, self.zero_point),
                (other.scale, other.zero_point),
                QuantizedOperation::Add,
            )?
        };

        let result_data = if self.is_compatible_with(other) {
            // Direct integer addition when scales match
            // スケールが一致する場合の直接整数加算
            Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| {
                    let sum = QuantizableInteger::to_i32(&a)
                        .saturating_add(QuantizableInteger::to_i32(&b))
                        .saturating_sub(self.zero_point);
                    Q::from_i32_clamped(sum)
                })
        } else {
            // Mixed precision addition with requantization
            // 再量子化を伴う混合精度加算
            Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| {
                    // Dequantize both values
                    let a_fp =
                        (QuantizableInteger::to_i32(&a) - self.zero_point) as f32 * self.scale;
                    let b_fp =
                        (QuantizableInteger::to_i32(&b) - other.zero_point) as f32 * other.scale;

                    // Add in floating point
                    let sum_fp = a_fp + b_fp;

                    // Requantize to result parameters
                    let quantized = (sum_fp / result_scale).round() as i32 + result_zero_point;
                    Q::from_i32_clamped(quantized)
                })
        };

        Ok(QuantizedTensor::new(
            result_data,
            result_scale,
            result_zero_point,
            self.device.clone(),
        ))
    }

    fn qsub(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>> {
        // Similar to addition but with subtraction
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        let (result_scale, result_zero_point) = if self.is_compatible_with(other) {
            (self.scale, self.zero_point)
        } else {
            compute_output_quantization_params(
                (self.scale, self.zero_point),
                (other.scale, other.zero_point),
                QuantizedOperation::Sub,
            )?
        };

        let result_data = if self.is_compatible_with(other) {
            Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| {
                    let diff = QuantizableInteger::to_i32(&a)
                        .saturating_sub(QuantizableInteger::to_i32(&b))
                        .saturating_add(self.zero_point);
                    Q::from_i32_clamped(diff)
                })
        } else {
            Zip::from(&self.data)
                .and(&other.data)
                .map_collect(|&a, &b| {
                    let a_fp =
                        (QuantizableInteger::to_i32(&a) - self.zero_point) as f32 * self.scale;
                    let b_fp =
                        (QuantizableInteger::to_i32(&b) - other.zero_point) as f32 * other.scale;
                    let diff_fp = a_fp - b_fp;
                    let quantized = (diff_fp / result_scale).round() as i32 + result_zero_point;
                    Q::from_i32_clamped(quantized)
                })
        };

        Ok(QuantizedTensor::new(
            result_data,
            result_scale,
            result_zero_point,
            self.device.clone(),
        ))
    }

    fn qmul(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        // For multiplication, the output scale is the product of input scales
        // 乗算の場合、出力スケールは入力スケールの積
        let result_scale = self.scale * other.scale;
        let result_zero_point = 0; // Multiplication typically uses zero_point = 0

        let result_data = Zip::from(&self.data)
            .and(&other.data)
            .map_collect(|&a, &b| {
                // Perform multiplication with proper zero point handling
                // 適切なゼロポイント処理で乗算を実行
                let a_adjusted = QuantizableInteger::to_i32(&a) - self.zero_point;
                let b_adjusted = QuantizableInteger::to_i32(&b) - other.zero_point;
                let product = a_adjusted * b_adjusted;

                // Scale down to target precision
                // 目標精度にスケールダウン
                let scaled_product =
                    (product as f32 / (self.scale * other.scale / result_scale)).round() as i32;
                Q::from_i32_clamped(scaled_product)
            });

        Ok(QuantizedTensor::new(
            result_data,
            result_scale,
            result_zero_point,
            self.device.clone(),
        ))
    }

    fn qmatmul(&self, other: &QuantizedTensor<Q>) -> RusTorchResult<QuantizedTensor<Q>> {
        // Check matrix multiplication compatibility
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape.len() < 2 || other_shape.len() < 2 {
            return Err(RusTorchError::TensorOp {
                message: "Matrix multiplication requires at least 2D tensors".to_string(),
                source: None,
            });
        }

        let self_cols = self_shape[self_shape.len() - 1];
        let other_rows = other_shape[other_shape.len() - 2];

        if self_cols != other_rows {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![self_cols],
                actual: vec![other_rows],
            });
        }

        // Simplified 2D matrix multiplication for now
        // 現在は簡略化された2D行列乗算
        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(RusTorchError::TensorOp {
                message: "Only 2D matrix multiplication currently supported".to_string(),
                source: None,
            });
        }

        let m = self_shape[0];
        let k = self_shape[1];
        let n = other_shape[1];

        // Output scale is product of input scales
        // 出力スケールは入力スケールの積
        let result_scale = self.scale * other.scale;
        let result_zero_point = 0;

        let mut result_data = ArrayD::zeros(vec![m, n]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i64; // Use i64 to prevent overflow

                for l in 0..k {
                    let a_val = QuantizableInteger::to_i32(&self.data[[i, l]]) - self.zero_point;
                    let b_val = QuantizableInteger::to_i32(&other.data[[l, j]]) - other.zero_point;
                    sum += (a_val as i64) * (b_val as i64);
                }

                // Scale and quantize result
                let fp_result = sum as f32 * result_scale;
                let quantized = (fp_result / result_scale).round() as i32;
                result_data[[i, j]] = Q::from_i32_clamped(quantized);
            }
        }

        Ok(QuantizedTensor::new(
            result_data,
            result_scale,
            result_zero_point,
            self.device.clone(),
        ))
    }

    fn qrelu(&self) -> RusTorchResult<QuantizedTensor<Q>> {
        // ReLU: max(0, x) in quantized space
        // ReLU: 量子化空間でのmax(0, x)
        let zero_quantized = Q::from_i32_clamped(self.zero_point);

        let result_data = self.data.mapv(|val| {
            if QuantizableInteger::to_i32(&val) > self.zero_point {
                val
            } else {
                zero_quantized
            }
        });

        Ok(QuantizedTensor::new(
            result_data,
            self.scale,
            self.zero_point,
            self.device.clone(),
        ))
    }

    fn qadd_scalar(&self, scalar: f32) -> RusTorchResult<QuantizedTensor<Q>> {
        // Quantize scalar with same parameters
        // 同じパラメータでスカラーを量子化
        let scalar_quantized = (scalar / self.scale).round() as i32 + self.zero_point;
        let scalar_clamped = Q::from_i32_clamped(scalar_quantized);

        let result_data = self.data.mapv(|val| {
            let sum = QuantizableInteger::to_i32(&val)
                .saturating_add(QuantizableInteger::to_i32(&scalar_clamped))
                .saturating_sub(self.zero_point);
            Q::from_i32_clamped(sum)
        });

        Ok(QuantizedTensor::new(
            result_data,
            self.scale,
            self.zero_point,
            self.device.clone(),
        ))
    }

    fn qmul_scalar(&self, scalar: f32) -> RusTorchResult<QuantizedTensor<Q>> {
        // For scalar multiplication, scale changes
        // スカラー乗算の場合、スケールが変更される
        let new_scale = self.scale * scalar.abs();

        let result_data = self.data.mapv(|val| {
            let adjusted = QuantizableInteger::to_i32(&val) - self.zero_point;
            let scaled = (adjusted as f32 * scalar).round() as i32;
            Q::from_i32_clamped(scaled)
        });

        Ok(QuantizedTensor::new(
            result_data,
            new_scale,
            0, // Zero point becomes 0 after scaling
            self.device.clone(),
        ))
    }
}

/// Trait for dequantization operations
/// 非量子化演算のトレイト
pub trait DequantizeOps<Q: QuantizableInteger> {
    /// Dequantize to f32 tensor
    /// f32テンソルに非量子化
    fn dequantize_f32(&self) -> ArrayD<f32>;

    /// Dequantize to f64 tensor
    /// f64テンソルに非量子化
    fn dequantize_f64(&self) -> ArrayD<f64>;

    /// Partial dequantization for mixed precision operations
    /// 混合精度演算のための部分非量子化
    fn dequantize_partial(
        &self,
        new_scale: f32,
        new_zero_point: i32,
    ) -> RusTorchResult<QuantizedTensor<Q>>;
}

impl<Q: QuantizableInteger> DequantizeOps<Q> for QuantizedTensor<Q> {
    fn dequantize_f32(&self) -> ArrayD<f32> {
        self.data.mapv(|q_val| {
            (QuantizableInteger::to_i32(&q_val) - self.zero_point) as f32 * self.scale
        })
    }

    fn dequantize_f64(&self) -> ArrayD<f64> {
        self.data.mapv(|q_val| {
            ((QuantizableInteger::to_i32(&q_val) - self.zero_point) as f32 * self.scale) as f64
        })
    }

    fn dequantize_partial(
        &self,
        new_scale: f32,
        new_zero_point: i32,
    ) -> RusTorchResult<QuantizedTensor<Q>> {
        let result_data = self.data.mapv(|q_val| {
            // Dequantize to floating point
            let fp_val = (QuantizableInteger::to_i32(&q_val) - self.zero_point) as f32 * self.scale;

            // Requantize with new parameters
            let new_q_val = (fp_val / new_scale).round() as i32 + new_zero_point;
            Q::from_i32_clamped(new_q_val)
        });

        Ok(QuantizedTensor::new(
            result_data,
            new_scale,
            new_zero_point,
            self.device.clone(),
        ))
    }
}

/// Quantized operation types for parameter computation
/// パラメータ計算のための量子化演算タイプ
#[derive(Debug, Clone, Copy)]
enum QuantizedOperation {
    Add,
    Sub,
    Mul,
    Div,
}

/// Compute optimal quantization parameters for operation outputs
/// 演算出力の最適量子化パラメータを計算
fn compute_output_quantization_params(
    params1: (f32, i32),
    params2: (f32, i32),
    operation: QuantizedOperation,
) -> RusTorchResult<(f32, i32)> {
    let (scale1, zp1) = params1;
    let (scale2, zp2) = params2;

    match operation {
        QuantizedOperation::Add | QuantizedOperation::Sub => {
            // For addition/subtraction, use the larger scale to preserve precision
            // 加算/減算の場合、精度を保持するためより大きなスケールを使用
            let result_scale = scale1.max(scale2);

            // For mixed addition, zero point is typically 0
            // 混合加算の場合、ゼロポイントは通常0
            let result_zero_point = 0;

            Ok((result_scale, result_zero_point))
        }
        QuantizedOperation::Mul => {
            // For multiplication, scale is product of input scales
            // 乗算の場合、スケールは入力スケールの積
            let result_scale = scale1 * scale2;
            let result_zero_point = 0; // Typically 0 for multiplication

            Ok((result_scale, result_zero_point))
        }
        QuantizedOperation::Div => {
            // For division, scale is ratio of input scales
            // 除算の場合、スケールは入力スケールの比
            let result_scale = if scale2 != 0.0 {
                scale1 / scale2
            } else {
                scale1
            };
            let result_zero_point = 0;

            Ok((result_scale, result_zero_point))
        }
    }
}

/// High-level quantized linear layer operation
/// 高レベル量子化線形層演算
pub fn qlinear<Q: QuantizableInteger>(
    input: &QuantizedTensor<Q>,
    weight: &QuantizedTensor<Q>,
    bias: Option<&QuantizedTensor<Q>>,
) -> RusTorchResult<QuantizedTensor<Q>> {
    // Quantized linear: output = input @ weight.T + bias
    // 量子化線形：output = input @ weight.T + bias

    // First perform matrix multiplication
    let output = input.qmatmul(weight)?;

    // Add bias if provided
    if let Some(bias_tensor) = bias {
        output.qadd(bias_tensor)
    } else {
        Ok(output)
    }
}

/// High-level quantized convolution operation (simplified 1D case)
/// 高レベル量子化畳み込み演算（簡略化1Dケース）
pub fn qconv1d<Q: QuantizableInteger>(
    input: &QuantizedTensor<Q>,
    weight: &QuantizedTensor<Q>,
    bias: Option<&QuantizedTensor<Q>>,
    stride: usize,
    padding: usize,
) -> RusTorchResult<QuantizedTensor<Q>> {
    // Simplified 1D convolution implementation
    // 簡略化1D畳み込み実装

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 3 || weight_shape.len() != 3 {
        return Err(RusTorchError::TensorOp {
            message: "Expected 3D tensors for 1D convolution [batch, channels, length]".to_string(),
            source: None,
        });
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    if weight_shape[1] != in_channels {
        return Err(RusTorchError::ShapeMismatch {
            expected: vec![in_channels],
            actual: vec![weight_shape[1]],
        });
    }

    // Calculate output length
    let output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    // For now, return a placeholder implementation
    // TODO: Implement full quantized convolution
    let result_scale = input.scale * weight.scale;
    let result_shape = vec![batch_size, out_channels, output_length];
    let result_data = ArrayD::zeros(result_shape);

    Ok(QuantizedTensor::new(
        result_data,
        result_scale,
        0,
        input.device.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::device::Device;
    use ndarray::Array2;

    #[test]
    fn test_quantized_addition() {
        let data1 = Array2::from_shape_vec((2, 2), vec![10i8, 20, 30, 40])
            .unwrap()
            .into_dyn();
        let data2 = Array2::from_shape_vec((2, 2), vec![5i8, 10, 15, 20])
            .unwrap()
            .into_dyn();

        let qtensor1 = QuantizedTensor::new(data1, 0.1, 0, Device::default());
        let qtensor2 = QuantizedTensor::new(data2, 0.1, 0, Device::default());

        let result = qtensor1.qadd(&qtensor2).unwrap();

        // Check that addition was performed
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.scale, 0.1);
        assert_eq!(result.zero_point, 0);
    }

    #[test]
    fn test_quantized_multiplication() {
        let data1 = Array2::from_shape_vec((2, 2), vec![2i8, 3, 4, 5])
            .unwrap()
            .into_dyn();
        let data2 = Array2::from_shape_vec((2, 2), vec![3i8, 4, 5, 6])
            .unwrap()
            .into_dyn();

        let qtensor1 = QuantizedTensor::new(data1, 0.1, 0, Device::default());
        let qtensor2 = QuantizedTensor::new(data2, 0.2, 0, Device::default());

        let result = qtensor1.qmul(&qtensor2).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.scale, 0.1 * 0.2); // Scales multiply
        assert_eq!(result.zero_point, 0);
    }

    #[test]
    fn test_quantized_relu() {
        let data = Array2::from_shape_vec((2, 2), vec![-10i8, -5, 5, 10])
            .unwrap()
            .into_dyn();
        let qtensor = QuantizedTensor::new(data, 0.1, 0, Device::default());

        let result = qtensor.qrelu().unwrap();

        // ReLU should clamp negative values to zero point
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.scale, 0.1);
        assert_eq!(result.zero_point, 0);
    }

    #[test]
    fn test_quantized_matmul() {
        let data1 = Array2::from_shape_vec((2, 3), vec![1i8, 2, 3, 4, 5, 6])
            .unwrap()
            .into_dyn();
        let data2 = Array2::from_shape_vec((3, 2), vec![7i8, 8, 9, 10, 11, 12])
            .unwrap()
            .into_dyn();

        let qtensor1 = QuantizedTensor::new(data1, 0.1, 0, Device::default());
        let qtensor2 = QuantizedTensor::new(data2, 0.1, 0, Device::default());

        let result = qtensor1.qmatmul(&qtensor2).unwrap();

        assert_eq!(result.shape(), &[2, 2]); // [2,3] @ [3,2] = [2,2]
        assert_eq!(result.scale, 0.1 * 0.1);
    }

    #[test]
    fn test_scalar_operations() {
        let data = Array2::from_shape_vec((2, 2), vec![10i8, 20, 30, 40])
            .unwrap()
            .into_dyn();
        let qtensor = QuantizedTensor::new(data, 0.1, 0, Device::default());

        // Test scalar addition
        let result_add = qtensor.qadd_scalar(5.0).unwrap();
        assert_eq!(result_add.scale, 0.1);

        // Test scalar multiplication
        let result_mul = qtensor.qmul_scalar(2.0).unwrap();
        assert_eq!(result_mul.scale, 0.1 * 2.0);
    }

    #[test]
    fn test_dequantization() {
        let data = Array2::from_shape_vec((2, 2), vec![10i8, 20, 30, 40])
            .unwrap()
            .into_dyn();
        let qtensor = QuantizedTensor::new(data, 0.1, 0, Device::default());

        let dequantized_f32 = qtensor.dequantize_f32();
        let dequantized_f64 = qtensor.dequantize_f64();

        assert_eq!(dequantized_f32.shape(), &[2, 2]);
        assert_eq!(dequantized_f64.shape(), &[2, 2]);

        // Check actual values
        assert_eq!(dequantized_f32[[0, 0]], 1.0); // 10 * 0.1 = 1.0
        assert_eq!(dequantized_f32[[0, 1]], 2.0); // 20 * 0.1 = 2.0
    }

    #[test]
    fn test_qlinear() {
        let input_data = Array2::from_shape_vec((1, 3), vec![1i8, 2, 3])
            .unwrap()
            .into_dyn();
        let weight_data = Array2::from_shape_vec((3, 2), vec![1i8, 2, 3, 4, 5, 6])
            .unwrap()
            .into_dyn();

        let input = QuantizedTensor::new(input_data, 0.1, 0, Device::default());
        let weight = QuantizedTensor::new(weight_data, 0.1, 0, Device::default());

        let result = qlinear(&input, &weight, None).unwrap();
        assert_eq!(result.shape(), &[1, 2]);
    }
}
