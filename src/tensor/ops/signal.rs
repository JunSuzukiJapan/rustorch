//! Signal processing operations for tensors
//! テンソルの信号処理演算
//!
//! This module provides signal processing operations including FFT, IFFT,
//! and frequency domain manipulations.
//! このモジュールはFFT、IFFT、周波数領域操作を含む信号処理演算を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Fast Fourier Transform (placeholder implementation)
    /// 高速フーリエ変換（プレースホルダー実装）
    pub fn fft(&self, n: Option<usize>, dim: Option<isize>, norm: Option<&str>) -> Result<(Self, Self), String> {
        // This is a placeholder implementation
        // In practice, you would integrate with a proper FFT library like rustfft
        // 実際にはrustfftなどの適切なFFTライブラリと統合します
        
        let _n = n.unwrap_or(self.shape().last().copied().unwrap_or(1));
        let _dim = dim.unwrap_or(-1);
        let _norm = norm.unwrap_or("backward");
        
        // For now, return input tensor as both real and imaginary parts
        // とりあえず、入力テンソルを実部と虚部の両方として返す
        let real_part = self.clone();
        let imag_part = Tensor::zeros(self.shape());
        
        Ok((real_part, imag_part))
    }

    /// Inverse Fast Fourier Transform (placeholder implementation)
    /// 逆高速フーリエ変換（プレースホルダー実装）
    pub fn ifft(&self, real_part: &Self, imag_part: &Self, n: Option<usize>, dim: Option<isize>, norm: Option<&str>) -> Result<(Self, Self), String> {
        if real_part.shape() != imag_part.shape() {
            return Err("Real and imaginary parts must have the same shape".to_string());
        }
        
        let _n = n.unwrap_or(real_part.shape().last().copied().unwrap_or(1));
        let _dim = dim.unwrap_or(-1);
        let _norm = norm.unwrap_or("backward");
        
        // Placeholder implementation - return input as-is
        // プレースホルダー実装 - 入力をそのまま返す
        Ok((real_part.clone(), imag_part.clone()))
    }

    /// Real Fast Fourier Transform (placeholder implementation)
    /// 実数高速フーリエ変換（プレースホルダー実装）
    pub fn rfft(&self, n: Option<usize>, dim: Option<isize>, norm: Option<&str>) -> Result<(Self, Self), String> {
        let _n = n.unwrap_or(self.shape().last().copied().unwrap_or(1));
        let _dim = dim.unwrap_or(-1);
        let _norm = norm.unwrap_or("backward");
        
        // Placeholder: for real FFT, output should be about half the size + 1
        // プレースホルダー: 実数FFTの場合、出力は約半分のサイズ + 1になるはず
        let mut output_shape = self.shape().to_vec();
        if let Some(last_dim) = output_shape.last_mut() {
            *last_dim = *last_dim / 2 + 1;
        }
        
        let real_part = Tensor::zeros(&output_shape);
        let imag_part = Tensor::zeros(&output_shape);
        
        Ok((real_part, imag_part))
    }

    /// Shift zero-frequency component to center of spectrum
    /// ゼロ周波数成分をスペクトラムの中央にシフト
    pub fn fftshift(&self, dim: Option<&[isize]>) -> Result<Self, String> {
        let shape = self.shape();
        let ndim = shape.len() as isize;
        
        // Determine which dimensions to shift
        let dims_to_shift: Vec<usize> = if let Some(dims) = dim {
            dims.iter()
                .map(|&d| {
                    let adjusted = if d < 0 { (ndim + d) as usize } else { d as usize };
                    if adjusted >= shape.len() {
                        return Err(format!("Dimension {} is out of bounds", d));
                    }
                    Ok(adjusted)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // If no dimensions specified, shift all dimensions
            (0..shape.len()).collect()
        };
        
        let mut result = self.clone();
        
        // Apply shift to each specified dimension
        for &dim_idx in &dims_to_shift {
            let dim_size = shape[dim_idx];
            let shift_amount = dim_size / 2;
            
            if shift_amount == 0 {
                continue; // No shift needed for size 1 dimensions
            }
            
            // For simplicity, this is a basic implementation
            // A more efficient implementation would use ndarray's axis operations
            // 簡単のため、これは基本的な実装です
            // より効率的な実装では、ndarrayの軸操作を使用します
            result = Self::_shift_along_axis(&result, dim_idx, shift_amount)?;
        }
        
        Ok(result)
    }

    /// Inverse of fftshift
    /// fftshiftの逆
    pub fn ifftshift(&self, dim: Option<&[isize]>) -> Result<Self, String> {
        let shape = self.shape();
        let ndim = shape.len() as isize;
        
        let dims_to_shift: Vec<usize> = if let Some(dims) = dim {
            dims.iter()
                .map(|&d| {
                    let adjusted = if d < 0 { (ndim + d) as usize } else { d as usize };
                    if adjusted >= shape.len() {
                        return Err(format!("Dimension {} is out of bounds", d));
                    }
                    Ok(adjusted)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..shape.len()).collect()
        };
        
        let mut result = self.clone();
        
        for &dim_idx in &dims_to_shift {
            let dim_size = shape[dim_idx];
            let shift_amount = (dim_size + 1) / 2; // Ceiling division for ifftshift
            
            if shift_amount == 0 || shift_amount == dim_size {
                continue;
            }
            
            result = Self::_shift_along_axis(&result, dim_idx, shift_amount)?;
        }
        
        Ok(result)
    }

    /// Helper function to shift data along a specific axis
    /// 特定の軸に沿ってデータをシフトするヘルパー関数
    fn _shift_along_axis(tensor: &Self, axis: usize, shift: usize) -> Result<Self, String> {
        let shape = tensor.shape();
        let axis_size = shape[axis];
        
        if shift == 0 || shift >= axis_size {
            return Ok(tensor.clone());
        }
        
        // For simplicity, we'll flatten, rearrange, and reshape
        // This is not the most efficient but works for demonstration
        // 簡単のため、平坦化、再配置、再形状化を行います
        // これは最も効率的ではありませんが、デモンストレーション用には動作します
        let data = tensor.as_slice().unwrap();
        let mut result_data = Vec::with_capacity(data.len());
        
        // Calculate strides for the given axis
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        let axis_stride = strides[axis];
        let outer_size = if axis == 0 { 1 } else { shape[..axis].iter().product() };
        let inner_size = if axis == shape.len() - 1 { 1 } else { shape[axis + 1..].iter().product() };
        
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Shift the data along the specified axis
                for i in 0..axis_size {
                    let src_idx = (i + shift) % axis_size;
                    let linear_idx = outer * (axis_size * inner_size) + src_idx * inner_size + inner;
                    result_data.push(data[linear_idx]);
                }
            }
        }
        
        Ok(Tensor::from_vec(result_data, shape.to_vec()))
    }

    /// Apply window function to the tensor
    /// テンソルに窓関数を適用
    pub fn apply_window(&self, window_type: WindowType, axis: Option<usize>) -> Result<Self, String> {
        let target_axis = axis.unwrap_or(self.shape().len() - 1);
        
        if target_axis >= self.shape().len() {
            return Err(format!("Axis {} is out of bounds", target_axis));
        }
        
        let window_size = self.shape()[target_axis];
        let window = Self::create_window(window_type, window_size)?;
        
        // Apply window along the specified axis
        // 指定軸に沿って窓関数を適用
        self.apply_window_along_axis(&window, target_axis)
    }

    /// Create a window function
    /// 窓関数を作成
    fn create_window(window_type: WindowType, size: usize) -> Result<Self, String> {
        let mut window_data = Vec::with_capacity(size);
        
        match window_type {
            WindowType::Hann => {
                for i in 0..size {
                    let val = T::from(0.5).unwrap() * (T::one() - T::cos(T::from(2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).unwrap()));
                    window_data.push(val);
                }
            },
            WindowType::Hamming => {
                for i in 0..size {
                    let val = T::from(0.54).unwrap() - T::from(0.46).unwrap() * T::cos(T::from(2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).unwrap());
                    window_data.push(val);
                }
            },
            WindowType::Blackman => {
                for i in 0..size {
                    let cos1 = T::cos(T::from(2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).unwrap());
                    let cos2 = T::cos(T::from(4.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).unwrap());
                    let val = T::from(0.42).unwrap() - T::from(0.5).unwrap() * cos1 + T::from(0.08).unwrap() * cos2;
                    window_data.push(val);
                }
            },
            WindowType::Rectangular => {
                for _ in 0..size {
                    window_data.push(T::one());
                }
            }
        }
        
        Ok(Tensor::from_vec(window_data, vec![size]))
    }

    /// Apply window along specified axis
    /// 指定軸に沿って窓関数を適用
    fn apply_window_along_axis(&self, window: &Self, axis: usize) -> Result<Self, String> {
        let shape = self.shape();
        let window_size = window.numel();
        
        if shape[axis] != window_size {
            return Err(format!("Window size {} doesn't match axis size {}", window_size, shape[axis]));
        }
        
        let data = self.as_slice().unwrap();
        let window_data = window.as_slice().unwrap();
        let mut result_data = Vec::with_capacity(data.len());
        
        // Calculate strides
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        let axis_stride = strides[axis];
        let outer_size = if axis == 0 { 1 } else { shape[..axis].iter().product() };
        let inner_size = if axis == shape.len() - 1 { 1 } else { shape[axis + 1..].iter().product() };
        
        for outer in 0..outer_size {
            for i in 0..shape[axis] {
                for inner in 0..inner_size {
                    let linear_idx = outer * (shape[axis] * inner_size) + i * inner_size + inner;
                    result_data.push(data[linear_idx] * window_data[i]);
                }
            }
        }
        
        Ok(Tensor::from_vec(result_data, shape.to_vec()))
    }
}

/// Types of window functions available
/// 利用可能な窓関数の種類
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Hann (Hanning) window
    /// ハン（ハニング）窓
    Hann,
    /// Hamming window
    /// ハミング窓
    Hamming,
    /// Blackman window
    /// ブラックマン窓
    Blackman,
    /// Rectangular (no) window
    /// 矩形（無）窓
    Rectangular,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_placeholder() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let (real, imag) = tensor.fft(None, None, None).unwrap();
        assert_eq!(real.shape(), tensor.shape());
        assert_eq!(imag.shape(), tensor.shape());
    }

    #[test]
    fn test_rfft_placeholder() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let (real, imag) = tensor.rfft(None, None, None).unwrap();
        assert_eq!(real.shape(), &[3]); // (4/2 + 1) = 3
        assert_eq!(imag.shape(), &[3]);
    }

    #[test]
    fn test_fftshift() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], vec![4]);
        
        let shifted = tensor.fftshift(None).unwrap();
        assert_eq!(shifted.shape(), tensor.shape());
        
        // For size 4, shift by 2: [0,1,2,3] -> [2,3,0,1]
        assert_eq!(shifted.as_slice().unwrap(), &[2.0f32, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ifftshift() {
        let tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], vec![4]);
        
        let shifted = tensor.ifftshift(None).unwrap();
        assert_eq!(shifted.shape(), tensor.shape());
        
        // ifftshift should shift by (4+1)/2 = 2, but the result depends on implementation
        assert_eq!(shifted.as_slice().unwrap(), &[2.0f32, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn test_window_functions() {
        let tensor = Tensor::from_vec(vec![1.0f32; 8], vec![8]);
        
        // Test Hann window
        let windowed = tensor.apply_window(WindowType::Hann, None).unwrap();
        assert_eq!(windowed.shape(), tensor.shape());
        
        // Hann window should start and end with 0
        let windowed_data = windowed.as_slice().unwrap();
        assert!(windowed_data[0].abs() < 1e-6);
        assert!(windowed_data[7].abs() < 1e-6);
        
        // Test Rectangular window (should be unchanged)
        let rectangular = tensor.apply_window(WindowType::Rectangular, None).unwrap();
        assert_eq!(rectangular.as_slice().unwrap(), tensor.as_slice().unwrap());
    }

    #[test]
    fn test_create_window() {
        // Test Hann window creation
        let hann = Tensor::<f32>::create_window(WindowType::Hann, 4).unwrap();
        assert_eq!(hann.shape(), &[4]);
        
        let hann_data = hann.as_slice().unwrap();
        assert!(hann_data[0].abs() < 1e-6); // Should start with ~0
        assert!(hann_data[3].abs() < 1e-6); // Should end with ~0
        
        // Test Rectangular window
        let rect = Tensor::<f32>::create_window(WindowType::Rectangular, 5).unwrap();
        let rect_data = rect.as_slice().unwrap();
        for &val in rect_data {
            assert_eq!(val, 1.0);
        }
    }

    #[test]
    fn test_multidimensional_fftshift() {
        let tensor = Tensor::from_vec(
            (0..8).map(|x| x as f32).collect(), 
            vec![2, 4]
        );
        
        // Test shifting along specific dimension
        let shifted = tensor.fftshift(Some(&[1])).unwrap();
        assert_eq!(shifted.shape(), tensor.shape());
    }

    #[test]
    fn test_fftshift_round_trip() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        
        let shifted = tensor.fftshift(None).unwrap();
        let back = shifted.ifftshift(None).unwrap();
        
        // Should be close to original (might have small numerical differences)
        let original_data = tensor.as_slice().unwrap();
        let back_data = back.as_slice().unwrap();
        
        for (i, (&orig, &restored)) in original_data.iter().zip(back_data.iter()).enumerate() {
            assert!((orig - restored).abs() < 1e-6, "Mismatch at index {}: {} vs {}", i, orig, restored);
        }
    }
}