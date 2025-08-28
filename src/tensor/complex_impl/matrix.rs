//! Complex matrix operations and linear algebra functions
//! 複素行列演算と線形代数関数

use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive, One, Zero};

use super::core::Complex;

/// Complex matrix operations
impl<T: Float + 'static> Tensor<Complex<T>> {
    /// Complex matrix multiplication
    /// 複素行列の乗算
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        // Support 2D matrix multiplication for now
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(format!(
                "Complex matmul currently supports only 2D matrices, got {}D @ {}D",
                self.ndim(),
                other.ndim()
            ));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape[1] != other_shape[0] {
            return Err(format!(
                "Complex matrix dimension mismatch: {}x{} @ {}x{}",
                self_shape[0], self_shape[1], other_shape[0], other_shape[1]
            ));
        }

        let m = self_shape[0];
        let n = other_shape[1];
        let k = self_shape[1];

        let mut result = vec![Complex::zero(); m * n];

        // Get slices for row-major access
        let self_slice = self.as_slice().unwrap();
        let other_slice = other.as_slice().unwrap();
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex::zero();
                for l in 0..k {
                    // Use row-major indexing: [i, l] = i * k + l
                    let self_val = self_slice[i * k + l];
                    let other_val = other_slice[l * n + j];
                    sum = sum + self_val * other_val;
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor::from_vec(result, vec![m, n]))
    }

    /// Complex matrix transpose
    /// 複素行列の転置
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Transpose currently supports only 2D matrices".to_string());
        }

        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut result = vec![Complex::zero(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                let dst_idx = j * rows + i;
                result[dst_idx] = self.data[[i, j]];
            }
        }

        Ok(Tensor::from_vec(result, vec![cols, rows]))
    }

    /// Complex matrix conjugate transpose (Hermitian transpose)
    /// 複素行列の共役転置（エルミート転置）
    pub fn conj_transpose(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Conjugate transpose currently supports only 2D matrices".to_string());
        }

        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut result = vec![Complex::zero(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                let dst_idx = j * rows + i;
                result[dst_idx] = self.data[[i, j]].conj();
            }
        }

        Ok(Tensor::from_vec(result, vec![cols, rows]))
    }

    /// Complex matrix trace (sum of diagonal elements)
    /// 複素行列のトレース（対角要素の和）
    pub fn trace(&self) -> Result<Complex<T>, String> {
        if self.ndim() != 2 {
            return Err("Trace requires a 2D matrix".to_string());
        }

        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        let min_dim = rows.min(cols);

        let mut sum = Complex::zero();
        for i in 0..min_dim {
            sum = sum + self.data[[i, i]];
        }

        Ok(sum)
    }

    /// Complex matrix determinant (2x2 only for now)
    /// 複素行列の行列式（現在は2x2のみ）
    pub fn determinant(&self) -> Result<Complex<T>, String> {
        if self.ndim() != 2 {
            return Err("Determinant requires a 2D matrix".to_string());
        }

        let shape = self.shape();
        if shape[0] != shape[1] {
            return Err("Determinant requires a square matrix".to_string());
        }

        if shape[0] == 1 {
            return Ok(self.data[[0, 0]]);
        } else if shape[0] == 2 {
            let a = self.data[[0, 0]];
            let b = self.data[[0, 1]];
            let c = self.data[[1, 0]];
            let d = self.data[[1, 1]];
            return Ok(a * d - b * c);
        } else {
            return Err(
                "Determinant only implemented for 1x1 and 2x2 matrices currently".to_string(),
            );
        }
    }

    /// Forward FFT for complex tensor
    /// 複素テンソルの順フーリエ変換
    pub fn fft(
        &self,
        n: Option<usize>,
        _dim: Option<isize>,
        norm: Option<&str>,
    ) -> Result<Self, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        if self.ndim() != 1 {
            return Err("Complex FFT currently supports only 1D tensors".to_string());
        }

        let input_len = self.shape()[0];
        let fft_len = n.unwrap_or(input_len);

        // Convert to Vec<Complex<T>> for FFT processing
        let mut fft_data: Vec<Complex<T>> = self.data.iter().cloned().collect();

        // Pad or truncate to desired length
        if fft_data.len() != fft_len {
            fft_data.resize(fft_len, Complex::zero());
        }

        // Apply FFT algorithm
        let result = if fft_len.is_power_of_two() {
            self.cooley_tukey_complex(&mut fft_data, false)?
        } else {
            self.dft_complex(&fft_data, false)?
        };

        // Apply normalization
        let normalized = self.apply_complex_normalization(result, fft_len, norm, false)?;

        Ok(Tensor::from_vec(normalized, vec![fft_len]))
    }

    /// Inverse FFT for complex tensor
    /// 複素テンソルの逆フーリエ変換
    pub fn ifft(
        &self,
        n: Option<usize>,
        _dim: Option<isize>,
        norm: Option<&str>,
    ) -> Result<Self, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        if self.ndim() != 1 {
            return Err("Complex IFFT currently supports only 1D tensors".to_string());
        }

        let input_len = self.shape()[0];
        let fft_len = n.unwrap_or(input_len);

        // Convert to Vec<Complex<T>> for IFFT processing
        let mut fft_data: Vec<Complex<T>> = self.data.iter().cloned().collect();

        // Pad or truncate to desired length
        if fft_data.len() != fft_len {
            fft_data.resize(fft_len, Complex::zero());
        }

        // Apply IFFT algorithm
        let result = if fft_len.is_power_of_two() {
            self.cooley_tukey_complex(&mut fft_data, true)?
        } else {
            self.dft_complex(&fft_data, true)?
        };

        // Apply normalization
        let normalized = self.apply_complex_normalization(result, fft_len, norm, true)?;

        Ok(Tensor::from_vec(normalized, vec![fft_len]))
    }

    /// FFT shift for complex tensor
    /// 複素テンソルのFFTシフト
    pub fn fftshift(&self, _dim: Option<&[isize]>) -> Result<Self, String> {
        if self.ndim() != 1 {
            return Err("Complex fftshift currently supports only 1D tensors".to_string());
        }

        let input_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        let input_len = input_data.len();
        let mid = input_len.div_ceil(2);
        let mut new_data = Vec::with_capacity(input_len);

        // Shift: second half to front, first half to back
        new_data.extend_from_slice(&input_data[mid..]);
        new_data.extend_from_slice(&input_data[..mid]);

        Ok(Tensor::from_vec(new_data, self.shape().to_vec()))
    }

    /// Inverse FFT shift for complex tensor
    /// 複素テンソルの逆FFTシフト
    pub fn ifftshift(&self, _dim: Option<&[isize]>) -> Result<Self, String> {
        if self.ndim() != 1 {
            return Err("Complex ifftshift currently supports only 1D tensors".to_string());
        }

        let input_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        let input_len = input_data.len();
        let mid = input_len / 2;
        let mut new_data = Vec::with_capacity(input_len);

        // Shift: second half to front, first half to back (different split)
        new_data.extend_from_slice(&input_data[mid..]);
        new_data.extend_from_slice(&input_data[..mid]);

        Ok(Tensor::from_vec(new_data, self.shape().to_vec()))
    }

    // Private helper functions for complex FFT
    fn cooley_tukey_complex(
        &self,
        data: &mut [Complex<T>],
        inverse: bool,
    ) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err("Cooley-Tukey algorithm requires power of two length".to_string());
        }

        // Bit reversal
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                data.swap(i, j);
            }
        }

        // Cooley-Tukey FFT
        let mut length = 2;
        while length <= n {
            let half_len = length / 2;
            let angle = if inverse {
                T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap()
                    / T::from(length).unwrap()
            } else {
                -T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap()
                    / T::from(length).unwrap()
            };

            let wlen = Complex::new(angle.cos(), angle.sin());

            for i in (0..n).step_by(length) {
                let mut w = Complex::one();
                for j in 0..half_len {
                    let u = data[i + j];
                    let v = data[i + j + half_len] * w;
                    data[i + j] = u + v;
                    data[i + j + half_len] = u - v;
                    w = w * wlen;
                }
            }
            length *= 2;
        }

        Ok(data.to_vec())
    }

    fn dft_complex(&self, data: &[Complex<T>], inverse: bool) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let n = data.len();
        let mut result = vec![Complex::zero(); n];

        let sign = if inverse { T::one() } else { -T::one() };
        let pi2 = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap();

        for k in 0..n {
            let mut sum = Complex::zero();
            for j in 0..n {
                let angle =
                    sign * pi2 * T::from(k).unwrap() * T::from(j).unwrap() / T::from(n).unwrap();
                let w = Complex::new(angle.cos(), angle.sin());
                sum = sum + data[j] * w;
            }
            result[k] = sum;
        }

        Ok(result)
    }

    fn apply_complex_normalization(
        &self,
        mut data: Vec<Complex<T>>,
        n: usize,
        norm: Option<&str>,
        inverse: bool,
    ) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        match norm {
            Some("forward") => {
                if !inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            }
            Some("backward") => {
                if inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            }
            Some("ortho") => {
                let scale = T::one() / T::from(n).unwrap().sqrt();
                for x in &mut data {
                    *x = *x * scale;
                }
            }
            _ => {
                if inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            }
        }
        Ok(data)
    }
}