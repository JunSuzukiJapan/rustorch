//! Complex Tensor Operations Demo
//! 複素テンソル操作デモ
//! 
//! This example demonstrates the comprehensive complex number support in RusTorch,
//! including tensor creation, mathematical operations, matrix operations, and FFT.
//! 
//! RusTorchの包括的な複素数サポートを実演。テンソル作成、数学演算、
//! 行列演算、FFTを含む。

use rustorch::tensor::{Tensor, complex::Complex};

fn main() {
    println!("=== RusTorch Complex Tensor Operations Demo ===");
    println!("=== RusTorch複素テンソル操作デモ ===\n");
    
    // 1. Complex number creation and basic operations
    // 1. 複素数の作成と基本演算
    demo_complex_numbers();
    
    // 2. Complex tensor creation
    // 2. 複素テンソルの作成
    demo_complex_tensor_creation();
    
    // 3. Complex mathematical functions
    // 3. 複素数学関数
    demo_complex_mathematical_functions();
    
    // 4. Complex matrix operations
    // 4. 複素行列演算
    demo_complex_matrix_operations();
    
    // 5. Complex FFT operations
    // 5. 複素FFT演算
    demo_complex_fft_operations();
    
    println!("Demo completed successfully! すべてのデモが正常に完了しました！");
}

fn demo_complex_numbers() {
    println!("📊 Complex Number Operations / 複素数演算");
    println!("----------------------------------------");
    
    // Create complex numbers
    let z1 = Complex::new(3.0, 4.0);  // 3 + 4i
    let z2 = Complex::new(1.0, -2.0); // 1 - 2i
    
    println!("z1 = {}", z1);
    println!("z2 = {}", z2);
    
    // Basic arithmetic
    println!("z1 + z2 = {}", z1 + z2);
    println!("z1 * z2 = {}", z1 * z2);
    println!("z1 / z2 = {}", z1 / z2);
    
    // Complex properties
    println!("Magnitude of z1: |z1| = {}", z1.abs());
    println!("Phase of z1: arg(z1) = {:.4} radians", z1.arg());
    println!("Complex conjugate: conj(z1) = {}", z1.conj());
    
    // Polar form
    let (r, theta) = z1.to_polar();
    println!("Polar form: z1 = {:.3} * exp(i * {:.3})", r, theta);
    let z1_from_polar = Complex::from_polar(r, theta);
    println!("Reconstructed from polar: {}", z1_from_polar);
    
    println!();
}

fn demo_complex_tensor_creation() {
    println!("🎯 Complex Tensor Creation / 複素テンソル作成");
    println!("------------------------------------------");
    
    // Create complex tensors from real and imaginary parts
    let real_part = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let imag_part = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
    let complex_tensor = Complex::from_tensors(&real_part, &imag_part).unwrap();
    
    println!("Real part: {:?}", real_part.data.as_slice().unwrap());
    println!("Imaginary part: {:?}", imag_part.data.as_slice().unwrap());
    println!("Complex tensor:");
    for (i, z) in complex_tensor.data.iter().enumerate() {
        println!("  [{:?}] = {}", i, z);
    }
    
    // Extract parts
    let extracted_real = Complex::tensor_real_part(&complex_tensor);
    let extracted_imag = Complex::tensor_imag_part(&complex_tensor);
    let magnitude = Complex::tensor_abs(&complex_tensor);
    let phase = Complex::tensor_arg(&complex_tensor);
    
    println!("Extracted real: {:?}", extracted_real.data.as_slice().unwrap());
    println!("Extracted imag: {:?}", extracted_imag.data.as_slice().unwrap());
    println!("Magnitude: {:?}", magnitude.data.as_slice().unwrap());
    println!("Phase: {:?}", phase.data.as_slice().unwrap());
    
    // Factory functions
    let zeros = Tensor::<Complex<f64>>::complex_zeros(&[2, 3]);
    let ones = Tensor::<Complex<f64>>::complex_ones(&[2, 3]);
    let i_tensor = Tensor::<Complex<f64>>::complex_i(&[3]);
    
    println!("Complex zeros shape: {:?}", zeros.shape());
    println!("Complex ones shape: {:?}", ones.shape());
    println!("Imaginary unit tensor: {:?}", i_tensor.data.as_slice().unwrap());
    
    println!();
}

fn demo_complex_mathematical_functions() {
    println!("🧮 Complex Mathematical Functions / 複素数学関数");
    println!("-----------------------------------------------");
    
    let complex_data = vec![
        Complex::new(1.0, 0.0),      // Real number
        Complex::new(0.0, 1.0),      // Imaginary unit
        Complex::new(1.0, 1.0),      // Complex number
        Complex::new(2.0, -1.0),     // Another complex number
    ];
    let z = Tensor::from_vec(complex_data.clone(), vec![4]);
    
    println!("Input tensor z:");
    for (i, val) in complex_data.iter().enumerate() {
        println!("  z[{}] = {}", i, val);
    }
    
    // Exponential function
    let exp_z = z.exp();
    println!("\nExponential e^z:");
    for (i, val) in exp_z.data.iter().enumerate() {
        println!("  e^z[{}] = {}", i, val);
    }
    
    // Natural logarithm
    let ln_z = z.ln();
    println!("\nNatural logarithm ln(z):");
    for (i, val) in ln_z.data.iter().enumerate() {
        println!("  ln(z[{}]) = {}", i, val);
    }
    
    // Square root
    let sqrt_z = z.sqrt();
    println!("\nSquare root √z:");
    for (i, val) in sqrt_z.data.iter().enumerate() {
        println!("  √z[{}] = {}", i, val);
    }
    
    // Trigonometric functions
    let sin_z = z.sin();
    let cos_z = z.cos();
    println!("\nTrigonometric functions:");
    for i in 0..4 {
        println!("  sin(z[{}]) = {}, cos(z[{}]) = {}", i, sin_z.data[i], i, cos_z.data[i]);
    }
    
    // Verify trigonometric identity: sin²(z) + cos²(z) = 1
    println!("\nVerifying sin²(z) + cos²(z) = 1:");
    for i in 0..4 {
        let sin_sq = sin_z.data[i] * sin_z.data[i];
        let cos_sq = cos_z.data[i] * cos_z.data[i];
        let identity = sin_sq + cos_sq;
        println!("  z[{}]: sin²+cos² = {} (should be 1+0i)", i, identity);
    }
    
    println!();
}

fn demo_complex_matrix_operations() {
    println!("🔢 Complex Matrix Operations / 複素行列演算");
    println!("--------------------------------------------");
    
    // Create complex matrices
    let a_data = vec![
        Complex::new(1.0, 1.0), Complex::new(2.0, 0.0),
        Complex::new(0.0, 1.0), Complex::new(1.0, -1.0),
    ];
    let a = Tensor::from_vec(a_data, vec![2, 2]);
    
    let b_data = vec![
        Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),
        Complex::new(1.0, 1.0), Complex::new(1.0, 0.0),
    ];
    let b = Tensor::from_vec(b_data, vec![2, 2]);
    
    println!("Matrix A:");
    print_complex_matrix(&a);
    println!("\nMatrix B:");
    print_complex_matrix(&b);
    
    // Matrix multiplication
    let ab = a.matmul(&b).unwrap();
    println!("\nMatrix multiplication A * B:");
    print_complex_matrix(&ab);
    
    // Transpose
    let a_t = a.transpose().unwrap();
    println!("\nTranspose A^T:");
    print_complex_matrix(&a_t);
    
    // Conjugate transpose (Hermitian)
    let a_h = a.conj_transpose().unwrap();
    println!("\nConjugate transpose A^H:");
    print_complex_matrix(&a_h);
    
    // Trace
    let trace_a = a.trace().unwrap();
    println!("\nTrace of A: tr(A) = {}", trace_a);
    
    // Determinant (2x2 only)
    let det_a = a.determinant().unwrap();
    println!("Determinant of A: det(A) = {}", det_a);
    
    println!();
}

fn demo_complex_fft_operations() {
    println!("🌊 Complex FFT Operations / 複素FFT演算");
    println!("---------------------------------------");
    
    // Create a complex signal (sum of sinusoids)
    let n = 8;
    let mut signal_data = Vec::new();
    
    for i in 0..n {
        let t = i as f64 / n as f64;
        // Create signal with frequency components at k=1 and k=2
        let real_part = (2.0 * std::f64::consts::PI * t).cos() + 
                       0.5 * (4.0 * std::f64::consts::PI * t).cos();
        let imag_part = (2.0 * std::f64::consts::PI * t).sin() + 
                       0.5 * (4.0 * std::f64::consts::PI * t).sin();
        signal_data.push(Complex::new(real_part, imag_part));
    }
    
    let signal = Tensor::from_vec(signal_data.clone(), vec![n]);
    
    println!("Input signal:");
    for (i, z) in signal_data.iter().enumerate() {
        println!("  x[{}] = {}", i, z);
    }
    
    // Forward FFT
    let fft_result = signal.fft(None, None, None).unwrap();
    println!("\nFFT result:");
    for (i, z) in fft_result.data.iter().enumerate() {
        let magnitude = z.abs();
        let phase = z.arg();
        println!("  X[{}] = {} (mag: {:.3}, phase: {:.3})", i, z, magnitude, phase);
    }
    
    // Inverse FFT (should reconstruct original signal)
    let ifft_result = fft_result.ifft(None, None, None).unwrap();
    println!("\nIFFT result (reconstructed signal):");
    for (i, z) in ifft_result.data.iter().enumerate() {
        println!("  x'[{}] = {} (original: {})", i, z, signal_data[i]);
    }
    
    // FFT shift
    let fft_shifted = fft_result.fftshift(None).unwrap();
    println!("\nFFT with fftshift (DC in center):");
    for (i, z) in fft_shifted.data.iter().enumerate() {
        println!("  X_shifted[{}] = {}", i, z);
    }
    
    // Power spectrum
    println!("\nPower spectrum:");
    for (i, z) in fft_result.data.iter().enumerate() {
        let power = z.abs_sq();
        println!("  |X[{}]|² = {:.3}", i, power);
    }
    
    println!();
}

fn print_complex_matrix(matrix: &Tensor<Complex<f64>>) {
    let shape = matrix.shape();
    if shape.len() != 2 {
        println!("Not a 2D matrix");
        return;
    }
    
    let rows = shape[0];
    let cols = shape[1];
    
    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            print!("{:>12}", format!("{}", matrix.data[[i, j]]));
            if j < cols - 1 {
                print!(" ");
            }
        }
        println!("]");
    }
}