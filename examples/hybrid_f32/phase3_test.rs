//! フェーズ3数学関数・信号処理テスト例
//! Phase 3 Mathematical Functions & Signal Processing Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::{F32Tensor, WindowType};
    use std::f32::consts::{PI, E};

    rustorch::hybrid_f32_experimental!();

    println!("🧮 フェーズ3数学関数・信号処理テスト");
    println!("🧮 Phase 3 Mathematical Functions & Signal Processing Test");
    println!("===================================================\n");

    // ===== 三角関数・逆三角関数デモ / Trigonometric Functions Demo =====
    println!("📐 1. 三角関数・逆三角関数デモ / Trigonometric Functions Demo");
    println!("-------------------------------------------------------");

    let angles = F32Tensor::from_vec(vec![0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0], vec![5])?;
    println!("  Angles (radians): {:?}", angles.as_slice());

    let sin_vals = angles.sin()?;
    let cos_vals = angles.cos()?;
    let tan_vals = angles.tan()?;

    println!("  sin(angles): {:?}", sin_vals.as_slice());
    println!("  cos(angles): {:?}", cos_vals.as_slice());
    println!("  tan(angles): {:?}", tan_vals.as_slice());

    // 逆三角関数
    let unit_vals = F32Tensor::from_vec(vec![0.0, 0.5, 0.7071068, 1.0], vec![4])?;
    let asin_vals = unit_vals.asin()?;
    let acos_vals = unit_vals.acos()?;

    println!("  asin([0, 0.5, √2/2, 1]): {:?}", asin_vals.as_slice());
    println!("  acos([0, 0.5, √2/2, 1]): {:?}", acos_vals.as_slice());

    // ===== 双曲線関数デモ / Hyperbolic Functions Demo =====
    println!("\n🌊 2. 双曲線関数デモ / Hyperbolic Functions Demo");
    println!("------------------------------------------");

    let hyp_vals = F32Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3])?;
    println!("  Values: {:?}", hyp_vals.as_slice());

    let sinh_vals = hyp_vals.sinh()?;
    let cosh_vals = hyp_vals.cosh()?;
    let tanh_vals = hyp_vals.tanh()?;

    println!("  sinh(x): {:?}", sinh_vals.as_slice());
    println!("  cosh(x): {:?}", cosh_vals.as_slice());
    println!("  tanh(x): {:?}", tanh_vals.as_slice());

    // ===== 指数・対数関数デモ / Exponential & Logarithmic Demo =====
    println!("\n⚡ 3. 指数・対数関数デモ / Exponential & Logarithmic Demo");
    println!("-----------------------------------------------------");

    let exp_vals = F32Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], vec![4])?;
    println!("  Values: {:?}", exp_vals.as_slice());

    let exp_result = exp_vals.exp()?;
    let exp2_result = exp_vals.exp2()?;

    println!("  exp(x): {:?}", exp_result.as_slice());
    println!("  2^x: {:?}", exp2_result.as_slice());

    let log_vals = F32Tensor::from_vec(vec![1.0, E, 10.0, 100.0], vec![4])?;
    let ln_result = log_vals.ln()?;
    let log10_result = log_vals.log10()?;
    let log2_result = log_vals.log2()?;

    println!("  ln([1, e, 10, 100]): {:?}", ln_result.as_slice());
    println!("  log10([1, e, 10, 100]): {:?}", log10_result.as_slice());
    println!("  log2([1, e, 10, 100]): {:?}", log2_result.as_slice());

    // ===== 特殊関数デモ / Special Functions Demo =====
    println!("\n🎯 4. 特殊関数デモ / Special Functions Demo");
    println!("----------------------------------------");

    // ガンマ関数（階乗の拡張）
    let gamma_vals = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    let gamma_result = gamma_vals.gamma()?;
    println!("  Gamma function:");
    println!("    Γ([1,2,3,4,5]): {:?}", gamma_result.as_slice());
    println!("    (Should be [1, 1, 2, 6, 24] for integer factorial)");

    // エラー関数
    let erf_vals = F32Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4])?;
    let erf_result = erf_vals.erf()?;
    let erfc_result = erf_vals.erfc()?;
    println!("  Error functions:");
    println!("    erf([-1,0,1,2]): {:?}", erf_result.as_slice());
    println!("    erfc([-1,0,1,2]): {:?}", erfc_result.as_slice());

    // ===== 丸め関数デモ / Rounding Functions Demo =====
    println!("\n🔄 5. 丸め関数デモ / Rounding Functions Demo");
    println!("------------------------------------------");

    let round_vals = F32Tensor::from_vec(vec![-2.7, -1.5, -0.3, 0.3, 1.5, 2.7], vec![6])?;
    println!("  Values: {:?}", round_vals.as_slice());

    let ceil_result = round_vals.ceil()?;
    let floor_result = round_vals.floor()?;
    let round_result = round_vals.round()?;
    let trunc_result = round_vals.trunc()?;

    println!("  ceil(x):  {:?}", ceil_result.as_slice());
    println!("  floor(x): {:?}", floor_result.as_slice());
    println!("  round(x): {:?}", round_result.as_slice());
    println!("  trunc(x): {:?}", trunc_result.as_slice());

    // ===== 信号処理デモ / Signal Processing Demo =====
    println!("\n🔊 6. 信号処理デモ / Signal Processing Demo");
    println!("-----------------------------------------");

    // 基本的な信号生成
    let n = 8; // 2の累乗
    let mut signal_data = Vec::new();
    for i in 0..n {
        let t = i as f32;
        let sample = (2.0 * PI * t / n as f32).sin() + 0.5 * (4.0 * PI * t / n as f32).cos();
        signal_data.push(sample);
    }
    let signal = F32Tensor::from_vec(signal_data, vec![n])?;
    println!("  Original signal (sin + cos): {:?}", signal.as_slice());

    // 窓関数適用
    let windowed_hanning = signal.apply_window(WindowType::Hanning)?;
    let windowed_hamming = signal.apply_window(WindowType::Hamming)?;
    let windowed_blackman = signal.apply_window(WindowType::Blackman)?;

    println!("  Hanning window applied: {:?}", windowed_hanning.as_slice());
    println!("  Hamming window applied: {:?}", windowed_hamming.as_slice());
    println!("  Blackman window applied: {:?}", windowed_blackman.as_slice());

    // FFT変換
    println!("\n  FFT Analysis:");
    let fft_result = signal.fft()?;
    println!("    FFT magnitude: {:?}", fft_result.as_slice());

    let fft_windowed = windowed_hanning.fft()?;
    println!("    FFT (Hanning): {:?}", fft_windowed.as_slice());

    // RFFT（実数FFT）
    let (rfft_real, rfft_imag) = signal.rfft()?;
    println!("    RFFT real: {:?}", rfft_real.as_slice());
    println!("    RFFT imag: {:?}", rfft_imag.as_slice());

    // FFTシフト
    let shifted = fft_result.fftshift()?;
    println!("    FFT shifted: {:?}", shifted.as_slice());

    // ===== 複合演算デモ / Composite Operations Demo =====
    println!("\n🎭 7. 複合演算デモ / Composite Operations Demo");
    println!("-------------------------------------------");

    // オイラーの公式: e^(iπ) + 1 = 0 の近似
    // exp(iπ) = cos(π) + i*sin(π) = -1 + i*0
    let pi_tensor = F32Tensor::from_vec(vec![PI], vec![1])?;
    let cos_pi = pi_tensor.cos()?;
    let sin_pi = pi_tensor.sin()?;
    println!("  Euler's formula verification:");
    println!("    cos(π) = {} (should be ≈ -1)", cos_pi.as_slice()[0]);
    println!("    sin(π) = {} (should be ≈ 0)", sin_pi.as_slice()[0]);

    // 恒等式: sin²(x) + cos²(x) = 1
    let test_angle = F32Tensor::from_vec(vec![PI/3.0], vec![1])?;
    let sin_val = test_angle.sin()?;
    let cos_val = test_angle.cos()?;
    let sin_squared = sin_val.square()?;
    let cos_squared = cos_val.square()?;
    let identity_sum = sin_squared.add(&cos_squared)?;
    println!("  Pythagorean identity:");
    println!("    sin²(π/3) + cos²(π/3) = {} (should be ≈ 1)", identity_sum.as_slice()[0]);

    // 指数・対数の逆関数性: exp(ln(x)) = x
    let test_vals = F32Tensor::from_vec(vec![1.0, 5.0, 10.0], vec![3])?;
    let ln_vals = test_vals.ln()?;
    let exp_ln_vals = ln_vals.exp()?;
    println!("  Exp-log inverse property:");
    println!("    Original: {:?}", test_vals.as_slice());
    println!("    exp(ln(x)): {:?}", exp_ln_vals.as_slice());

    // ===== パフォーマンステスト / Performance Test =====
    println!("\n🚀 8. パフォーマンステスト / Performance Test");
    println!("--------------------------------------------");

    use std::time::Instant;

    // 大きな配列での演算テスト
    let large_size = 1024; // 2^10
    let large_data: Vec<f32> = (0..large_size).map(|i| (i as f32) * 0.01).collect();
    let large_tensor = F32Tensor::from_vec(large_data, vec![large_size])?;

    let start = Instant::now();
    let _large_sin = large_tensor.sin()?;
    let sin_time = start.elapsed();

    let start = Instant::now();
    let _large_exp = large_tensor.exp()?;
    let exp_time = start.elapsed();

    // 2の累乗でFFTテスト
    let fft_size = 1024;
    let fft_data: Vec<f32> = (0..fft_size).map(|i| (i as f32 / 100.0).sin()).collect();
    let fft_tensor = F32Tensor::from_vec(fft_data, vec![fft_size])?;

    let start = Instant::now();
    let _large_fft = fft_tensor.fft()?;
    let fft_time = start.elapsed();

    println!("  Performance results (size: {}):", large_size);
    println!("    sin() operation: {:?}", sin_time);
    println!("    exp() operation: {:?}", exp_time);
    println!("    FFT operation: {:?}", fft_time);

    println!("\n✅ フェーズ3テスト完了！");
    println!("✅ Phase 3 tests completed!");
    println!("\n📊 実装済みメソッド数: 31メソッド（累計: 98メソッド）");
    println!("📊 Implemented methods: 31 methods (Total: 98 methods)");
    println!("   - 数学関数: 25メソッド (Mathematical functions: 25 methods)");
    println!("     * 三角関数: sin, cos, tan, asin, acos, atan, atan2");
    println!("     * 双曲線関数: sinh, cosh, tanh, asinh, acosh, atanh");
    println!("     * 指数・対数: exp, exp2, expm1, ln, log10, log2, log1p");
    println!("     * 丸め関数: ceil, floor, round, trunc, fract");
    println!("     * 特殊関数: sign, reciprocal, square, cbrt, gamma, lgamma, erf, erfc");
    println!("   - 信号処理: 6メソッド (Signal processing: 6 methods)");
    println!("     * FFT変換: fft, ifft, rfft");
    println!("     * FFTユーティリティ: fftshift, ifftshift, apply_window");
    println!("     * 窓関数: Rectangular, Hanning, Hamming, Blackman");

    println!("\n🎯 フェーズ3の特徴:");
    println!("🎯 Phase 3 Features:");
    println!("   ✓ 完全f32専用実装（変換コスト0）");
    println!("   ✓ Complete f32-specific implementation (zero conversion cost)");
    println!("   ✓ 高精度数学関数（Lanczos近似、Abramowitz-Stegun近似）");
    println!("   ✓ High-precision mathematical functions (Lanczos, Abramowitz-Stegun approximations)");
    println!("   ✓ 効率的FFT実装（Cooley-Tukey アルゴリズム）");
    println!("   ✓ Efficient FFT implementation (Cooley-Tukey algorithm)");
    println!("   ✓ 複数窓関数サポート（信号解析用）");
    println!("   ✓ Multiple window function support (for signal analysis)");
    println!("   ✓ PyTorch互換API設計");
    println!("   ✓ PyTorch-compatible API design");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example hybrid_f32_phase3_test --features hybrid-f32");
}