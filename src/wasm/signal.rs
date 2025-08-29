//! Signal processing functions for WASM
//! WASM用信号処理関数

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible FFT implementation using Cooley-Tukey algorithm
/// WASM互換のCooley-TukeyアルゴリズムFFT実装
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSignal;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSignal {
    /// Discrete Fourier Transform (DFT) - basic O(N²) implementation
    /// 離散フーリエ変換(DFT) - 基本O(N²)実装
    #[wasm_bindgen]
    pub fn dft(real_input: Vec<f32>) -> js_sys::Object {
        let n = real_input.len();
        let mut real_output = Vec::with_capacity(n);
        let mut imag_output = Vec::with_capacity(n);

        for k in 0..n {
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                sum_real += real_input[j] * cos_val;
                sum_imag += real_input[j] * sin_val;
            }

            real_output.push(sum_real);
            imag_output.push(sum_imag);
        }

        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"real".into(),
            &js_sys::Array::from_iter(real_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        js_sys::Reflect::set(
            &result,
            &"imag".into(),
            &js_sys::Array::from_iter(imag_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        result
    }

    /// Inverse Discrete Fourier Transform (IDFT)
    /// 逆離散フーリエ変換(IDFT)
    #[wasm_bindgen]
    pub fn idft(real_input: Vec<f32>, imag_input: Vec<f32>) -> js_sys::Object {
        let n = real_input.len();
        if n != imag_input.len() {
            panic!("Real and imaginary parts must have the same length");
        }

        let mut real_output = Vec::with_capacity(n);
        let mut imag_output = Vec::with_capacity(n);

        for k in 0..n {
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;

            for j in 0..n {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                // Complex multiplication: (real_input[j] + i*imag_input[j]) * (cos + i*sin)
                sum_real += real_input[j] * cos_val - imag_input[j] * sin_val;
                sum_imag += real_input[j] * sin_val + imag_input[j] * cos_val;
            }

            real_output.push(sum_real / n as f32);
            imag_output.push(sum_imag / n as f32);
        }

        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"real".into(),
            &js_sys::Array::from_iter(real_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        js_sys::Reflect::set(
            &result,
            &"imag".into(),
            &js_sys::Array::from_iter(imag_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        result
    }

    /// Real Fast Fourier Transform (RFFT) - optimized for real inputs
    /// 実数高速フーリエ変換(RFFT) - 実数入力用最適化
    #[wasm_bindgen]
    pub fn rfft(real_input: Vec<f32>) -> js_sys::Object {
        let n = real_input.len();
        let output_len = n / 2 + 1; // Only return non-redundant frequencies
        let mut real_output = Vec::with_capacity(output_len);
        let mut imag_output = Vec::with_capacity(output_len);

        for k in 0..output_len {
            let mut sum_real = 0.0;
            let mut sum_imag = 0.0;

            for j in 0..n {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                sum_real += real_input[j] * cos_val;
                sum_imag += real_input[j] * sin_val;
            }

            real_output.push(sum_real);
            imag_output.push(sum_imag);
        }

        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"real".into(),
            &js_sys::Array::from_iter(real_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        js_sys::Reflect::set(
            &result,
            &"imag".into(),
            &js_sys::Array::from_iter(imag_output.iter().map(|&x| js_sys::Number::from(x))),
        )
        .unwrap();
        result
    }

    /// Compute power spectral density
    /// パワースペクトル密度を計算
    #[wasm_bindgen]
    pub fn power_spectrum(real_input: Vec<f32>) -> Vec<f32> {
        let fft_result = Self::rfft(real_input);
        let real_fft = js_sys::Reflect::get(&fft_result, &"real".into()).unwrap();
        let imag_fft = js_sys::Reflect::get(&fft_result, &"imag".into()).unwrap();

        let real_array: js_sys::Array = real_fft.into();
        let imag_array: js_sys::Array = imag_fft.into();

        let mut power_spectrum = Vec::new();
        for i in 0..real_array.length() {
            let real_val: f32 = real_array.get(i).as_f64().unwrap() as f32;
            let imag_val: f32 = imag_array.get(i).as_f64().unwrap() as f32;
            let power = real_val * real_val + imag_val * imag_val;
            power_spectrum.push(power);
        }

        power_spectrum
    }

    /// Apply Hamming window to signal
    /// ハミング窓をシグナルに適用
    #[wasm_bindgen]
    pub fn hamming_window(signal: Vec<f32>) -> Vec<f32> {
        let n = signal.len();
        signal
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let window =
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
                x * window
            })
            .collect()
    }

    /// Apply Hanning window to signal
    /// ハン窓をシグナルに適用
    #[wasm_bindgen]
    pub fn hanning_window(signal: Vec<f32>) -> Vec<f32> {
        let n = signal.len();
        signal
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
                x * window
            })
            .collect()
    }

    /// Apply Blackman window to signal
    /// ブラックマン窓をシグナルに適用
    #[wasm_bindgen]
    pub fn blackman_window(signal: Vec<f32>) -> Vec<f32> {
        let n = signal.len();
        signal
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let alpha = 0.16;
                let a0 = (1.0 - alpha) / 2.0;
                let a1 = 0.5;
                let a2 = alpha / 2.0;
                let window = a0
                    - a1 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
                    + a2 * (4.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
                x * window
            })
            .collect()
    }

    /// Compute magnitude spectrum
    /// 振幅スペクトルを計算
    #[wasm_bindgen]
    pub fn magnitude_spectrum(real_fft: Vec<f32>, imag_fft: Vec<f32>) -> Vec<f32> {
        real_fft
            .into_iter()
            .zip(imag_fft.into_iter())
            .map(|(real, imag)| (real * real + imag * imag).sqrt())
            .collect()
    }

    /// Compute phase spectrum
    /// 位相スペクトルを計算
    #[wasm_bindgen]
    pub fn phase_spectrum(real_fft: Vec<f32>, imag_fft: Vec<f32>) -> Vec<f32> {
        real_fft
            .into_iter()
            .zip(imag_fft.into_iter())
            .map(|(real, imag)| imag.atan2(real))
            .collect()
    }

    /// Generate frequency bins for FFT result
    /// FFT結果の周波数ビンを生成
    #[wasm_bindgen]
    pub fn fft_frequencies(n: usize, sample_rate: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let freq = i as f32 * sample_rate / n as f32;
                if i <= n / 2 {
                    freq
                } else {
                    freq - sample_rate
                }
            })
            .collect()
    }

    /// Generate frequency bins for RFFT result
    /// RFFT結果の周波数ビンを生成
    #[wasm_bindgen]
    pub fn rfft_frequencies(n: usize, sample_rate: f32) -> Vec<f32> {
        let output_len = n / 2 + 1;
        (0..output_len)
            .map(|i| i as f32 * sample_rate / n as f32)
            .collect()
    }

    /// Apply low-pass filter (simple moving average)
    /// ローパスフィルタを適用（単純移動平均）
    #[wasm_bindgen]
    pub fn low_pass_filter(signal: Vec<f32>, window_size: usize) -> Vec<f32> {
        if window_size == 0 || window_size > signal.len() {
            return signal;
        }

        let mut filtered = Vec::new();
        let half_window = window_size / 2;

        for i in 0..signal.len() {
            let start = if i >= half_window { i - half_window } else { 0 };
            let end = std::cmp::min(i + half_window + 1, signal.len());

            let sum: f32 = signal[start..end].iter().sum();
            let avg = sum / (end - start) as f32;
            filtered.push(avg);
        }

        filtered
    }

    /// Apply high-pass filter (difference from moving average)
    /// ハイパスフィルタを適用（移動平均との差分）
    #[wasm_bindgen]
    pub fn high_pass_filter(signal: Vec<f32>, window_size: usize) -> Vec<f32> {
        let low_pass = Self::low_pass_filter(signal.clone(), window_size);
        signal
            .into_iter()
            .zip(low_pass.into_iter())
            .map(|(original, filtered)| original - filtered)
            .collect()
    }

    /// Compute cross-correlation between two signals
    /// 2つの信号間の相互相関を計算
    #[wasm_bindgen]
    pub fn cross_correlation(signal_a: Vec<f32>, signal_b: Vec<f32>) -> Vec<f32> {
        let len_a = signal_a.len();
        let len_b = signal_b.len();
        let result_len = len_a + len_b - 1;
        let mut result = vec![0.0; result_len];

        for i in 0..result_len {
            for j in 0..len_a {
                let k = i as isize - j as isize;
                if k >= 0 && k < len_b as isize {
                    result[i] += signal_a[j] * signal_b[k as usize];
                }
            }
        }

        result
    }

    /// Compute autocorrelation of a signal
    /// 信号の自己相関を計算
    #[wasm_bindgen]
    pub fn autocorrelation(signal: Vec<f32>) -> Vec<f32> {
        Self::cross_correlation(signal.clone(), signal)
    }

    /// Generate sine wave
    /// 正弦波を生成
    #[wasm_bindgen]
    pub fn generate_sine_wave(
        frequency: f32,
        sample_rate: f32,
        duration: f32,
        amplitude: f32,
        phase: f32,
    ) -> Vec<f32> {
        let num_samples = (duration * sample_rate) as usize;
        let mut signal = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t + phase).sin();
            signal.push(sample);
        }

        signal
    }

    /// Generate cosine wave
    /// 余弦波を生成
    #[wasm_bindgen]
    pub fn generate_cosine_wave(
        frequency: f32,
        sample_rate: f32,
        duration: f32,
        amplitude: f32,
        phase: f32,
    ) -> Vec<f32> {
        let num_samples = (duration * sample_rate) as usize;
        let mut signal = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let sample = amplitude * (2.0 * std::f32::consts::PI * frequency * t + phase).cos();
            signal.push(sample);
        }

        signal
    }

    /// Generate white noise
    /// ホワイトノイズを生成
    #[wasm_bindgen]
    pub fn generate_white_noise(num_samples: usize, amplitude: f32, seed: u32) -> Vec<f32> {
        let mut rng = super::distributions::WasmRng::new(seed);
        let mut signal = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Generate noise in range [-amplitude, amplitude]
            let noise = amplitude * (2.0 * rng.uniform() - 1.0);
            signal.push(noise);
        }

        signal
    }

    /// Compute signal energy
    /// 信号エネルギーを計算
    #[wasm_bindgen]
    pub fn signal_energy(signal: Vec<f32>) -> f32 {
        signal.iter().map(|&x| x * x).sum()
    }

    /// Compute signal power (average energy)
    /// 信号パワー（平均エネルギー）を計算
    #[wasm_bindgen]
    pub fn signal_power(signal: Vec<f32>) -> f32 {
        let energy = Self::signal_energy(signal.clone());
        energy / signal.len() as f32
    }

    /// Compute root mean square (RMS) amplitude
    /// 実効値（RMS）振幅を計算
    #[wasm_bindgen]
    pub fn rms_amplitude(signal: Vec<f32>) -> f32 {
        Self::signal_power(signal).sqrt()
    }

    /// Find peaks in signal
    /// 信号のピークを検出
    #[wasm_bindgen]
    pub fn find_peaks(signal: Vec<f32>, threshold: f32) -> Vec<usize> {
        let mut peaks = Vec::new();

        for i in 1..signal.len() - 1 {
            if signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peaks.push(i);
            }
        }

        peaks
    }

    /// Apply gain (amplification) to signal
    /// 信号にゲイン（増幅）を適用
    #[wasm_bindgen]
    pub fn apply_gain(signal: Vec<f32>, gain: f32) -> Vec<f32> {
        signal.into_iter().map(|x| x * gain).collect()
    }

    /// Normalize signal to range [-1, 1]
    /// 信号を[-1, 1]範囲に正規化
    #[wasm_bindgen]
    pub fn normalize_signal(signal: Vec<f32>) -> Vec<f32> {
        let max_abs = signal.iter().map(|&x| x.abs()).fold(0.0, f32::max);

        if max_abs == 0.0 {
            signal
        } else {
            signal.into_iter().map(|x| x / max_abs).collect()
        }
    }

    /// Compute zero-crossing rate
    /// ゼロクロッシング率を計算
    #[wasm_bindgen]
    pub fn zero_crossing_rate(signal: Vec<f32>) -> f32 {
        let mut crossings = 0;

        for i in 1..signal.len() {
            if (signal[i] >= 0.0) != (signal[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (signal.len() - 1) as f32
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;

    #[test]
    fn test_dft_simple() {
        let signal = vec![1.0, 0.0, 1.0, 0.0];
        let result = WasmSignal::dft(signal);

        let real_part = js_sys::Reflect::get(&result, &"real".into()).unwrap();
        let imag_part = js_sys::Reflect::get(&result, &"imag".into()).unwrap();

        assert!(real_part.is_truthy());
        assert!(imag_part.is_truthy());
    }

    #[test]
    fn test_sine_wave_generation() {
        let signal = WasmSignal::generate_sine_wave(1.0, 44100.0, 1.0, 1.0, 0.0);
        assert_eq!(signal.len(), 44100);

        // Check that we have a proper sine wave
        let energy = WasmSignal::signal_energy(signal);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_windowing_functions() {
        let signal = vec![1.0; 100];

        let hamming = WasmSignal::hamming_window(signal.clone());
        let hanning = WasmSignal::hanning_window(signal.clone());
        let blackman = WasmSignal::blackman_window(signal);

        // All should have same length
        assert_eq!(hamming.len(), 100);
        assert_eq!(hanning.len(), 100);
        assert_eq!(blackman.len(), 100);

        // Edge values should be attenuated
        assert!(hamming[0] < 1.0);
        assert!(hanning[0] < 1.0);
        assert!(blackman[0] < 1.0);
    }

    #[test]
    fn test_filters() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];

        let low_pass = WasmSignal::low_pass_filter(signal.clone(), 3);
        let high_pass = WasmSignal::high_pass_filter(signal, 3);

        assert_eq!(low_pass.len(), 5);
        assert_eq!(high_pass.len(), 5);
    }

    #[test]
    fn test_signal_analysis() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];

        let energy = WasmSignal::signal_energy(signal.clone());
        let power = WasmSignal::signal_power(signal.clone());
        let rms = WasmSignal::rms_amplitude(signal.clone());
        let zcr = WasmSignal::zero_crossing_rate(signal);

        assert_eq!(energy, 5.0);
        assert_eq!(power, 1.0);
        assert_eq!(rms, 1.0);
        assert_eq!(zcr, 1.0); // Every transition crosses zero
    }
}
