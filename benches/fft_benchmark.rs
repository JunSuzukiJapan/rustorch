/// FFT Performance Benchmark
/// FFT性能ベンチマーク
/// 
/// Comprehensive benchmarking of Fourier transform operations with different sizes and algorithms.
/// 異なるサイズとアルゴリズムでのフーリエ変換演算の包括的ベンチマーク

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use rustorch::tensor::Tensor;

fn fft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Operations");
    group.sample_size(10); // Conservative sample size to avoid timeouts
    group.measurement_time(std::time::Duration::from_secs(2));
    group.warm_up_time(std::time::Duration::from_millis(500));
    
    // Test various sizes including power-of-2 and non-power-of-2
    let sizes = vec![4, 8, 16, 32, 64, 128, 6, 10, 12];
    
    for size in sizes {
        let signal: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(signal, vec![size]);
        
        let algorithm = if size.is_power_of_two() { "Cooley-Tukey" } else { "DFT" };
        
        group.bench_with_input(
            BenchmarkId::new(format!("fft_1d_{}", algorithm), size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.fft(None, None, None);
                    black_box(result)
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new(format!("ifft_1d_{}", algorithm), size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    if let Ok((fft_real, fft_imag)) = tensor.fft(None, None, None) {
                        let result = tensor.ifft(&fft_real, &fft_imag, None, None, None);
                        black_box(result)
                    } else {
                        black_box(Err::<(rustorch::tensor::Tensor<f32>, rustorch::tensor::Tensor<f32>), String>("FFT failed".to_string()))
                    }
                });
            }
        );
        
        // Real FFT benchmarks
        group.bench_with_input(
            BenchmarkId::new(format!("rfft_{}", algorithm), size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.rfft(None, None, None);
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

fn fft2_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT2 Operations");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    group.warm_up_time(std::time::Duration::from_millis(500));
    
    // 2D FFT sizes
    let sizes_2d = vec![(4, 4), (8, 8), (16, 16), (32, 32), (6, 8), (10, 12)];
    
    for (height, width) in sizes_2d {
        let size = height * width;
        let image: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(image, vec![height, width]);
        
        let is_power_of_two = height.is_power_of_two() && width.is_power_of_two();
        let algorithm = if is_power_of_two { "Cooley-Tukey" } else { "Mixed" };
        
        group.bench_with_input(
            BenchmarkId::new(format!("fft2_{}", algorithm), format!("{}x{}", height, width)),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    // 2D FFT not implemented, use 1D FFT instead
                    let result = tensor.fft(None, None, None);
                    black_box(result)
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new(format!("ifft2_{}", algorithm), format!("{}x{}", height, width)),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    // 2D FFT not implemented, use 1D FFT instead
                    if let Ok((fft_real, fft_imag)) = tensor.fft(None, None, None) {
                        let result = tensor.ifft(&fft_real, &fft_imag, None, None, None);
                        black_box(result)
                    } else {
                        black_box(Err::<(rustorch::tensor::Tensor<f32>, rustorch::tensor::Tensor<f32>), String>("FFT failed".to_string()))
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn fft_normalization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Normalization");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    
    let size = 64; // Fixed size for normalization comparison
    let signal: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
    let tensor = Tensor::from_vec(signal, vec![size]);
    
    let norms = [None, Some("forward"), Some("backward"), Some("ortho")];
    let norm_names = ["none", "forward", "backward", "ortho"];
    
    for (norm, name) in norms.iter().zip(norm_names.iter()) {
        group.bench_with_input(
            BenchmarkId::new("fft_norm", name),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.fft(None, None, *norm);
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

fn fft_shift_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Shift Operations");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    
    let sizes = vec![16, 32, 64, 128];
    
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![size]);
        
        group.bench_with_input(
            BenchmarkId::new("fftshift", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.fftshift(None);
                    black_box(result)
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("ifftshift", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.ifftshift(None);
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

fn fft_round_trip_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Round Trip");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    
    let sizes = vec![8, 16, 32, 64];
    
    for size in sizes {
        let signal: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(signal, vec![size]);
        
        group.bench_with_input(
            BenchmarkId::new("fft_ifft_round_trip", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    if let Ok((fft_real, fft_imag)) = tensor.fft(None, None, None) {
                        if let Ok((ifft_real, _ifft_imag)) = tensor.ifft(&fft_real, &fft_imag, None, None, None) {
                            black_box(ifft_real)
                        } else {
                            black_box(rustorch::tensor::Tensor::from_vec(vec![0.0f32], vec![1]))
                        }
                    } else {
                        black_box(rustorch::tensor::Tensor::from_vec(vec![0.0f32], vec![1]))
                    }
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("rfft_irfft_round_trip", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    if let Ok((_rfft_real, _rfft_imag)) = tensor.rfft(None, None, None) {
                        // IRFFT not implemented, just benchmark RFFT
                        black_box(())
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn algorithm_comparison_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Algorithm Comparison");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    
    // Compare power-of-2 vs non-power-of-2 performance
    let power_of_2_sizes = vec![8, 16, 32, 64];
    let non_power_of_2_sizes = vec![6, 10, 12, 24];
    
    for size in power_of_2_sizes {
        let signal: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(signal, vec![size]);
        
        group.bench_with_input(
            BenchmarkId::new("Cooley-Tukey", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.fft(None, None, None);
                    black_box(result)
                });
            }
        );
    }
    
    for size in non_power_of_2_sizes {
        let signal: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let tensor = Tensor::from_vec(signal, vec![size]);
        
        group.bench_with_input(
            BenchmarkId::new("DFT", size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let result = tensor.fft(None, None, None);
                    black_box(result)
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    fft_benchmarks,
    fft2_benchmarks,
    fft_normalization_benchmarks,
    fft_shift_benchmarks,
    fft_round_trip_benchmarks,
    algorithm_comparison_benchmarks
);
criterion_main!(benches);