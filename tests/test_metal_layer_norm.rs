//! Test Metal LayerNorm GPU acceleration
//! Metal LayerNorm GPUアクセラレーションのテスト

#[cfg(feature = "metal")]
#[test]
fn test_metal_layer_norm_basic() {
    use rustorch::gpu::metal_kernels::metal_layer_norm_f64;

    // Simple test: 1 batch, 1 sequence, 4 features
    let batch_size = 1;
    let seq_len = 1;
    let features = 4;
    let eps = 1e-5;

    let input = vec![1.0f64, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f64, 1.0, 1.0, 1.0]; // weight (all ones)
    let beta = vec![0.0f64, 0.0, 0.0, 0.0];  // bias (all zeros)
    let mut output = vec![0.0f64; 4];

    let result = metal_layer_norm_f64(
        &input,
        &mut output,
        &gamma,
        &beta,
        batch_size,
        seq_len,
        features,
        eps,
    );

    if let Err(e) = &result {
        eprintln!("Metal LayerNorm error: {:?}", e);
    }
    assert!(result.is_ok(), "Metal LayerNorm should execute successfully: {:?}", result);

    // Verify output is normalized (mean ≈ 0, std ≈ 1)
    let mean: f64 = output.iter().sum::<f64>() / features as f64;
    let variance: f64 = output.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / features as f64;
    let std = variance.sqrt();

    println!("Input: {:?}", input);
    println!("Output: {:?}", output);
    println!("Mean: {}, Std: {}", mean, std);

    assert!(mean.abs() < 1e-6, "Mean should be close to 0");
    assert!((std - 1.0).abs() < 1e-6, "Std should be close to 1");
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_layer_norm_with_gpt_dimensions() {
    use rustorch::gpu::metal_kernels::metal_layer_norm_f64;

    // GPT-like dimensions: batch=1, seq_len=2, features=2048 (d_model)
    let batch_size = 1;
    let seq_len = 2;
    let features = 16; // Smaller for test, real GPT uses 2048

    let mut input = vec![0.0f64; batch_size * seq_len * features];
    // Fill with some pattern
    for i in 0..input.len() {
        input[i] = (i as f64) * 0.1;
    }

    let gamma = vec![1.0f64; features];
    let beta = vec![0.0f64; features];
    let mut output = vec![0.0f64; batch_size * seq_len * features];

    let result = metal_layer_norm_f64(
        &input,
        &mut output,
        &gamma,
        &beta,
        batch_size,
        seq_len,
        features,
        1e-5,
    );

    if let Err(e) = &result {
        eprintln!("Metal LayerNorm error (GPT dims): {:?}", e);
    }
    assert!(result.is_ok(), "Metal LayerNorm should handle GPT-like dimensions: {:?}", result);

    // Verify each position is normalized
    for s in 0..seq_len {
        let offset = s * features;
        let position_output = &output[offset..offset + features];

        let mean: f64 = position_output.iter().sum::<f64>() / features as f64;
        let variance: f64 = position_output.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / features as f64;
        let std = variance.sqrt();

        println!("Position {}: mean={:.6}, std={:.6}", s, mean, std);

        assert!(mean.abs() < 1e-5, "Mean should be close to 0 at position {}", s);
        assert!((std - 1.0).abs() < 1e-5, "Std should be close to 1 at position {}", s);
    }
}
