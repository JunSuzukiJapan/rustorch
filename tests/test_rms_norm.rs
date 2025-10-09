/// RMS Norm (Root Mean Square Normalization) unit tests
/// RMS Normの単体テスト

#[cfg(test)]
mod rms_norm_tests {
    /// RMS Norm implementation matching gpt.rs
    fn rms_norm_f32(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
    ) {
        for seq_idx in 0..seq_len {
            let offset = seq_idx * hidden_size;
            let row = &input[offset..offset + hidden_size];

            // Compute RMS (Root Mean Square)
            let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
            let rms = (mean_sq + eps).sqrt();

            // Normalize and scale with weight
            for i in 0..hidden_size {
                output[offset + i] = (row[i] / rms) * weight[i];
            }
        }
    }

    /// Calculate RMS of a vector
    fn calculate_rms(vec: &[f32]) -> f32 {
        let mean_sq: f32 = vec.iter().map(|&v| v * v).sum::<f32>() / (vec.len() as f32);
        mean_sq.sqrt()
    }

    #[test]
    fn test_rms_norm_basic() {
        // Test case: single position, simple values
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // identity weight
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        // Calculate expected RMS
        let input_rms = calculate_rms(&input);
        println!("Input RMS: {}", input_rms);

        // With identity weight, output should be normalized input
        let expected: Vec<f32> = input.iter().map(|&x| x / input_rms).collect();

        for i in 0..4 {
            assert!((output[i] - expected[i]).abs() < 1e-5,
                    "output[{}] = {}, expected = {}", i, output[i], expected[i]);
        }

        // Output RMS should be close to 1.0 with identity weight
        let output_rms = calculate_rms(&output);
        println!("Output RMS: {} (expected ~1.0)", output_rms);
        assert!((output_rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_with_weight() {
        // Test case: weight affects output
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0]; // 2x weight
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);
        let output_rms = calculate_rms(&output);
        let weight_rms = calculate_rms(&weight);

        println!("Input RMS: {}", input_rms);
        println!("Weight RMS: {}", weight_rms);
        println!("Output RMS: {}", output_rms);

        // Output RMS should be approximately equal to weight RMS
        // because normalized input has RMS ≈ 1.0
        assert!((output_rms - weight_rms).abs() < 1e-4,
                "output_rms = {}, weight_rms = {}", output_rms, weight_rms);
    }

    #[test]
    fn test_rms_norm_small_weight() {
        // Test case: small weight like observed in debugging (rms=0.046)
        let input = vec![0.01, -0.02, 0.015, -0.005]; // RMS ≈ 0.014
        let weight = vec![0.05, 0.04, 0.05, 0.04]; // RMS ≈ 0.045
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);
        let output_rms = calculate_rms(&output);
        let weight_rms = calculate_rms(&weight);

        println!("Input RMS: {}", input_rms);
        println!("Weight RMS: {}", weight_rms);
        println!("Output RMS: {}", output_rms);

        // Output RMS should match weight RMS (±10%)
        assert!((output_rms - weight_rms).abs() / weight_rms < 0.1,
                "output_rms = {}, weight_rms = {}, ratio = {}",
                output_rms, weight_rms, output_rms / weight_rms);
    }

    #[test]
    fn test_rms_norm_multiple_positions() {
        // Test case: multiple positions (seq_len > 1)
        let input = vec![
            1.0, 2.0, 3.0, 4.0,  // pos 0
            5.0, 6.0, 7.0, 8.0,  // pos 1
        ];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 8];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 2, 4, eps);

        // Each position should be normalized independently
        let pos0_input_rms = calculate_rms(&input[0..4]);
        let pos0_output_rms = calculate_rms(&output[0..4]);

        let pos1_input_rms = calculate_rms(&input[4..8]);
        let pos1_output_rms = calculate_rms(&output[4..8]);

        println!("Pos 0 - Input RMS: {}, Output RMS: {}", pos0_input_rms, pos0_output_rms);
        println!("Pos 1 - Input RMS: {}, Output RMS: {}", pos1_input_rms, pos1_output_rms);

        // Both positions should have output RMS ≈ 1.0 with identity weight
        assert!((pos0_output_rms - 1.0).abs() < 1e-4);
        assert!((pos1_output_rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_zero_input() {
        // Edge case: zero input
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        // With eps, should not divide by zero
        // Output should be input / sqrt(eps) * weight
        let expected_scale = 1.0 / eps.sqrt();
        for i in 0..4 {
            assert!(output[i].is_finite(), "output[{}] should be finite", i);
        }
    }

    #[test]
    fn test_rms_norm_varying_weights() {
        // Test case: varying weights across dimensions
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.1, 0.5, 1.0, 2.0]; // varying weights
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);

        // Each output element should be: (input[i] / input_rms) * weight[i]
        for i in 0..4 {
            let expected = (input[i] / input_rms) * weight[i];
            assert!((output[i] - expected).abs() < 1e-5,
                    "output[{}] = {}, expected = {}", i, output[i], expected);
        }
    }

    #[test]
    fn test_rms_norm_reproduces_observed_issue() {
        // Reproduce the observed issue from debugging:
        // Input RMS: 0.007, Weight RMS: 0.046, Output RMS: 0.130
        let hidden_size = 2048;

        // Create input with RMS ≈ 0.007
        let input: Vec<f32> = (0..hidden_size)
            .map(|i| 0.007 * (i as f32 / hidden_size as f32 - 0.5) * 2.0)
            .collect();

        // Create weight with RMS ≈ 0.046
        let weight: Vec<f32> = (0..hidden_size)
            .map(|i| 0.046 * (1.0 + 0.1 * ((i as f32 * 0.1).sin())))
            .collect();

        let mut output = vec![0.0; hidden_size];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, hidden_size, eps);

        let input_rms = calculate_rms(&input);
        let weight_rms = calculate_rms(&weight);
        let output_rms = calculate_rms(&output);

        println!("Observed issue reproduction:");
        println!("  Input RMS: {:.6}", input_rms);
        println!("  Weight RMS: {:.6}", weight_rms);
        println!("  Output RMS: {:.6}", output_rms);
        println!("  Expected output RMS: {:.6} (≈ weight_rms)", weight_rms);
        println!("  Actual / Expected ratio: {:.2}", output_rms / weight_rms);

        // Output RMS should match weight RMS (within 10%)
        assert!((output_rms - weight_rms).abs() / weight_rms < 0.1,
                "Output RMS ({}) should be close to weight RMS ({})",
                output_rms, weight_rms);
    }
}
