/// Standalone RMS Norm test program
/// RMS Normの独立テストプログラム

/// RMS Norm implementation
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

fn main() {
    println!("=== RMS Norm Test Suite ===\n");

    // Test 1: Basic functionality with identity weight
    println!("Test 1: Basic functionality with identity weight");
    {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);
        let output_rms = calculate_rms(&output);

        println!("  Input:  {:?}", input);
        println!("  Weight: {:?}", weight);
        println!("  Output: {:?}", output);
        println!("  Input RMS:  {:.6}", input_rms);
        println!("  Output RMS: {:.6} (expected ~1.0)", output_rms);
        assert!((output_rms - 1.0).abs() < 1e-4, "FAILED: Output RMS should be ~1.0");
        println!("  ✓ PASSED\n");
    }

    // Test 2: Weight scaling effect
    println!("Test 2: Weight scaling effect");
    {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);
        let weight_rms = calculate_rms(&weight);
        let output_rms = calculate_rms(&output);

        println!("  Input RMS:  {:.6}", input_rms);
        println!("  Weight RMS: {:.6}", weight_rms);
        println!("  Output RMS: {:.6} (expected ~{:.6})", output_rms, weight_rms);

        let ratio = output_rms / weight_rms;
        println!("  Ratio (output/weight): {:.4}", ratio);
        assert!((ratio - 1.0).abs() < 0.01, "FAILED: Output RMS should match weight RMS");
        println!("  ✓ PASSED\n");
    }

    // Test 3: Small weight (reproducing observed issue)
    println!("Test 3: Small weight (reproducing observed issue)");
    {
        let input = vec![0.01, -0.02, 0.015, -0.005];
        let weight = vec![0.05, 0.04, 0.05, 0.04];
        let mut output = vec![0.0; 4];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, 4, eps);

        let input_rms = calculate_rms(&input);
        let weight_rms = calculate_rms(&weight);
        let output_rms = calculate_rms(&output);

        println!("  Input RMS:  {:.6}", input_rms);
        println!("  Weight RMS: {:.6}", weight_rms);
        println!("  Output RMS: {:.6} (expected ~{:.6})", output_rms, weight_rms);

        let ratio = output_rms / weight_rms;
        println!("  Ratio (output/weight): {:.4}", ratio);
        assert!((ratio - 1.0).abs() < 0.1, "FAILED: Output RMS should be within 10% of weight RMS");
        println!("  ✓ PASSED\n");
    }

    // Test 4: Large scale reproduction (observed issue)
    println!("Test 4: Reproducing observed issue (2048 dimensions)");
    {
        let hidden_size = 2048;

        // Create input with RMS ≈ 0.007
        let input: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let normalized = i as f32 / hidden_size as f32;
                0.007 * ((normalized - 0.5) * 2.0 + 0.1 * (normalized * 10.0).sin())
            })
            .collect();

        // Create weight with RMS ≈ 0.046
        let weight: Vec<f32> = (0..hidden_size)
            .map(|i| {
                let normalized = i as f32 / hidden_size as f32;
                0.046 * (1.0 + 0.2 * (normalized * 20.0).sin())
            })
            .collect();

        // Check actual weight statistics
        let weight_min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
        let weight_max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weight_mean = weight.iter().sum::<f32>() / weight.len() as f32;
        println!("  Weight statistics:");
        println!("    min={:.6}, max={:.6}, mean={:.6}", weight_min, weight_max, weight_mean);

        let mut output = vec![0.0; hidden_size];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 1, hidden_size, eps);

        let input_rms = calculate_rms(&input);
        let weight_rms = calculate_rms(&weight);
        let output_rms = calculate_rms(&output);

        println!("  Input RMS:  {:.6} (target: ~0.007)", input_rms);
        println!("  Weight RMS: {:.6} (target: ~0.046)", weight_rms);
        println!("  Output RMS: {:.6} (expected: ~{:.6})", output_rms, weight_rms);

        let ratio = output_rms / weight_rms;
        println!("  Ratio (output/weight): {:.4}", ratio);

        // This is the critical test - if ratio > 1.1, we have the bug
        if (ratio - 1.0).abs() > 0.1 {
            println!("  ⚠️  ISSUE DETECTED: Output RMS is {:.1}% different from expected",
                     (ratio - 1.0) * 100.0);
        } else {
            println!("  ✓ PASSED");
        }
        println!();
    }

    // Test 5: Multiple positions
    println!("Test 5: Multiple positions (seq_len=2)");
    {
        let input = vec![
            1.0, 2.0, 3.0, 4.0,  // pos 0
            5.0, 6.0, 7.0, 8.0,  // pos 1
        ];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut output = vec![0.0; 8];
        let eps = 1e-5;

        rms_norm_f32(&input, &weight, &mut output, 2, 4, eps);

        let pos0_output_rms = calculate_rms(&output[0..4]);
        let pos1_output_rms = calculate_rms(&output[4..8]);

        println!("  Pos 0 Output RMS: {:.6} (expected ~1.0)", pos0_output_rms);
        println!("  Pos 1 Output RMS: {:.6} (expected ~1.0)", pos1_output_rms);

        assert!((pos0_output_rms - 1.0).abs() < 1e-4);
        assert!((pos1_output_rms - 1.0).abs() < 1e-4);
        println!("  ✓ PASSED\n");
    }

    // Test 6: Detailed step-by-step verification
    println!("Test 6: Step-by-step verification");
    {
        let input = vec![3.0, 4.0]; // Simple 2D case
        let weight = vec![0.5, 0.5];
        let mut output = vec![0.0; 2];
        let eps = 1e-5;

        println!("  Input:  {:?}", input);
        println!("  Weight: {:?}", weight);

        // Manual calculation
        let mean_sq: f32 = (3.0*3.0 + 4.0*4.0) / 2.0; // (9 + 16) / 2 = 12.5
        let rms: f32 = (mean_sq + eps).sqrt(); // sqrt(12.5) ≈ 3.5355
        let normalized = [3.0 / rms, 4.0 / rms]; // [0.8485, 1.1314]
        let expected = [normalized[0] * 0.5, normalized[1] * 0.5]; // [0.4243, 0.5657]

        println!("  Manual calculation:");
        println!("    mean_sq = {:.6}", mean_sq);
        println!("    rms = {:.6}", rms);
        println!("    normalized = [{:.6}, {:.6}]", normalized[0], normalized[1]);
        println!("    expected output = [{:.6}, {:.6}]", expected[0], expected[1]);

        rms_norm_f32(&input, &weight, &mut output, 1, 2, eps);

        println!("  Actual output: [{:.6}, {:.6}]", output[0], output[1]);

        assert!((output[0] - expected[0]).abs() < 1e-4);
        assert!((output[1] - expected[1]).abs() < 1e-4);

        let output_rms = calculate_rms(&output);
        let weight_rms = calculate_rms(&weight);
        println!("  Output RMS: {:.6}, Weight RMS: {:.6}", output_rms, weight_rms);
        println!("  ✓ PASSED\n");
    }

    // Test 7: Uniform small weights (2048 dimensions)
    println!("Test 7: Uniform small weights (2048 dimensions)");
    {
        let hidden_size = 2048;

        // Input with small RMS
        let input: Vec<f32> = (0..hidden_size)
            .map(|i| 0.007 * ((i as f32 / 100.0).sin()))
            .collect();

        // Uniform weight with target RMS = 0.046
        let weight = vec![0.046; hidden_size];

        let mut output = vec![0.0; hidden_size];
        let eps = 1e-5;

        // Calculate normalized input manually
        let input_rms_calc = calculate_rms(&input);
        let normalized: Vec<f32> = input.iter().map(|&x| x / input_rms_calc).collect();
        let normalized_rms = calculate_rms(&normalized);

        rms_norm_f32(&input, &weight, &mut output, 1, hidden_size, eps);

        let input_rms = calculate_rms(&input);
        let weight_rms = calculate_rms(&weight);
        let output_rms = calculate_rms(&output);

        println!("  Input RMS:  {:.6}", input_rms);
        println!("  Normalized input RMS: {:.6} (should be 1.0)", normalized_rms);
        println!("  Weight value: {:.6} (uniform)", weight[0]);
        println!("  Weight RMS: {:.6} (should equal weight value for uniform)", weight_rms);
        println!("  Output RMS: {:.6} (expected: {:.6})", output_rms, weight[0]);

        // Manual calculation: normalized * weight
        let expected_output: Vec<f32> = normalized.iter().map(|&x| x * weight[0]).collect();
        let expected_output_rms = calculate_rms(&expected_output);
        println!("  Manual calc output RMS: {:.6}", expected_output_rms);

        let ratio = output_rms / weight[0];
        println!("  Ratio (output/weight): {:.4}", ratio);
        println!("  Ratio (manual/weight): {:.4}", expected_output_rms / weight[0]);

        if (ratio - 1.0).abs() < 0.01 {
            println!("  ✓ PASSED\n");
        } else {
            println!("  ⚠️  Deviation: {:.1}%\n", (ratio - 1.0) * 100.0);
        }
    }

    println!("=== All tests completed ===");
}
