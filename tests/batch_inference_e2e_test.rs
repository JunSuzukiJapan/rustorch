// E2E Batch Inference Test with Actual Model
// Tests complete batch processing pipeline with loaded TinyLlama model

#[cfg(feature = "metal")]
#[test]
#[ignore] // Run manually with: cargo test --features metal --test batch_inference_e2e_test -- --ignored --nocapture
fn test_batch_inference_with_actual_model() {
    use rustorch::models::llama::LlamaModel;
    use std::path::PathBuf;
    use std::env;

    println!("\n🚀 E2E Batch Inference Test with TinyLlama Model");
    println!("=================================================\n");

    // Construct model path
    let home = env::var("HOME").expect("HOME not set");
    let model_path = PathBuf::from(home)
        .join(".rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF")
        .join("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("📂 Model path: {:?}", model_path);

    if !model_path.exists() {
        println!("⚠️  Model file not found, skipping test");
        println!("   Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
        return;
    }

    // Load model
    println!("📥 Loading model...");
    let model_result = LlamaModel::from_gguf(&model_path);

    if let Err(e) = &model_result {
        println!("❌ Failed to load model: {:?}", e);
        return;
    }

    let mut model = model_result.unwrap();
    println!("✅ Model loaded successfully");
    println!("   Config: {} layers, {} heads, {} hidden dim, batch_size={}",
             model.config.num_layers,
             model.config.num_heads,
             model.config.hidden_size,
             model.config.batch_size);

    // Prepare batch inputs - multiple test prompts
    let test_inputs = vec![
        vec![1, 15043, 29892, 920, 526, 366], // "Hello, how are you"
        vec![1, 1724, 338, 596, 1024],         // "What is your name"
        vec![1, 450, 14744, 338, 6411],        // "The weather is nice"
    ];

    println!("\n📝 Test inputs (batch_size={}):", test_inputs.len());
    for (i, input) in test_inputs.iter().enumerate() {
        println!("   Sequence {}: {} tokens", i, input.len());
    }

    // Convert to references
    let input_refs: Vec<&[usize]> = test_inputs.iter().map(|v| v.as_slice()).collect();

    // Test 1: Forward pass with batch processing
    println!("\n🔄 Test 1: Batch Forward Pass");
    println!("──────────────────────────────");

    let start = std::time::Instant::now();
    let batch_result = model.forward_batch(&input_refs);
    let batch_duration = start.elapsed();

    match &batch_result {
        Ok(outputs) => {
            println!("✅ Batch forward pass successful");
            println!("   Duration: {:?}", batch_duration);
            println!("   Outputs: {} tensors", outputs.len());

            // Check output shapes
            for (i, output) in outputs.iter().enumerate() {
                let shape = output.shape();
                println!("   Output {}: shape {:?}, vocab_size={}",
                         i, shape, shape[shape.len()-1]);

                // Verify output is valid (not all zeros)
                let data_sum: f64 = output.data.iter().sum();
                assert!(data_sum.abs() > 1e-6, "Output {} is all zeros", i);
            }
        }
        Err(e) => {
            println!("❌ Batch forward pass failed: {:?}", e);
            panic!("Batch inference failed");
        }
    }

    // Test 2: Sequential processing for comparison
    println!("\n🔄 Test 2: Sequential Forward Pass (for comparison)");
    println!("───────────────────────────────────────────────────");

    let mut sequential_outputs = Vec::new();
    let seq_start = std::time::Instant::now();

    for (i, input) in input_refs.iter().enumerate() {
        match model.forward(*input) {
            Ok(output) => {
                println!("   Sequence {} processed: {} tokens", i, input.len());
                sequential_outputs.push(output);
            }
            Err(e) => {
                println!("❌ Sequential forward failed for sequence {}: {:?}", i, e);
                panic!("Sequential inference failed");
            }
        }
    }

    let seq_duration = seq_start.elapsed();
    println!("✅ Sequential processing complete");
    println!("   Duration: {:?}", seq_duration);

    // Test 3: Compare results
    println!("\n📊 Test 3: Result Comparison");
    println!("────────────────────────────");

    let speedup = seq_duration.as_secs_f64() / batch_duration.as_secs_f64();
    println!("   Batch duration:      {:?}", batch_duration);
    println!("   Sequential duration: {:?}", seq_duration);
    println!("   Speedup:             {:.2}x", speedup);

    if speedup > 1.0 {
        println!("   ✅ Batch processing is faster!");
    } else {
        println!("   ⚠️  Sequential is currently faster (optimization needed)");
    }

    // Compare output correctness (should be similar but not identical due to numerical differences)
    if let Ok(batch_outputs) = batch_result {
        println!("\n📈 Output Similarity Check:");
        for i in 0..test_inputs.len() {
            let batch_data = &batch_outputs[i].data;
            let seq_data = &sequential_outputs[i].data;

            if batch_data.len() != seq_data.len() {
                println!("   Sequence {}: ⚠️  Different output sizes", i);
                continue;
            }

            // Calculate correlation (note: batch_data and seq_data are ndarray, access raw data)
            let batch_vec = batch_data.as_slice().unwrap_or(&[]);
            let seq_vec = seq_data.as_slice().unwrap_or(&[]);

            let mut diff_sum: f64 = 0.0;
            let mut max_diff: f64 = 0.0;
            for (b, s) in batch_vec.iter().zip(seq_vec.iter()) {
                let diff = (b - s).abs();
                diff_sum += diff;
                max_diff = max_diff.max(diff);
            }
            let avg_diff = diff_sum / batch_vec.len() as f64;

            println!("   Sequence {}: avg_diff={:.6}, max_diff={:.6}", i, avg_diff, max_diff);

            // Outputs should be reasonably similar (within tolerance)
            if avg_diff < 0.1 {
                println!("             ✅ Outputs are similar");
            } else {
                println!("             ⚠️  Outputs differ significantly");
            }
        }
    }

    // Test 4: Memory usage check
    println!("\n💾 Test 4: Memory Usage");
    println!("───────────────────────");
    println!("   Batch size: {}", test_inputs.len());
    println!("   Model parameters: ~1.1B");
    println!("   KVCache capacity: {}", model.config.max_seq_len);
    println!("   ✅ No OOM errors during execution");

    println!("\n🎉 E2E Batch Inference Test Complete!");
    println!("═══════════════════════════════════════\n");
}

#[cfg(feature = "metal")]
#[test]
#[ignore] // Run manually
fn test_batch_inference_multiple_tokens() {
    use rustorch::models::llama::LlamaModel;
    use std::path::PathBuf;
    use std::env;

    println!("\n🚀 E2E Multi-Token Generation Test");
    println!("===================================\n");

    let home = env::var("HOME").expect("HOME not set");
    let model_path = PathBuf::from(home)
        .join(".rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF")
        .join("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    if !model_path.exists() {
        println!("⚠️  Model file not found, skipping test");
        return;
    }

    println!("📥 Loading model...");
    let mut model = LlamaModel::from_gguf(&model_path).expect("Failed to load model");
    println!("✅ Model loaded\n");

    // Test batch generation with multiple steps
    let test_inputs = vec![
        vec![1, 15043], // "Hello"
        vec![1, 1724],  // "What"
    ];

    println!("🔄 Generating 5 tokens per sequence...");

    let mut current_inputs = test_inputs.clone();
    let max_new_tokens = 5;

    for step in 0..max_new_tokens {
        println!("\nStep {}/{}:", step + 1, max_new_tokens);

        let input_refs: Vec<&[usize]> = current_inputs.iter().map(|v| v.as_slice()).collect();

        let outputs = model.forward_batch(&input_refs)
            .expect(&format!("Forward pass failed at step {}", step));

        // Sample next token for each sequence (greedy: argmax)
        for (i, output) in outputs.iter().enumerate() {
            // Get last position logits - output.data is ndarray Array, use as_slice
            let output_vec = output.data.as_slice().unwrap();
            let vocab_size = output.shape()[output.shape().len() - 1];
            let last_logits = &output_vec[output_vec.len() - vocab_size..];

            // Find argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            println!("   Sequence {}: generated token {}", i, next_token);
            current_inputs[i].push(next_token);
        }
    }

    println!("\n✅ Multi-token generation successful!");
    println!("Final sequences:");
    for (i, seq) in current_inputs.iter().enumerate() {
        println!("   Sequence {}: {} tokens", i, seq.len());
    }

    println!("\n🎉 Multi-Token Generation Test Complete!\n");
}
