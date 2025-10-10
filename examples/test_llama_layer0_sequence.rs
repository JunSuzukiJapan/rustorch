//! Test full Layer 0 sequence to reproduce V projection crash
//! Layer 0の完全な実行シーケンスでV projectionクラッシュを再現

use rustorch::formats::gguf::GGUFLoader;
use rustorch::gpu::metal_kernels::MetalKernelExecutor;
use std::path::Path;

/// RMS Normalization
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

        let mean_sq: f32 = row.iter().map(|&v| v * v).sum::<f32>() / (hidden_size as f32);
        let rms = (mean_sq + eps).sqrt();
        let scale = 1.0 / rms;

        for i in 0..hidden_size {
            output[offset + i] = row[i] * scale * weight[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing full Layer 0 sequence");
    println!("==================================\n");

    // Load GGUF model
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
    println!("1. Loading GGUF model...");
    let loader = GGUFLoader::from_file(model_path)?;
    println!("   ✅ GGUF loaded\n");

    // Initialize Metal executor
    println!("2. Initializing Metal executor...");
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    let executor = executor_guard.as_ref()
        .ok_or("Metal executor not initialized")?;
    println!("   ✅ Metal executor initialized\n");

    // Load embeddings
    println!("3. Loading token embeddings...");
    let token_emb = loader.load_tensor("token_embd.weight")?;
    println!("   Token embedding shape: {:?}\n", token_emb.data.shape());

    // Simulate token IDs matching the real crash scenario (15 tokens)
    let input_ids = vec![1usize, 29966, 29989, 1792, 29989, 29958, 13, 29896, 29966, 29989, 465, 22137, 29989, 29958, 2];
    let seq_len = input_ids.len();

    println!("   Using {} tokens (matching real crash scenario)", seq_len);
    let d_model = 2048;

    println!("4. Token embedding lookup...");
    let emb_data = token_emb.data.as_slice()
        .ok_or("Failed to get embedding data")?;
    let mut x_f32 = Vec::new();

    for &token_id in &input_ids {
        let start = token_id * d_model;
        let end = start + d_model;
        x_f32.extend(emb_data[start..end].iter().map(|&v| v as f32));
    }
    println!("   ✅ Embeddings created: {} values\n", x_f32.len());

    // Load Layer 0 weights
    println!("5. Loading Layer 0 weights...");
    let ln1_weight = loader.load_tensor("blk.0.attn_norm.weight")?;
    let q_weight = loader.load_tensor("blk.0.attn_q.weight")?;
    let k_weight = loader.load_tensor("blk.0.attn_k.weight")?;
    let v_weight = loader.load_tensor("blk.0.attn_v.weight")?;
    let o_weight = loader.load_tensor("blk.0.attn_output.weight")?;

    println!("   LN1 shape: {:?}", ln1_weight.data.shape());
    println!("   Q shape: {:?}", q_weight.data.shape());
    println!("   K shape: {:?}", k_weight.data.shape());
    println!("   V shape: {:?}", v_weight.data.shape());
    println!("   O shape: {:?}\n", o_weight.data.shape());

    // Convert to f32
    println!("6. Converting weights to f32...");
    let ln1_gamma_f32: Vec<f32> = ln1_weight.data.iter().map(|&v| v as f32).collect();
    let q_weight_f32: Vec<f32> = q_weight.data.iter().map(|&v| v as f32).collect();
    let k_weight_f32: Vec<f32> = k_weight.data.iter().map(|&v| v as f32).collect();
    let v_weight_f32: Vec<f32> = v_weight.data.iter().map(|&v| v as f32).collect();
    let o_weight_f32: Vec<f32> = o_weight.data.iter().map(|&v| v as f32).collect();
    println!("   ✅ Weights converted\n");

    // Get dimensions
    let q_out_dim = q_weight.data.shape()[1];
    let k_out_dim = k_weight.data.shape()[1];
    let v_out_dim = v_weight.data.shape()[1];

    println!("7. Dimensions:");
    println!("   seq_len: {}", seq_len);
    println!("   d_model: {}", d_model);
    println!("   q_out_dim: {}", q_out_dim);
    println!("   k_out_dim: {}", k_out_dim);
    println!("   v_out_dim: {}\n", v_out_dim);

    // Step 1: RMS Norm
    println!("8. RMS Normalization...");
    let mut x_ln1 = vec![0.0f32; x_f32.len()];
    rms_norm_f32(&x_f32, &ln1_gamma_f32, &mut x_ln1, seq_len, d_model, 1e-5);
    println!("   ✅ RMS Norm complete");
    println!("   x_ln1[0..5]: {:?}\n", &x_ln1[0..5]);

    // Step 2: Q projection
    println!("9. Q projection ({}x{} x {}x{})...", seq_len, d_model, d_model, q_out_dim);
    let mut q_out = vec![0.0f32; seq_len * q_out_dim];
    executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, d_model, q_out_dim)?;
    println!("   ✅ Q projection complete");
    println!("   q_out[0..5]: {:?}\n", &q_out[0..5]);

    // Step 3: K projection
    println!("10. K projection ({}x{} x {}x{})...", seq_len, d_model, d_model, k_out_dim);
    let mut k_out = vec![0.0f32; seq_len * k_out_dim];
    executor.matmul_f32(&x_ln1, &k_weight_f32, &mut k_out, seq_len, d_model, k_out_dim)?;
    println!("   ✅ K projection complete");
    println!("   k_out[0..5]: {:?}\n", &k_out[0..5]);

    // Step 4: V projection - THE CRITICAL ONE
    println!("11. V projection ({}x{} x {}x{}) - CRITICAL TEST...", seq_len, d_model, d_model, v_out_dim);
    println!("    This follows the exact sequence from the crash!");
    println!("    Previous operations: embedding lookup -> RMS norm -> Q matmul -> K matmul");
    println!();

    let mut v_out = vec![0.0f32; seq_len * v_out_dim];

    println!("    About to call matmul_f32 for V...");
    match executor.matmul_f32(&x_ln1, &v_weight_f32, &mut v_out, seq_len, d_model, v_out_dim) {
        Ok(_) => {
            println!("   ✅ V projection complete");
            println!("   v_out[0..5]: {:?}\n", &v_out[0..5]);

            // Continue with O projection to complete attention
            println!("12. O projection (output)...");
            let mut attn_out = vec![0.0f32; seq_len * d_model];
            // Simplified: use q_out as attention output
            attn_out.copy_from_slice(&q_out[0..seq_len * d_model]);

            let mut attn_proj = vec![0.0f32; seq_len * d_model];
            executor.matmul_f32(&attn_out, &o_weight_f32, &mut attn_proj, seq_len, d_model, d_model)?;
            println!("   ✅ O projection complete");
            println!("   attn_proj[0..5]: {:?}\n", &attn_proj[0..5]);

            println!("✅ Full Layer 0 sequence completed successfully!");
        }
        Err(e) => {
            println!("   ❌ V projection FAILED: {}", e);
            println!("\n❌ Crash reproduced at V projection!");
            return Err(e.into());
        }
    }

    Ok(())
}
