//! Test V projection with real GGUF weights
//! 実際のGGUF weightsを使ってV projectionをテスト

use rustorch::formats::gguf::GGUFLoader;
use rustorch::gpu::metal_kernels::MetalKernelExecutor;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing V projection with real GGUF weights");
    println!("================================================\n");

    // Load GGUF model
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
    println!("1. Loading GGUF model: {:?}", model_path);
    let loader = GGUFLoader::from_file(model_path)?;
    println!("   ✅ GGUF loaded\n");

    // Initialize Metal executor
    println!("2. Initializing Metal executor...");
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    let executor = executor_guard.as_ref()
        .ok_or("Metal executor not initialized")?;
    println!("   ✅ Metal executor initialized\n");

    // Load layer 0 weights
    println!("3. Loading layer 0 attention weights...");
    let q_weight = loader.load_tensor("blk.0.attn_q.weight")?;
    let k_weight = loader.load_tensor("blk.0.attn_k.weight")?;
    let v_weight = loader.load_tensor("blk.0.attn_v.weight")?;

    println!("   Q weight shape: {:?}", q_weight.data.shape());
    println!("   K weight shape: {:?}", k_weight.data.shape());
    println!("   V weight shape: {:?}", v_weight.data.shape());
    println!();

    // Convert to f32
    println!("4. Converting weights to f32...");
    let q_weight_f32: Vec<f32> = q_weight.data.iter().map(|&v| v as f32).collect();
    let k_weight_f32: Vec<f32> = k_weight.data.iter().map(|&v| v as f32).collect();
    let v_weight_f32: Vec<f32> = v_weight.data.iter().map(|&v| v as f32).collect();

    println!("   Q weight len: {}", q_weight_f32.len());
    println!("   K weight len: {}", k_weight_f32.len());
    println!("   V weight len: {}", v_weight_f32.len());
    println!("   V weight[0..10]: {:?}", &v_weight_f32[0..10]);
    println!();

    // Create test input
    let seq_len = 15;
    let d_model = 2048;
    let q_shape = q_weight.data.shape();
    let k_shape = k_weight.data.shape();
    let v_shape = v_weight.data.shape();

    let q_out_dim = q_shape[1];
    let k_out_dim = k_shape[1];
    let v_out_dim = v_shape[1];

    println!("5. Test dimensions:");
    println!("   seq_len: {}", seq_len);
    println!("   d_model: {}", d_model);
    println!("   q_out_dim: {}", q_out_dim);
    println!("   k_out_dim: {}", k_out_dim);
    println!("   v_out_dim: {}", v_out_dim);
    println!();

    // Create test input (RMS normalized values)
    println!("6. Creating test input...");
    let x_ln1: Vec<f32> = (0..seq_len * d_model).map(|i| (i % 100) as f32 * 0.01).collect();
    println!("   x_ln1 len: {}", x_ln1.len());
    println!("   x_ln1[0..10]: {:?}", &x_ln1[0..10]);
    println!();

    // Test Q projection
    println!("7. Testing Q projection...");
    let mut q_out = vec![0.0f32; seq_len * q_out_dim];
    match executor.matmul_f32(&x_ln1, &q_weight_f32, &mut q_out, seq_len, d_model, q_out_dim) {
        Ok(_) => {
            println!("   ✅ Q projection succeeded");
            println!("   q_out[0..5]: {:?}\n", &q_out[0..5]);
        }
        Err(e) => {
            println!("   ❌ Q projection failed: {}\n", e);
            return Ok(());
        }
    }

    // Test K projection
    println!("8. Testing K projection...");
    let mut k_out = vec![0.0f32; seq_len * k_out_dim];
    match executor.matmul_f32(&x_ln1, &k_weight_f32, &mut k_out, seq_len, d_model, k_out_dim) {
        Ok(_) => {
            println!("   ✅ K projection succeeded");
            println!("   k_out[0..5]: {:?}\n", &k_out[0..5]);
        }
        Err(e) => {
            println!("   ❌ K projection failed: {}\n", e);
            return Ok(());
        }
    }

    // Test V projection - THE CRITICAL TEST
    println!("9. Testing V projection with real weights...");
    println!("   This is where it crashes in the actual model!");
    println!("   v_weight_f32 len: {}", v_weight_f32.len());
    println!("   Expected size: {} x {} = {}", d_model, v_out_dim, d_model * v_out_dim);
    println!();

    let mut v_out = vec![0.0f32; seq_len * v_out_dim];

    println!("   Calling matmul_f32...");
    match executor.matmul_f32(&x_ln1, &v_weight_f32, &mut v_out, seq_len, d_model, v_out_dim) {
        Ok(_) => {
            println!("   ✅ V projection succeeded!");
            println!("   v_out[0..5]: {:?}", &v_out[0..5]);
            println!("\n✅ All projections with real weights succeeded!");
        }
        Err(e) => {
            println!("   ❌ V projection failed: {}", e);
            println!("\n❌ V projection with real weights crashed");
        }
    }

    Ok(())
}
