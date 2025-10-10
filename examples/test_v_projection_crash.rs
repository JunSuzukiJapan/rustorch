//! Test V projection matmul crash
//! V projection matmul_f32のクラッシュを再現・調査するテスト

use rustorch::gpu::metal_kernels::MetalKernelExecutor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing V projection matmul crash");
    println!("======================================\n");

    // Initialize Metal executor
    println!("1. Initializing Metal executor...");
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    let executor = executor_guard.as_ref()
        .ok_or("Metal executor not initialized")?;
    println!("   ✅ Metal executor initialized\n");

    // Test parameters matching the crash scenario
    let seq_len = 15;
    let d_model = 2048;
    let v_out_dim = 256;

    println!("2. Test parameters:");
    println!("   seq_len: {}", seq_len);
    println!("   d_model: {}", d_model);
    println!("   v_out_dim: {}\n", v_out_dim);

    // Create test data
    println!("3. Creating test data...");
    let x_ln1_len = seq_len * d_model;
    let v_weight_len = d_model * v_out_dim;
    let v_out_len = seq_len * v_out_dim;

    println!("   x_ln1 len: {}", x_ln1_len);
    println!("   v_weight len: {}", v_weight_len);
    println!("   v_out len: {}\n", v_out_len);

    // Initialize with small test values
    let x_ln1: Vec<f32> = (0..x_ln1_len).map(|i| (i % 100) as f32 * 0.01).collect();
    let v_weight: Vec<f32> = (0..v_weight_len).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut v_out: Vec<f32> = vec![0.0f32; v_out_len];

    println!("   ✅ Test data created");
    println!("   x_ln1[0..5]: {:?}", &x_ln1[0..5]);
    println!("   v_weight[0..5]: {:?}", &v_weight[0..5]);
    println!();

    // Test 1: Small matmul (should work)
    println!("4. Test 1: Small matmul (2x2 x 2x2)...");
    let small_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let small_b = vec![5.0f32, 6.0, 7.0, 8.0];
    let mut small_c = vec![0.0f32; 4];

    match executor.matmul_f32(&small_a, &small_b, &mut small_c, 2, 2, 2) {
        Ok(_) => {
            println!("   ✅ Small matmul succeeded");
            println!("   Result: {:?}\n", small_c);
        }
        Err(e) => {
            println!("   ❌ Small matmul failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 2: Q projection size (this worked)
    println!("5. Test 2: Q projection size (15x2048 x 2048x2048)...");
    let q_out_dim = 2048;
    let q_weight: Vec<f32> = vec![0.01f32; d_model * q_out_dim];
    let mut q_out: Vec<f32> = vec![0.0f32; seq_len * q_out_dim];

    match executor.matmul_f32(&x_ln1, &q_weight, &mut q_out, seq_len, d_model, q_out_dim) {
        Ok(_) => {
            println!("   ✅ Q projection succeeded");
            println!("   q_out[0..5]: {:?}\n", &q_out[0..5]);
        }
        Err(e) => {
            println!("   ❌ Q projection failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 3: K projection size (this worked)
    println!("6. Test 3: K projection size (15x2048 x 2048x256)...");
    let k_out_dim = 256;
    let k_weight: Vec<f32> = vec![0.01f32; d_model * k_out_dim];
    let mut k_out: Vec<f32> = vec![0.0f32; seq_len * k_out_dim];

    match executor.matmul_f32(&x_ln1, &k_weight, &mut k_out, seq_len, d_model, k_out_dim) {
        Ok(_) => {
            println!("   ✅ K projection succeeded");
            println!("   k_out[0..5]: {:?}\n", &k_out[0..5]);
        }
        Err(e) => {
            println!("   ❌ K projection failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 4: V projection size (THIS CRASHES)
    println!("7. Test 4: V projection size (15x2048 x 2048x256) - THE CRASH...");
    println!("   About to call matmul_f32 with:");
    println!("   - input: len={} ({}x{})", x_ln1.len(), seq_len, d_model);
    println!("   - weight: len={} ({}x{})", v_weight.len(), d_model, v_out_dim);
    println!("   - output: len={} ({}x{})", v_out.len(), seq_len, v_out_dim);
    println!("   - dimensions: M={}, K={}, N={}", seq_len, d_model, v_out_dim);
    println!();

    match executor.matmul_f32(&x_ln1, &v_weight, &mut v_out, seq_len, d_model, v_out_dim) {
        Ok(_) => {
            println!("   ✅ V projection succeeded!");
            println!("   v_out[0..5]: {:?}", &v_out[0..5]);
            println!("\n✅ All tests passed!");
        }
        Err(e) => {
            println!("   ❌ V projection failed: {}", e);
            println!("\n❌ V projection crashed as expected");
        }
    }

    Ok(())
}
