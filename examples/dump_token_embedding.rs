/// Dump Token 1 embedding vector from RusTorch
///
/// Compare with llama.cpp to verify weight extraction

use rustorch::formats::gguf::GGUFLoader;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Dumping Token 1 Embedding\n");

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            std::env::var("HOME").unwrap() +
            "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        });

    println!("📂 Model: {}", model_path.split('/').last().unwrap_or(&model_path));

    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| format!("GGUF load failed: {}", e))?;

    // First, load output.weight to check its type
    println!("🔍 Checking output.weight type first...");
    let _output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| format!("Output tensor load failed: {}", e))?;

    println!("\n🔍 Now loading token_embd.weight...");
    let embd_tensor = loader.load_tensor("token_embd.weight")
        .map_err(|e| format!("Tensor load failed: {}", e))?;

    println!("✅ token_embd.weight loaded");
    println!("   Shape: {:?}", embd_tensor.shape());

    let data: Vec<f64> = embd_tensor.data.iter().cloned().collect();

    // Check multiple tokens
    let hidden_size = 2048;
    let tokens_to_check = vec![0, 1, 100, 1000, 10000, 15043, 25323];

    println!("\n📊 Checking {} tokens:", tokens_to_check.len());

    for &token_id in &tokens_to_check {
        let start_idx = token_id * hidden_size;
        let end_idx = start_idx + hidden_size;
        let token_embd = &data[start_idx..end_idx];

        let mean: f64 = token_embd.iter().sum::<f64>() / hidden_size as f64;
        let abs_mean: f64 = token_embd.iter().map(|x| x.abs()).sum::<f64>() / hidden_size as f64;
        let max = token_embd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = token_embd.iter().cloned().fold(f64::INFINITY, f64::min);
        let rms = (token_embd.iter().map(|x| x*x).sum::<f64>() / hidden_size as f64).sqrt();

        println!("\nToken {}: mean={:.6}, abs_mean={:.6}, rms={:.6}, max={:.6}, min={:.6}",
            token_id, mean, abs_mean, rms, max, min);

        print!("  First 5: [");
        for i in 0..5 {
            print!("{:.6}", token_embd[i]);
            if i < 4 { print!(", "); }
        }
        println!("]");
    }

    // Write token 1 to file for detailed comparison
    let token_id = 1;
    let start_idx = token_id * hidden_size;
    let end_idx = start_idx + hidden_size;
    let token_1_embd = &data[start_idx..end_idx];

    println!("\n📝 Writing token 1 to /tmp/rustorch_token1_embedding.txt");
    let mut file = std::fs::File::create("/tmp/rustorch_token1_embedding.txt")?;
    for (i, val) in token_1_embd.iter().enumerate() {
        writeln!(file, "{} {:.10}", i, val)?;
    }
    println!("✅ Done! {} values written", token_1_embd.len());

    // Also check Q projection weight
    println!("\n🔍 Checking blk.0.attn_q.weight...");
    let q_weight_tensor = loader.load_tensor("blk.0.attn_q.weight")?;
    println!("✅ Q weight loaded");
    println!("   Shape: {:?}", q_weight_tensor.shape());

    let q_data: Vec<f64> = q_weight_tensor.data.iter().cloned().collect();
    let q_mean: f64 = q_data.iter().sum::<f64>() / q_data.len() as f64;
    let q_abs_mean: f64 = q_data.iter().map(|x| x.abs()).sum::<f64>() / q_data.len() as f64;
    let q_rms = (q_data.iter().map(|x| x*x).sum::<f64>() / q_data.len() as f64).sqrt();
    let q_max = q_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let q_min = q_data.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("\nQ weight: mean={:.6}, abs_mean={:.6}, rms={:.6}, max={:.6}, min={:.6}",
        q_mean, q_abs_mean, q_rms, q_max, q_min);
    print!("First 10: [");
    for i in 0..10 {
        print!("{:.6}", q_data[i]);
        if i < 9 { print!(", "); }
    }
    println!("]");

    Ok(())
}
