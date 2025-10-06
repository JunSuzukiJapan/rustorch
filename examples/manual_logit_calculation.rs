/// 手動でlogitを計算してmatmulの結果と比較
///
/// Token 20780が常に最高logitになる理由を突き止める

use rustorch::formats::gguf::GGUFLoader;
use rustorch::hybrid_f32::error::{F32Result, F32Error};
use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};

fn main() -> F32Result<()> {
    println!("🧮 手動Logit計算テスト\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    // モデルのforward pass
    println!("📂 モデル読み込みと推論...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    let input = vec![1]; // BOS
    let logits = model.forward(&input)?;
    let logits_data = logits.as_slice();

    println!("✅ Forward pass完了");
    println!("   Logit Token 450: {:.8}", logits_data[450]);
    println!("   Logit Token 20780: {:.8}", logits_data[20780]);
    println!("   Logit Token 12517: {:.8}", logits_data[12517]);

    // 最後のhidden stateを取得（/tmp/hidden_state.txtに保存されている）
    println!("\n📄 Hidden stateを読み込み...");
    let hidden_state_str = std::fs::read_to_string("/tmp/hidden_state.txt")
        .map_err(|e| F32Error::device_error(format!("Failed to read hidden state: {}", e)))?;

    let hidden_state: Vec<f64> = hidden_state_str
        .lines()
        .map(|line| line.trim().parse::<f64>().unwrap_or(0.0))
        .collect();

    println!("✅ Hidden state読み込み完了: {} 要素", hidden_state.len());
    println!("   First 10: {:?}", &hidden_state[0..10]);

    // output.weightを直接読み込み
    println!("\n📂 output.weightを読み込み...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| F32Error::device_error(format!("Failed to load GGUF: {}", e)))?;

    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| F32Error::device_error(format!("Failed to load tensor: {}", e)))?;

    let output_data: Vec<f64> = output_tensor.data.iter().cloned().collect();
    println!("✅ output.weight読み込み完了");
    println!("   Shape: {:?}", output_tensor.shape());
    println!("   Data length: {}", output_data.len());

    // 手動でToken 450のlogitを計算
    println!("\n═══════════════════════════════════════");
    println!("🧮 手動計算: Token 450");
    println!("═══════════════════════════════════════");

    let hidden_size = 2048;
    let vocab_size = 32000;
    let token_450 = 450;

    // output.weightは [hidden_size, vocab_size] = [2048, 32000]
    // Token 450の列 = [output_data[0*32000 + 450], output_data[1*32000 + 450], ..., output_data[2047*32000 + 450]]
    let mut logit_450_manual = 0.0f64;
    for dim in 0..hidden_size {
        let weight_idx = dim * vocab_size + token_450;
        logit_450_manual += hidden_state[dim] * output_data[weight_idx];
    }

    println!("手動計算 logit: {:.8}", logit_450_manual);
    println!("Matmul logit:    {:.8}", logits_data[450] as f64);
    println!("差分:            {:.8}", (logit_450_manual - logits_data[450] as f64).abs());

    if (logit_450_manual - logits_data[450] as f64).abs() < 0.01 {
        println!("✅ 一致！");
    } else {
        println!("❌ 不一致！");
    }

    // Token 20780も計算
    println!("\n═══════════════════════════════════════");
    println!("🧮 手動計算: Token 20780");
    println!("═══════════════════════════════════════");

    let token_20780 = 20780;
    let mut logit_20780_manual = 0.0f64;
    for dim in 0..hidden_size {
        let weight_idx = dim * vocab_size + token_20780;
        logit_20780_manual += hidden_state[dim] * output_data[weight_idx];
    }

    println!("手動計算 logit: {:.8}", logit_20780_manual);
    println!("Matmul logit:    {:.8}", logits_data[20780] as f64);
    println!("差分:            {:.8}", (logit_20780_manual - logits_data[20780] as f64).abs());

    if (logit_20780_manual - logits_data[20780] as f64).abs() < 0.01 {
        println!("✅ 一致！");
    } else {
        println!("❌ 不一致！");
    }

    // 最も寄与が大きい次元を見つける
    println!("\n═══════════════════════════════════════");
    println!("🔍 Token 20780で最も寄与が大きい次元");
    println!("═══════════════════════════════════════\n");

    let mut contributions: Vec<(usize, f64)> = Vec::new();
    for dim in 0..hidden_size {
        let weight_idx = dim * vocab_size + token_20780;
        let contrib = hidden_state[dim] * output_data[weight_idx];
        contributions.push((dim, contrib));
    }

    contributions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Top 10寄与（絶対値）:");
    for (i, (dim, contrib)) in contributions.iter().take(10).enumerate() {
        let hidden_val = hidden_state[*dim];
        let weight_val = output_data[dim * vocab_size + token_20780];
        println!("   {}. dim[{}]: contrib={:.6}, hidden={:.6}, weight={:.6}",
                 i + 1, dim, contrib, hidden_val, weight_val);
    }

    // Token 450でも同様に
    println!("\n🔍 Token 450で最も寄与が大きい次元\n");

    let mut contributions_450: Vec<(usize, f64)> = Vec::new();
    for dim in 0..hidden_size {
        let weight_idx = dim * vocab_size + token_450;
        let contrib = hidden_state[dim] * output_data[weight_idx];
        contributions_450.push((dim, contrib));
    }

    contributions_450.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    println!("Top 10寄与（絶対値）:");
    for (i, (dim, contrib)) in contributions_450.iter().take(10).enumerate() {
        let hidden_val = hidden_state[*dim];
        let weight_val = output_data[dim * vocab_size + token_450];
        println!("   {}. dim[{}]: contrib={:.6}, hidden={:.6}, weight={:.6}",
                 i + 1, dim, contrib, hidden_val, weight_val);
    }

    Ok(())
}
