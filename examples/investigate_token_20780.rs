/// Token 20780が常に予測される原因を調査
///
/// すべての異なる入力で同じToken 20780が予測されるのは異常。
/// LM headのweight列を詳しく調べる。

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;
use rustorch::formats::gguf::GGUFLoader;

fn main() -> F32Result<()> {
    println!("🔍 Token 20780の謎を解明\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    // GGUFLoaderで直接weightを読み込む
    println!("📂 output.weightを直接読み込み中...");
    let loader = GGUFLoader::from_file(&model_path)
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("GGUF load failed: {}", e)))?;

    let output_tensor = loader.load_tensor("output.weight")
        .map_err(|e| rustorch::hybrid_f32::error::F32Error::device_error(format!("Tensor load failed: {}", e)))?;

    println!("✅ output.weight読み込み完了");
    println!("   Shape: {:?}", output_tensor.shape());
    println!("   Expected: [2048, 32000] (hidden_size, vocab_size)");

    let output_data: Vec<f64> = output_tensor.data.iter().cloned().collect();
    println!("   Data length: {}", output_data.len());

    // Token 20780のweight列を抽出
    // Shape [2048, 32000] で row-major の場合:
    // Token tの列 = [data[0*32000 + t], data[1*32000 + t], ..., data[2047*32000 + t]]

    println!("\n═══════════════════════════════════════");
    println!("📊 Token 20780のWeight列を調査");
    println!("═══════════════════════════════════════\n");

    let token_id = 20780;
    let hidden_size = 2048;
    let vocab_size = 32000;

    let mut token_20780_weights = Vec::new();
    for dim in 0..hidden_size {
        let idx = dim * vocab_size + token_id;
        token_20780_weights.push(output_data[idx]);
    }

    // 統計を計算
    let sum: f64 = token_20780_weights.iter().sum();
    let mean = sum / hidden_size as f64;
    let max = token_20780_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = token_20780_weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let abs_sum: f64 = token_20780_weights.iter().map(|x| x.abs()).sum();
    let abs_mean = abs_sum / hidden_size as f64;

    println!("Token 20780のweight統計:");
    println!("   Sum: {:.8}", sum);
    println!("   Mean: {:.8}", mean);
    println!("   Abs Mean: {:.8}", abs_mean);
    println!("   Max: {:.8}", max);
    println!("   Min: {:.8}", min);

    println!("\n最初の10個の値:");
    for i in 0..10 {
        println!("   [{}]: {:.8}", i, token_20780_weights[i]);
    }

    // Token 450と比較
    println!("\n═══════════════════════════════════════");
    println!("📊 Token 450 (\" The\")のWeight列を調査");
    println!("═══════════════════════════════════════\n");

    let token_450 = 450;
    let mut token_450_weights = Vec::new();
    for dim in 0..hidden_size {
        let idx = dim * vocab_size + token_450;
        token_450_weights.push(output_data[idx]);
    }

    let sum_450: f64 = token_450_weights.iter().sum();
    let mean_450 = sum_450 / hidden_size as f64;
    let max_450 = token_450_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_450 = token_450_weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let abs_sum_450: f64 = token_450_weights.iter().map(|x| x.abs()).sum();
    let abs_mean_450 = abs_sum_450 / hidden_size as f64;

    println!("Token 450のweight統計:");
    println!("   Sum: {:.8}", sum_450);
    println!("   Mean: {:.8}", mean_450);
    println!("   Abs Mean: {:.8}", abs_mean_450);
    println!("   Max: {:.8}", max_450);
    println!("   Min: {:.8}", min_450);

    println!("\n最初の10個の値:");
    for i in 0..10 {
        println!("   [{}]: {:.8}", i, token_450_weights[i]);
    }

    // 他のいくつかのトークンも確認
    println!("\n═══════════════════════════════════════");
    println!("📊 他のトークンとの比較");
    println!("═══════════════════════════════════════\n");

    let test_tokens = vec![
        (1, "<s>"),
        (450, " The"),
        (1552, "the"),
        (3681, "Paris"),
        (12517, "?"),
        (20780, "?"),
    ];

    println!("| Token | Name | Abs Mean Weight | Sum |");
    println!("|-------|------|-----------------|-----|");

    for (token_id, name) in test_tokens {
        let mut weights = Vec::new();
        for dim in 0..hidden_size {
            let idx = dim * vocab_size + token_id;
            weights.push(output_data[idx]);
        }
        let sum: f64 = weights.iter().sum();
        let abs_mean = weights.iter().map(|x| x.abs()).sum::<f64>() / hidden_size as f64;
        println!("| {} | {} | {:.8} | {:.8} |", token_id, name, abs_mean, sum);
    }

    // 手動でlogitを計算してみる
    println!("\n═══════════════════════════════════════");
    println!("🧮 手動Logit計算");
    println!("═══════════════════════════════════════\n");

    println!("モデルを読み込んで実際のhidden stateを取得...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    let input = vec![1]; // BOS
    let output = model.forward(&input)?;
    let logits = output.as_slice();

    println!("実際のlogit:");
    println!("   Token 450: {:.8}", logits[450]);
    println!("   Token 20780: {:.8}", logits[20780]);
    println!("   Token 12517: {:.8}", logits[12517]);

    println!("\n🔍 分析:");
    if abs_mean > abs_mean_450 * 2.0 {
        println!("   ⚠️  Token 20780のweight値が異常に大きい！");
        println!("   これが常にToken 20780が予測される原因の可能性大");
    } else {
        println!("   ✅ Weight値は正常範囲内");
        println!("   問題は別の場所にある可能性");
    }

    Ok(())
}
