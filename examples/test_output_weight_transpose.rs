/// output.weightのtransposeが必要かテスト
///
/// 仮説：output.weightがcolumn-majorで格納されているが、
/// row-majorとして解釈しているため、予測が間違っている

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 output.weightのtransposeテスト\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 モデル読み込み中...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // 現在の実装での予測
    println!("\n═══════════════════════════════════════");
    println!("テスト1: 現在の実装（transposeなし）");
    println!("═══════════════════════════════════════");

    let input = vec![1]; // BOS
    let logits_normal = model.forward(&input)?;
    let logits_normal_data = logits_normal.as_slice();

    let mut indexed_normal: Vec<(usize, f32)> = logits_normal_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_normal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 5予測:");
    for (i, (token, logit)) in indexed_normal.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    println!("\n特定トークン:");
    println!("   Token 450 (\" The\"): {:.6}", logits_normal_data[450]);
    println!("   Token 20780: {:.6}", logits_normal_data[20780]);

    // transposeしたweightで再テスト
    println!("\n═══════════════════════════════════════");
    println!("テスト2: output.weightをtransposeして試す");
    println!("═══════════════════════════════════════");

    // output.weightを取得してtranspose
    let lm_head = model.weights.get("output.weight")
        .ok_or(rustorch::hybrid_f32::error::F32Error::device_error("output.weight not found"))?;

    println!("\n元のoutput.weight shape: {:?}", lm_head.shape());
    let lm_head_transposed = lm_head.transpose()?;
    println!("Transposed shape: {:?}", lm_head_transposed.shape());

    // 新しいモデルインスタンスを作成
    println!("\n新しいモデルインスタンスを作成...");
    let mut model2 = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // transposeしたweightを設定
    model2.weights.insert("output.weight".to_string(), lm_head_transposed.clone());

    let logits_transposed = model2.forward(&input)?;
    let logits_transposed_data = logits_transposed.as_slice();

    let mut indexed_transposed: Vec<(usize, f32)> = logits_transposed_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_transposed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 5予測:");
    for (i, (token, logit)) in indexed_transposed.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    println!("\n特定トークン:");
    println!("   Token 450 (\" The\"): {:.6}", logits_transposed_data[450]);
    println!("   Token 20780: {:.6}", logits_transposed_data[20780]);

    // 比較
    println!("\n═══════════════════════════════════════");
    println!("🔍 比較結果");
    println!("═══════════════════════════════════════\n");

    println!("通常版:");
    println!("   Top token: {} (logit: {:.6})", indexed_normal[0].0, indexed_normal[0].1);
    println!("   Token 450: {:.6}", logits_normal_data[450]);
    println!();

    println!("Transpose版:");
    println!("   Top token: {} (logit: {:.6})", indexed_transposed[0].0, indexed_transposed[0].1);
    println!("   Token 450: {:.6}", logits_transposed_data[450]);
    println!();

    if indexed_transposed[0].0 == 450 {
        println!("✅ SUCCESS! Transposeで正しい予測になった！");
        println!("   → output.weightはcolumn-majorで格納されている");
        println!("   → row-majorとして解釈していたのが問題");
    } else if indexed_transposed[0].0 == indexed_normal[0].0 {
        println!("❌ Transposeしても同じ予測");
        println!("   → Layoutの問題ではない");
    } else {
        println!("⚠️  Transposeで異なる予測だが、まだ正しくない");
        println!("   → Token {}", indexed_transposed[0].0);
    }

    Ok(())
}
