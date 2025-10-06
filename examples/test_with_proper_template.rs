/// 正しいチャットテンプレートを使用してllama.cppと比較可能な結果を得る

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 チャットテンプレート付きテスト\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 モデル読み込み中...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // TinyLlamaのチャットテンプレート
    // llama-tokenizeで確認したトークンID列
    let chat_template = vec![
        1,      // <s>
        529,    // <
        29989,  // |
        5205,   // system
        29989,  // |
        29958,  // >
        13,     // \n
        3492,   // You
        526,    // are
        263,    // a
        8444,   // helpful
        20255,  // assistant
        29889,  // .
        2,      // </s>
        29871,  // (space)
        13,     // \n
        29966,  // <
        29989,  // |
        1792,   // user
        29989,  // |
        29958,  // >
        13,     // \n
        5618,   // What
        338,    // is
        278,    // the
        7483,   // capital
        310,    // of
        3444,   // France
        29973,  // ?
        2,      // </s>
        29871,  // (space)
        13,     // \n
        29966,  // <
        29989,  // |
        465,    // ass
        22137,  // istant
        29989,  // |
        29958,  // >
        13,     // \n
    ];

    println!("📝 プロンプト: \"What is the capital of France?\"");
    println!("   トークン数: {}", chat_template.len());

    println!("\n🔄 推論中...");
    let logits = model.forward(&chat_template)?;
    let logits_data = logits.as_slice();

    // Top 10を表示
    let mut indexed: Vec<(usize, f32)> = logits_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10予測:");
    for (i, (token, logit)) in indexed.iter().take(10).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    // 期待されるトークンをチェック
    // "Paris"は以下のいずれか:
    // - 3681: "Paris"
    // - 9626: " Paris"
    println!("\n📊 特定トークンのlogit:");
    println!("   Token 3681 (\"Paris\"): {:.6}", logits_data[3681]);
    println!("   Token 9626 (\" Paris\"): {:.6}", logits_data[9626]);
    println!("   Token 450 (\" The\"): {:.6}", logits_data[450]);
    println!("   Token 20780: {:.6}", logits_data[20780]);

    // llama.cppと比較
    println!("\n═══════════════════════════════════════");
    println!("🔍 llama.cppとの比較");
    println!("═══════════════════════════════════════\n");

    println!("予想される動作:");
    println!("  - llama.cppは同じプロンプトで \"Paris\" または \"The capital\" を生成");
    println!("  - RusTorchも同様のトークンを予測すべき");
    println!();

    if indexed[0].0 == 3681 || indexed[0].0 == 9626 {
        println!("✅ SUCCESS! \"Paris\"を正しく予測！");
    } else if indexed[0].0 == 450 {
        println!("✅ GOOD! \"The\"を予測（妥当な回答の始まり）");
    } else {
        println!("🤔 Token {}を予測", indexed[0].0);
        println!("   これが正しいかllama.cppで確認する必要あり");
    }

    // 簡単な生成もテスト
    println!("\n═══════════════════════════════════════");
    println!("🔄 3トークン生成テスト");
    println!("═══════════════════════════════════════\n");

    let mut current_tokens = chat_template.clone();
    println!("生成トークン:");

    for step in 0..3 {
        let logits_step = if step == 0 {
            logits.clone()
        } else {
            model.forward(&current_tokens[current_tokens.len()-1..])?
        };

        let logits_step_data = logits_step.as_slice();
        let mut indexed_step: Vec<(usize, f32)> = logits_step_data.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed_step.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let next_token = indexed_step[0].0;
        println!("  Step {}: Token {} (logit: {:.3})", step + 1, next_token, indexed_step[0].1);
        current_tokens.push(next_token);
    }

    println!("\n💡 ヒント:");
    println!("  同じプロンプトをllama.cppで試すには:");
    println!("  echo \"What is the capital of France?\" | llama-cli -m <model> -n 3");

    Ok(())
}
