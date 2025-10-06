/// 実際のチャットフォーマットでのテスト
///
/// 生のBOSトークンではなく、適切なチャットテンプレートを使用して
/// llama.cppとの比較可能な結果を得る

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 チャットフォーマットでのテスト\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("📂 モデル読み込み中...");
    let mut model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    // TinyLlamaのチャットフォーマット:
    // <|system|>
    // {system_message}</s>
    // <|user|>
    // {user_message}</s>
    // <|assistant|>

    // まずは簡単なテスト: 空のプロンプトから開始
    println!("\n═══════════════════════════════════════");
    println!("テスト1: 生のBOSトークンのみ");
    println!("═══════════════════════════════════════");

    let input_bos = vec![1]; // <s>
    let logits_bos = model.forward(&input_bos)?;
    let logits_bos_data = logits_bos.as_slice();

    let mut indexed_bos: Vec<(usize, f32)> = logits_bos_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_bos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 5予測:");
    for (i, (token, logit)) in indexed_bos.iter().take(5).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }
    println!("\n特定トークン:");
    println!("   Token 450 (\" The\"): {:.6}", logits_bos_data[450]);

    // テスト2: システムメッセージ付き
    // TinyLlamaのトークン化を手動で行う必要があるため、
    // まずは一般的なパターンをテスト
    println!("\n═══════════════════════════════════════");
    println!("テスト2: 複数トークンでの推論");
    println!("═══════════════════════════════════════");
    println!("入力: BOSの後に数トークン追加");

    // llama.cppで確認された正しいトークン化:
    // "<s> <|user|>" のようなシーケンス
    // 実際のトークンIDは不明なので、いくつかの一般的なトークンを試す

    // 空白2個 + "What" + " is" のトークンシーケンス
    // （これはllama.cppで確認済み: 1 -> '<s>', 259 -> '  ', 5618 -> 'What', 338 -> ' is'）
    let input_multi = vec![1, 259, 5618, 338]; // "<s>  What is"

    println!("入力トークン: {:?}", input_multi);
    let logits_multi = model.forward(&input_multi)?;
    let logits_multi_data = logits_multi.as_slice();

    let mut indexed_multi: Vec<(usize, f32)> = logits_multi_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_multi.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10予測:");
    for (i, (token, logit)) in indexed_multi.iter().take(10).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    // llama.cppでの確認:
    // "What is" の次は "the" (token 1552) が来るはず
    println!("\n特定トークン:");
    println!("   Token 1552 (\"the\"): {:.6}", logits_multi_data[1552]);
    println!("   Token 278 (\" the\"): {:.6}", logits_multi_data[278]);

    // テスト3: もっと長いシーケンス
    println!("\n═══════════════════════════════════════");
    println!("テスト3: より長いプロンプト");
    println!("═══════════════════════════════════════");

    // "The capital of France is" のようなシーケンス
    // Token IDs (推測):
    // 1: <s>
    // 450: " The"
    // 7483: " capital"
    // 310: " of"
    // 3444: " France"
    // 338: " is"
    let input_long = vec![1, 450, 7483, 310, 3444, 338];

    println!("入力トークン: {:?}", input_long);
    let logits_long = model.forward(&input_long)?;
    let logits_long_data = logits_long.as_slice();

    let mut indexed_long: Vec<(usize, f32)> = logits_long_data.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_long.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 10予測:");
    for (i, (token, logit)) in indexed_long.iter().take(10).enumerate() {
        println!("   {}. Token {}: {:.6}", i + 1, token, logit);
    }

    // "Paris" のトークンIDを探す（推測: 3681 or 9626）
    println!("\n特定トークン:");
    println!("   Token 3681 (\"Paris\"?): {:.6}", logits_long_data[3681]);
    println!("   Token 9626 (\" Paris\"?): {:.6}", logits_long_data[9626]);

    println!("\n═══════════════════════════════════════");
    println!("🔍 分析");
    println!("═══════════════════════════════════════\n");

    println!("観察:");
    println!("  1. BOSのみ: Token {}", indexed_bos[0].0);
    println!("  2. 複数トークン: Token {}", indexed_multi[0].0);
    println!("  3. 長いプロンプト: Token {}", indexed_long[0].0);
    println!();
    println!("次のステップ:");
    println!("  - llama.cppで同じトークンシーケンスをテスト");
    println!("  - 正確なトークナイザーを使用してプロンプトを変換");
    println!("  - より長いシーケンスで生成品質を評価");

    Ok(())
}
