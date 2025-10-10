//! Layer 0 の出力をダンプして llama.cpp と比較
//! 
//! 使用方法:
//! cargo run --release --example dump_layer0_output -- <model.gguf> <input_text>

use rustorch::backends::DeviceType;
use rustorch::error::RusTorchError;
use rustorch::formats::gguf::GGUFLoader;
use rustorch::models::gpt::{GPTConfig, GPTModel};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <input_text>", args[0]);
        eprintln!("Example: {} tinyllama-1.1b-chat-v1.0.Q8_0.gguf \"1\"", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let input_text = &args[2];
    
    println!("📂 モデル読み込み: {}", model_path);
    println!("📝 入力テキスト: \"{}\"", input_text);

    // モデル読み込み
    let loader = GGUFLoader::from_file(model_path)?;
    let params = loader.get_model_params()?;
    let config = GPTConfig::from_model_params(&params);
    
    println!("\n=== モデル設定 ===");
    println!("hidden_size (d_model): {}", config.d_model);
    println!("num_layers: {}", config.num_layers);
    println!("num_heads: {}", config.num_heads);
    println!("num_kv_heads: {}", config.num_kv_heads);
    
    // トークナイザー（簡易版 - 実際のトークナイザーに置き換える必要があります）
    // ここでは "1" -> [29896] のような簡単なマッピングを想定
    let token_ids = if input_text == "1" {
        vec![29896]
    } else {
        eprintln!("⚠️  簡易トークナイザー: \"1\" のみサポート");
        vec![29896]
    };
    
    println!("\n=== トークン化 ===");
    println!("Token IDs: {:?}", token_ids);
    
    // GPTモデル作成
    let mut model = GPTModel::from_gguf(model_path, DeviceType::Cpu)?;
    
    println!("\n=== 🔍 Token Embedding 出力 ===");
    
    // Token embedding を手動で取得
    let token_emb_key = "token_embd.weight";
    let token_emb_tensor = loader.load_tensor(token_emb_key)?;
    let emb_shape = token_emb_tensor.data.shape();
    let hidden_size = emb_shape[0];
    let vocab_size = emb_shape[1];
    let emb_data = token_emb_tensor.data.as_slice().unwrap();
    
    println!("Token embedding shape: {:?}", emb_shape);
    println!("hidden_size: {}, vocab_size: {}", hidden_size, vocab_size);
    
    for (idx, &token_id) in token_ids.iter().enumerate() {
        let start = token_id * hidden_size;
        let end = start + hidden_size;
        let embedding = &emb_data[start..end];
        
        let mean: f64 = embedding.iter().sum::<f64>() / embedding.len() as f64;
        let rms = (embedding.iter().map(|&v| v * v).sum::<f64>() / embedding.len() as f64).sqrt();
        
        println!("\n📌 Position {} (Token ID: {})", idx, token_id);
        println!("   統計: mean={:.9}, rms={:.9}", mean, rms);
        println!("   最初の20要素:");
        for i in 0..20.min(embedding.len()) {
            if i % 5 == 0 {
                print!("      ");
            }
            print!("{:12.9}", embedding[i]);
            if (i + 1) % 5 == 0 {
                println!();
            } else {
                print!(" ");
            }
        }
    }
    
    println!("\n=== 🔍 Layer 0 RMS Norm Weight ===");
    let ln1_key = "blk.0.attn_norm.weight";
    let ln1_tensor = loader.load_tensor(ln1_key)?;
    let ln1_data = ln1_tensor.data.as_slice().unwrap();
    
    println!("RMS Norm weight shape: {:?}", ln1_tensor.data.shape());
    println!("Length: {}", ln1_data.len());
    println!("期待値: 2048");
    
    if ln1_data.len() == 2048 {
        println!("✅ 長さが正しい");
    } else {
        println!("❌ 長さが {} です！", ln1_data.len());
    }
    
    let mean: f64 = ln1_data.iter().sum::<f64>() / ln1_data.len() as f64;
    let rms = (ln1_data.iter().map(|&v| v * v).sum::<f64>() / ln1_data.len() as f64).sqrt();
    println!("統計: mean={:.9}, rms={:.9}", mean, rms);
    println!("最初の10要素: {:?}", &ln1_data[0..10]);
    
    println!("\n=== 🎯 Layer 0 出力計算の準備完了 ===");
    println!("\nllama.cpp との比較手順:");
    println!("1. llama.cpp で同じ入力 \"1\" を処理");
    println!("2. Layer 0 の出力（Attention + FFN 後）をダンプ");
    println!("3. RusTorch の Layer 0 出力と要素ごとに比較");
    println!("4. 差異がある場合、どこで発生しているか特定:");
    println!("   - Token Embedding");
    println!("   - RMS Norm (Attention前)");
    println!("   - Attention 計算");
    println!("   - RMS Norm (FFN前)");
    println!("   - FFN 計算");
    
    println!("\n💡 デバッグのヒント:");
    println!("- RUSTORCH_DEBUG=1 を設定して詳細ログを有効化");
    println!("- RMS Norm の hidden_size パラメータが 2048 であることを確認");
    println!("- 各ステップの RMS 値を llama.cpp と比較");
    
    Ok(())
}
