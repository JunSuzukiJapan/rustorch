//! RMS Norm と Token Embedding の検証ツール
//! llama.cpp との比較用にデータをダンプします

use rustorch::formats::gguf::GGUFLoader;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        eprintln!("Example: {} tinyllama-1.1b-chat-v1.0.Q8_0.gguf", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("📂 Loading model: {}", model_path);

    let loader = GGUFLoader::from_file(model_path)?;
    let params = loader.get_model_params()?;
    
    println!("\n=== モデルパラメータ ===");
    println!("vocab_size: {}", params.vocab_size);
    println!("hidden_size: {}", params.hidden_size);
    println!("num_layers: {}", params.num_layers);
    println!("num_heads: {}", params.num_heads);
    println!("num_kv_heads: {}", params.num_kv_heads);
    println!("context_length: {}", params.context_length);

    // ✅ 1. RMS Norm の hidden_size パラメータ確認
    println!("\n=== ✅ 1. RMS Norm hidden_size 確認 ===");
    println!("d_model (hidden_size): {}", params.hidden_size);
    println!("期待値: 2048");
    if params.hidden_size == 2048 {
        println!("✅ 正しい値です");
    } else {
        println!("❌ 値が異なります！");
    }

    // RMS Norm weight の確認
    println!("\n=== RMS Norm Weight 確認 ===");
    let rms_norm_keys = vec![
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    ];

    for key in &rms_norm_keys {
        if let Some(_tensor_info) = loader.get_tensor(key) {
            let tensor = loader.load_tensor(key)?;
            let shape = tensor.data.shape();
            let data_slice = tensor.data.as_slice().unwrap();
            println!("\n🔍 {}", key);
            println!("   Shape: {:?}", shape);
            println!("   Length: {}", data_slice.len());
            println!("   期待値: [2048]");
            
            if data_slice.len() == 2048 {
                println!("   ✅ 長さが正しい");
            } else {
                println!("   ❌ 長さが異なります！");
            }
            
            // 統計情報
            let mean: f64 = data_slice.iter().sum::<f64>() / data_slice.len() as f64;
            let min = data_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = data_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let rms = (data_slice.iter().map(|&v| v * v).sum::<f64>() / data_slice.len() as f64).sqrt();
            
            println!("   統計: mean={:.6}, min={:.6}, max={:.6}, rms={:.6}", mean, min, max, rms);
            println!("   最初の10要素: {:?}", &data_slice[0..10.min(data_slice.len())]);
        } else {
            println!("❌ {} が見つかりません", key);
        }
    }

    // ✅ 2. Token Embedding 値の確認
    println!("\n\n=== ✅ 2. Token Embedding 値確認 ===");
    println!("llama.cpp との比較用");
    
    let token_emb_key = "token_embd.weight";
    if let Some(_tensor_info) = loader.get_tensor(token_emb_key) {
        let tensor = loader.load_tensor(token_emb_key)?;
        let shape = tensor.data.shape();
        let data_slice = tensor.data.as_slice().unwrap();
        
        println!("\n📊 Token Embedding テンソル情報:");
        println!("   Shape: {:?}", shape);
        println!("   Total elements: {}", data_slice.len());
        
        let hidden_size = shape[0];
        let vocab_size = shape[1];
        println!("   hidden_size (shape[0]): {}", hidden_size);
        println!("   vocab_size (shape[1]): {}", vocab_size);
        
        // テストトークン: "1" のトークンID (通常29896)
        let test_tokens = vec![
            (29896, "Token 29896 (\"1\")"),
            (1, "Token 1"),
            (2, "Token 2"),
            (0, "Token 0 (BOS)"),
        ];
        
        println!("\n🔍 テストトークンの埋め込み:");
        for (token_id, description) in test_tokens {
            if token_id >= vocab_size {
                println!("\n❌ {} - 範囲外", description);
                continue;
            }
            
            let start = token_id * hidden_size;
            let end = start + hidden_size;
            let embedding = &data_slice[start..end];
            
            // 統計計算
            let mean: f64 = embedding.iter().sum::<f64>() / embedding.len() as f64;
            let sq_sum: f64 = embedding.iter().map(|&v| v * v).sum();
            let rms = (sq_sum / embedding.len() as f64).sqrt();
            let min = embedding.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = embedding.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            println!("\n📌 {}", description);
            println!("   最初の10要素: {:?}", &embedding[0..10]);
            println!("   統計: mean={:.9}, rms={:.9}", mean, rms);
            println!("   範囲: min={:.9}, max={:.9}", min, max);
            
            // llama.cpp 比較用の完全なダンプ（最初の20要素）
            println!("   [llama.cpp比較用] 最初の20要素:");
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
    } else {
        println!("❌ Token embedding が見つかりません");
    }

    // ✅ 3. Layer 0 の重み確認（llama.cpp比較の準備）
    println!("\n\n=== ✅ 3. Layer 0 重み確認 ===");
    println!("llama.cpp との Layer 0 出力比較の準備");
    
    let layer0_keys = vec![
        ("blk.0.attn_norm.weight", "Attention RMS Norm"),
        ("blk.0.attn_q.weight", "Query projection"),
        ("blk.0.attn_k.weight", "Key projection"),
        ("blk.0.attn_v.weight", "Value projection"),
        ("blk.0.attn_output.weight", "Attention output"),
        ("blk.0.ffn_norm.weight", "FFN RMS Norm"),
        ("blk.0.ffn_gate.weight", "FFN gate"),
        ("blk.0.ffn_up.weight", "FFN up"),
        ("blk.0.ffn_down.weight", "FFN down"),
    ];
    
    for (key, description) in layer0_keys {
        if let Some(_tensor_info) = loader.get_tensor(key) {
            let tensor = loader.load_tensor(key)?;
            let shape = tensor.data.shape();
            let data_slice = tensor.data.as_slice().unwrap();
            
            println!("\n🔍 {} ({})", description, key);
            println!("   Shape: {:?}", shape);
            println!("   Elements: {}", data_slice.len());
            
            // 統計
            let mean: f64 = data_slice.iter().sum::<f64>() / data_slice.len() as f64;
            let rms = (data_slice.iter().map(|&v| v * v).sum::<f64>() / data_slice.len() as f64).sqrt();
            let min = data_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = data_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            println!("   統計: mean={:.9}, rms={:.9}", mean, rms);
            println!("   範囲: [{:.9}, {:.9}]", min, max);
            println!("   最初の5要素: {:?}", &data_slice[0..5.min(data_slice.len())]);
        } else {
            println!("❌ {} が見つかりません", key);
        }
    }

    println!("\n\n=== 🎯 検証完了 ===");
    println!("次のステップ:");
    println!("1. hidden_size が 2048 であることを確認");
    println!("2. Token 29896 の埋め込みを llama.cpp の出力と比較");
    println!("3. Layer 0 の出力を llama.cpp と 1対1 で比較");
    
    Ok(())
}
