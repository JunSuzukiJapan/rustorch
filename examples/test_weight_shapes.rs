use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🔍 Verifying weight shapes and layout\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;
    println!("✅ Model loaded\n");

    // Check critical weight shapes
    let weights_to_check = vec![
        ("token_embd.weight", "Embedding matrix"),
        ("output.weight", "LM head projection"),
        ("output_norm.weight", "Final RMSNorm"),
        ("blk.0.attn_q.weight", "Layer 0 Query projection"),
        ("blk.0.attn_k.weight", "Layer 0 Key projection"),
        ("blk.0.attn_v.weight", "Layer 0 Value projection"),
        ("blk.0.attn_output.weight", "Layer 0 Attention output"),
        ("blk.0.ffn_gate.weight", "Layer 0 FFN gate"),
        ("blk.0.ffn_up.weight", "Layer 0 FFN up"),
        ("blk.0.ffn_down.weight", "Layer 0 FFN down"),
        ("blk.0.attn_norm.weight", "Layer 0 Attention norm"),
        ("blk.0.ffn_norm.weight", "Layer 0 FFN norm"),
    ];

    println!("Weight shapes:");
    println!("{:<30} {:<40} {:<20}", "Name", "Description", "Shape");
    println!("{:-<90}", "");

    for (name, desc) in weights_to_check {
        if let Some(weight) = model.get_weight(name) {
            println!("{:<30} {:<40} {:?}", name, desc, weight.shape());
        } else {
            println!("{:<30} {:<40} NOT FOUND", name, desc);
        }
    }

    println!("\n📊 Expected shapes (Llama-2 TinyLlama 1.1B):");
    println!("  Hidden size: 2048");
    println!("  Num heads: 32");
    println!("  Num KV heads: 4 (GQA)");
    println!("  Head dim: 64 (2048 / 32)");
    println!("  FFN intermediate: 5632");
    println!("  Vocab size: 32000");

    println!("\n✅ Expected matrix shapes:");
    println!("  token_embd.weight:     [2048, 32000]");
    println!("  output.weight:         [2048, 32000]");
    println!("  output_norm.weight:    [2048]");
    println!("  attn_q.weight:         [2048, 2048]  (hidden → num_heads * head_dim)");
    println!("  attn_k.weight:         [2048, 256]   (hidden → num_kv_heads * head_dim)");
    println!("  attn_v.weight:         [2048, 256]   (hidden → num_kv_heads * head_dim)");
    println!("  attn_output.weight:    [2048, 2048]");
    println!("  ffn_gate.weight:       [2048, 5632]");
    println!("  ffn_up.weight:         [2048, 5632]");
    println!("  ffn_down.weight:       [5632, 2048]");
    println!("  attn_norm.weight:      [2048]");
    println!("  ffn_norm.weight:       [2048]");

    // Verify matmul dimensions
    println!("\n🔢 Matmul dimension verification:");

    let q_weight = model.get_weight("blk.0.attn_q.weight").expect("Q weight not found");
    let k_weight = model.get_weight("blk.0.attn_k.weight").expect("K weight not found");

    println!("  Input [seq, 2048] @ Q_weight {:?} → [seq, {}]",
        q_weight.shape(), q_weight.shape()[1]);
    println!("  Input [seq, 2048] @ K_weight {:?} → [seq, {}]",
        k_weight.shape(), k_weight.shape()[1]);

    Ok(())
}
