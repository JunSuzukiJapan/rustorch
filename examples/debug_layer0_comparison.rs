//! Layer 0 comparison tool - Debug RusTorch vs llama.cpp
//!
//! このツールは以下を行います：
//! 1. GGUFファイルから直接トークン1(BOS)の埋め込みを抽出
//! 2. Layer 0のRMSNorm計算を手動実装
//! 3. RoPE適用を手動実装
//! 4. Attention計算を段階的に実行
//! 5. 各ステップでRusTorchの出力と比較

use anyhow::Result;
use rustorch::formats::gguf::GGUFLoader;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .unwrap_or_else(|_| "/Users/junsuzuki".to_string())
                + "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        });

    println!("🔍 GGUF Layer 0 Comparison Tool");
    println!("================================\n");
    println!("モデルパス: {}\n", model_path);

    // Load GGUF
    let loader = GGUFLoader::from_file(&model_path)?;

    // Step 1: トークン1(BOS)の埋め込みを抽出
    println!("📊 Step 1: トークン1(BOS)の埋め込みを抽出");
    println!("------------------------------------------");

    let token_embd = loader.load_tensor("token_embd.weight")?;

    println!("token_embd.weight shape: {:?}", token_embd.shape());

    // GGUF形式: [hidden_size, vocab_size] = [2048, 32000]
    // Token 1の埋め込み: flatten()[dim * 32000 + 1] for dim in 0..2048
    let hidden_size = 2048;
    let vocab_size = 32000;
    let token_id = 1usize; // BOS token

    // Tensorをflattenしてから値を取得
    let token_embd_flat = token_embd.data.iter().copied().collect::<Vec<_>>();
    println!("token_embd.weight total elements: {}", token_embd_flat.len());

    let mut embedding = Vec::with_capacity(hidden_size);
    for dim in 0..hidden_size {
        let idx = dim * vocab_size + token_id;
        embedding.push(token_embd_flat[idx]);
    }

    println!("\n✅ トークン {} の埋め込み（最初の10値）:", token_id);
    for i in 0..10 {
        println!("  embedding[{}] = {:.10}", i, embedding[i]);
    }

    // 統計情報
    let sum: f64 = embedding.iter().sum();
    let mean = sum / embedding.len() as f64;
    let sum_sq: f64 = embedding.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / embedding.len() as f64).sqrt();
    let min = embedding.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = embedding.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let non_zero = embedding.iter().filter(|&&x| x.abs() > 1e-10).count();

    println!("\n📈 埋め込み統計:");
    println!("  RMS:      {:.10}", rms);
    println!("  Mean:     {:.10}", mean);
    println!("  Min:      {:.10}", min);
    println!("  Max:      {:.10}", max);
    println!("  Non-zero: {}/{}", non_zero, embedding.len());

    // Step 2: Layer 0のRMSNorm weightを抽出
    println!("\n📊 Step 2: Layer 0のRMSNorm weightを抽出");
    println!("------------------------------------------");

    let attn_norm = loader.load_tensor("blk.0.attn_norm.weight")?;
    let attn_norm_flat = attn_norm.data.iter().copied().collect::<Vec<_>>();

    println!("blk.0.attn_norm.weight shape: {:?}", attn_norm.shape());
    println!("blk.0.attn_norm.weight elements: {}", attn_norm_flat.len());

    println!("\n最初の10値:");
    for i in 0..10 {
        println!("  attn_norm[{}] = {:.10}", i, attn_norm_flat[i]);
    }

    // Step 3: RMSNormを手動計算
    println!("\n📊 Step 3: RMSNormを手動計算");
    println!("------------------------------------------");

    let epsilon = 1e-5;

    // RMSNorm formula: x * weight / sqrt(mean(x^2) + epsilon)
    let sum_sq: f64 = embedding.iter().map(|&x| x * x).sum();
    let mean_sq = sum_sq / embedding.len() as f64;
    let rms_norm_scale = 1.0 / (mean_sq + epsilon).sqrt();

    println!("sum_sq:         {:.10}", sum_sq);
    println!("mean_sq:        {:.10}", mean_sq);
    println!("rms_norm_scale: {:.10}", rms_norm_scale);

    let mut normalized = Vec::with_capacity(hidden_size);
    for i in 0..hidden_size {
        let normed = embedding[i] * rms_norm_scale * attn_norm_flat[i];
        normalized.push(normed);
    }

    println!("\n✅ RMSNorm適用後（最初の10値）:");
    for i in 0..10 {
        println!("  normalized[{}] = {:.10}", i, normalized[i]);
    }

    // 正規化後の統計
    let sum: f64 = normalized.iter().sum();
    let mean = sum / normalized.len() as f64;
    let sum_sq: f64 = normalized.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / normalized.len() as f64).sqrt();
    let min = normalized.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = normalized.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\n📈 正規化後の統計:");
    println!("  RMS:  {:.10}", rms);
    println!("  Mean: {:.10}", mean);
    println!("  Min:  {:.10}", min);
    println!("  Max:  {:.10}", max);

    // Step 4: Q projection weightを抽出して最初の値を計算
    println!("\n📊 Step 4: Q projection計算（最初の10次元のみ）");
    println!("------------------------------------------");

    let q_weight = loader.load_tensor("blk.0.attn_q.weight")?;
    let q_weight_flat = q_weight.data.iter().copied().collect::<Vec<_>>();

    println!("blk.0.attn_q.weight shape: {:?}", q_weight.shape());
    println!("blk.0.attn_q.weight elements: {}", q_weight_flat.len());

    // Q projection: normalized @ q_weight
    // q_weight shape: [hidden_size, hidden_size] = [2048, 2048]
    // normalized shape: [1, 2048]
    // result: [1, 2048]

    println!("\nQ projection 最初の10次元を計算:");
    for i in 0..10 {
        let mut sum = 0.0;
        for j in 0..hidden_size {
            // Row-major layout: q_weight[j, i] = data[j * 2048 + i]
            let idx = j * hidden_size + i;
            sum += normalized[j] * q_weight_flat[idx];
        }
        println!("  q[{}] = {:.10}", i, sum);
    }

    // Step 5: RoPEの周波数を計算
    println!("\n📊 Step 5: RoPE周波数の計算");
    println!("------------------------------------------");

    let rope_base = 10000.0_f64;
    let head_dim = 64; // 2048 / 32 heads
    let position = 0; // 最初のトークン

    println!("RoPE base: {}", rope_base);
    println!("Head dimension: {}", head_dim);
    println!("Position: {}", position);

    println!("\n最初の10次元のRoPE周波数:");
    for i in 0..5 {
        let dim = i * 2;
        let freq = 1.0 / rope_base.powf(dim as f64 / head_dim as f64);
        let angle = position as f64 * freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        println!("  dim {}: freq={:.10}, angle={:.10}, cos={:.10}, sin={:.10}",
                 dim, freq, angle, cos_val, sin_val);
    }

    // Step 6: output_norm.weightを確認
    println!("\n📊 Step 6: output_norm.weight（最終RMSNorm）を確認");
    println!("------------------------------------------");

    let output_norm = loader.load_tensor("output_norm.weight")?;
    let output_norm_flat = output_norm.data.iter().copied().collect::<Vec<_>>();

    println!("output_norm.weight shape: {:?}", output_norm.shape());
    println!("output_norm.weight elements: {}", output_norm_flat.len());

    println!("\n最初の10値:");
    for i in 0..10 {
        println!("  output_norm[{}] = {:.10}", i, output_norm_flat[i]);
    }

    // 統計
    let sum: f64 = output_norm_flat.iter().sum();
    let mean = sum / output_norm_flat.len() as f64;
    let sum_sq: f64 = output_norm_flat.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / output_norm_flat.len() as f64).sqrt();
    let min = output_norm_flat.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = output_norm_flat.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\noutput_norm.weight統計:");
    println!("  RMS:  {:.10}", rms);
    println!("  Mean: {:.10}", mean);
    println!("  Min:  {:.10}", min);
    println!("  Max:  {:.10}", max);

    // Step 7: output.weightを確認
    println!("\n📊 Step 7: output.weight（lm_head）を確認");
    println!("------------------------------------------");

    let output_weight = loader.load_tensor("output.weight")?;
    let output_flat = output_weight.data.iter().copied().collect::<Vec<_>>();

    println!("output.weight shape: {:?}", output_weight.shape());
    println!("output.weight elements: {}", output_flat.len());

    println!("\n最初の10値:");
    for i in 0..10 {
        println!("  output[{}] = {:.10}", i, output_flat[i]);
    }

    // Token 13487の重みを確認（RusTorchで最高ロジットのトークン）
    println!("\n⚠️  Token 13487 (diplom - RusTorchで最高ロジット) の重み:");
    println!("output.weight layout: [hidden_size, vocab_size] = [2048, 32000]");
    println!("Token 13487の重み = output_weight[:, 13487]");

    let token_13487_start = 13487;
    println!("\n最初の10次元:");
    for dim in 0..10 {
        let idx = dim * vocab_size + token_13487_start;
        println!("  dim {}: {:.10}", dim, output_flat[idx]);
    }

    // 統計
    let mut token_13487_weights = Vec::with_capacity(hidden_size);
    for dim in 0..hidden_size {
        let idx = dim * vocab_size + token_13487_start;
        token_13487_weights.push(output_flat[idx]);
    }

    let sum: f64 = token_13487_weights.iter().sum();
    let mean = sum / token_13487_weights.len() as f64;
    let sum_sq: f64 = token_13487_weights.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / token_13487_weights.len() as f64).sqrt();
    let min = token_13487_weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = token_13487_weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("\nToken 13487の重み統計:");
    println!("  RMS:  {:.10}", rms);
    println!("  Mean: {:.10}", mean);
    println!("  Min:  {:.10}", min);
    println!("  Max:  {:.10}", max);

    println!("\n✅ 比較準備完了");
    println!("=====================================");
    println!("\n次のステップ:");
    println!("1. RusTorchを実行して上記の値と比較");
    println!("2. 発散している箇所を特定");
    println!("3. 該当するコードを修正");

    Ok(())
}