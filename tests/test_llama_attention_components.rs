// Llama Attention Layer Component Tests
// 各コンポーネントを個別にテストして問題を特定

use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::tensor::F32Tensor;
use std::path::Path;

#[test]
#[ignore] // 手動実行用
fn test_attention_components_step_by_step() {
    println!("\n=== Llama Attention Components Test ===\n");

    // 1. モデルロード
    let model_path = Path::new("/Users/junsuzuki/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    println!("📂 Loading model from: {}", model_path.display());

    let model = F32LlamaModel::from_gguf_with_device(model_path, DeviceType::Cpu)
        .expect("Failed to load model");

    println!("✅ Model loaded successfully");
    println!("   Config: vocab={}, hidden={}, layers={}, heads={}",
        model.config().vocab_size,
        model.config().hidden_size,
        model.config().num_layers,
        model.config().num_heads
    );

    // 2. テスト用の入力を準備（Token 1: BOS）
    println!("\n📝 Preparing test input (Token 1: BOS)");
    let test_embedding = model.get_embedding(1).expect("Failed to get embedding");

    // 最初の10個の値を表示
    let first_10: Vec<f32> = test_embedding.iter().take(10).copied().collect();
    let non_zero = test_embedding.iter().filter(|&&x| x.abs() > 1e-8).count();
    println!("   Embedding[0..10]: {:?}", first_10);
    println!("   Non-zero count: {}/{}", non_zero, test_embedding.len());

    // 3. 入力テンソル作成
    let hidden_size = model.config().hidden_size;
    let x = F32Tensor::from_vec(test_embedding.clone(), &[1, hidden_size])
        .expect("Failed to create input tensor");
    println!("   Input tensor shape: {:?}", x.shape());

    // 4. Layer 0のAttention重みを取得
    let layer_idx = 0;
    println!("\n🔍 Testing Layer {} Attention Components", layer_idx);

    let q_weight = model.get_weight(&format!("blk.{}.attn_q.weight", layer_idx))
        .expect("Q weight not found");
    let k_weight = model.get_weight(&format!("blk.{}.attn_k.weight", layer_idx))
        .expect("K weight not found");
    let v_weight = model.get_weight(&format!("blk.{}.attn_v.weight", layer_idx))
        .expect("V weight not found");
    let attn_norm_weight = model.get_weight(&format!("blk.{}.attn_norm.weight", layer_idx))
        .expect("Attention norm weight not found");

    println!("   Q weight shape: {:?}", q_weight.shape());
    println!("   K weight shape: {:?}", k_weight.shape());
    println!("   V weight shape: {:?}", v_weight.shape());
    println!("   Norm weight shape: {:?}", attn_norm_weight.shape());

    // 5. RMSNorm テスト
    println!("\n🧪 Test 1: RMSNorm");
    let normed = test_rms_norm(&x, attn_norm_weight);
    println!("   ✅ RMSNorm completed");
    println!("   Output[0..10]: {:?}", &normed.as_slice()[..10]);

    // 6. Linear Projection テスト (Q, K, V)
    println!("\n🧪 Test 2: Q/K/V Linear Projections");
    let q = normed.matmul(q_weight).expect("Q projection failed");
    let k = normed.matmul(k_weight).expect("K projection failed");
    let v = normed.matmul(v_weight).expect("V projection failed");

    println!("   ✅ Q projection: shape={:?}", q.shape());
    println!("      Q[0..10]: {:?}", &q.as_slice()[..10]);
    println!("   ✅ K projection: shape={:?}", k.shape());
    println!("      K[0..10]: {:?}", &k.as_slice()[..10]);
    println!("   ✅ V projection: shape={:?}", v.shape());
    println!("      V[0..10]: {:?}", &v.as_slice()[..10]);

    // 7. RoPE テスト
    println!("\n🧪 Test 3: RoPE (Rotary Position Embedding)");
    let (rope_cos, rope_sin) = precompute_rope(model.config());
    let q_rope = apply_rope_test(&q, &rope_cos, &rope_sin, 0, model.config().head_dim());
    let k_rope = apply_rope_test(&k, &rope_cos, &rope_sin, 0, model.config().head_dim() / (model.config().num_heads / model.config().num_kv_heads));

    println!("   ✅ RoPE applied to Q");
    println!("      Q_rope[0..10]: {:?}", &q_rope.as_slice()[..10]);
    println!("   ✅ RoPE applied to K");
    println!("      K_rope[0..10]: {:?}", &k_rope.as_slice()[..10]);

    // 8. Attention Score 計算テスト
    println!("\n🧪 Test 4: Attention Scores");
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = model.config().head_dim();
    let kv_head_dim = 256 / num_kv_heads; // K/V head dimension

    println!("   num_heads={}, num_kv_heads={}, head_dim={}, kv_head_dim={}",
        num_heads, num_kv_heads, head_dim, kv_head_dim);

    // Q, K を multi-head 形式に reshape
    let q_heads = reshape_for_multihead(&q_rope, 1, num_heads, head_dim);
    let k_heads = reshape_for_multihead(&k_rope, 1, num_kv_heads, kv_head_dim);

    println!("   Q heads shape: {:?}", q_heads.shape());
    println!("   K heads shape: {:?}", k_heads.shape());

    // 簡略化: 最初のheadのみテスト
    let q_head0 = extract_head(&q_heads, 0, head_dim);
    let k_head0 = extract_head(&k_heads, 0, kv_head_dim);

    println!("   Q head[0] shape: {:?}", q_head0.shape());
    println!("   K head[0] shape: {:?}", k_head0.shape());

    // Attention score = Q @ K^T / sqrt(head_dim)
    let k_head0_t = k_head0.transpose().expect("K transpose failed");
    let attn_scores = q_head0.matmul(&k_head0_t).expect("Attention score calculation failed");
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scaled_scores = scale_tensor(&attn_scores, scale);

    println!("   ✅ Attention scores calculated");
    println!("      Scores shape: {:?}", scaled_scores.shape());
    println!("      Scores[0..5]: {:?}", &scaled_scores.as_slice()[..5.min(scaled_scores.as_slice().len())]);

    // 9. Softmax テスト
    println!("\n🧪 Test 5: Softmax");
    let attn_weights = softmax_test(&scaled_scores);
    println!("   ✅ Softmax applied");
    println!("      Weights[0..5]: {:?}", &attn_weights.as_slice()[..5.min(attn_weights.as_slice().len())]);
    println!("      Sum of weights: {}", attn_weights.as_slice().iter().sum::<f32>());

    // 10. Attention Output 計算
    println!("\n🧪 Test 6: Attention Output");
    let v_heads = reshape_for_multihead(&v, 1, num_kv_heads, kv_head_dim);
    let v_head0 = extract_head(&v_heads, 0, kv_head_dim);
    let attn_out = attn_weights.matmul(&v_head0).expect("Attention output failed");

    println!("   ✅ Attention output calculated");
    println!("      Output shape: {:?}", attn_out.shape());
    println!("      Output[0..10]: {:?}", &attn_out.as_slice()[..10.min(attn_out.as_slice().len())]);

    println!("\n✅ All component tests completed!");
}

// ヘルパー関数

fn test_rms_norm(x: &F32Tensor, weight: &F32Tensor) -> F32Tensor {
    let eps = 1e-6;
    let x_slice = x.as_slice();
    let weight_slice = weight.as_slice();

    // Calculate RMS
    let sum_sq: f32 = x_slice.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / x_slice.len() as f32 + eps).sqrt();

    // Normalize and scale
    let normalized: Vec<f32> = x_slice.iter()
        .zip(weight_slice.iter())
        .map(|(&x_val, &w_val)| (x_val / rms) * w_val)
        .collect();

    F32Tensor::from_vec(normalized, x.shape()).expect("RMSNorm failed")
}

fn precompute_rope(config: &rustorch::hybrid_f32::models::LlamaConfig) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.head_dim();
    let max_seq_len = config.max_seq_len;
    let theta = config.rope_theta;

    let mut cos_values = Vec::new();
    let mut sin_values = Vec::new();

    for pos in 0..max_seq_len {
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
            let angle = (pos as f32) * freq;
            cos_values.push(angle.cos());
            sin_values.push(angle.sin());
        }
    }

    (cos_values, sin_values)
}

fn apply_rope_test(x: &F32Tensor, cos: &[f32], sin: &[f32], position: usize, head_dim: usize) -> F32Tensor {
    let x_slice = x.as_slice();
    let mut output = Vec::with_capacity(x_slice.len());

    let num_pairs = head_dim / 2;
    let num_heads = x_slice.len() / head_dim;

    for head_idx in 0..num_heads {
        let head_offset = head_idx * head_dim;
        let head_data = &x_slice[head_offset..head_offset + head_dim];

        for i in 0..num_pairs {
            let rope_idx = position * num_pairs + i;
            let c = cos[rope_idx];
            let s = sin[rope_idx];

            let x0 = head_data[2 * i];
            let x1 = head_data[2 * i + 1];

            output.push(x0 * c - x1 * s);
            output.push(x0 * s + x1 * c);
        }
    }

    F32Tensor::from_vec(output, x.shape()).expect("RoPE failed")
}

fn reshape_for_multihead(x: &F32Tensor, seq_len: usize, num_heads: usize, head_dim: usize) -> F32Tensor {
    // [seq_len, num_heads * head_dim] -> [seq_len, num_heads, head_dim]
    F32Tensor::from_vec(x.as_slice().to_vec(), &[seq_len, num_heads, head_dim])
        .expect("Reshape failed")
}

fn extract_head(x: &F32Tensor, head_idx: usize, head_dim: usize) -> F32Tensor {
    let shape = x.shape();
    let seq_len = shape[0];
    let x_slice = x.as_slice();

    let mut head_data = Vec::new();
    for t in 0..seq_len {
        let offset = t * shape[1] * head_dim + head_idx * head_dim;
        head_data.extend_from_slice(&x_slice[offset..offset + head_dim]);
    }

    F32Tensor::from_vec(head_data, &[seq_len, head_dim])
        .expect("Extract head failed")
}

fn scale_tensor(x: &F32Tensor, scale: f32) -> F32Tensor {
    let scaled: Vec<f32> = x.as_slice().iter().map(|&v| v * scale).collect();
    F32Tensor::from_vec(scaled, x.shape()).expect("Scale failed")
}

fn softmax_test(x: &F32Tensor) -> F32Tensor {
    let x_slice = x.as_slice();

    // Find max for numerical stability
    let max_val = x_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Exp
    let exp_vals: Vec<f32> = x_slice.iter().map(|&v| (v - max_val).exp()).collect();

    // Sum
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    let softmax: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();

    F32Tensor::from_vec(softmax, x.shape()).expect("Softmax failed")
}
