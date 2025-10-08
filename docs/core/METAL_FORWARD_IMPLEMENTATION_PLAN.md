# Metal Forward Pass Implementation Plan

## 現状分析

### 利用可能なMetal演算（src/gpu/metal_kernels.rs）
- ✅ `matmul_f32` - 行列乗算
- ✅ `elementwise_add_f32` - 要素ごと加算
- ✅ `layer_norm_f32` - Layer normalization
- ✅ `gelu_f32` - GELU活性化関数
- ✅ `sigmoid_f32`, `tanh_f32`, `relu_f32` - 活性化関数

### GPT Forward Passに必要な演算

1. **Embedding Lookup**: input_ids → embeddings
2. **Positional Encoding**: embeddings + positional
3. **Transformer Blocks** (繰り返し):
   - Multi-head Attention
   - Layer Normalization
   - Feed-forward Network
   - Residual connections

4. **Output Projection**: hidden → logits

## 実装戦略

### Phase 2A: テンソル変換ユーティリティ

現在の問題：
- GPTModel: `Tensor<f64>` (CPU)
- MetalKernelExecutor: `&[f32]` (GPU)

必要な変換：
```rust
// CPU → GPU
fn tensor_f64_to_f32_vec(tensor: &Tensor<f64>) -> Vec<f32>

// GPU → CPU
fn f32_vec_to_tensor_f64(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f64>
```

### Phase 2B: 簡易Metal Forward実装

**アプローチ**: 段階的実装
1. まず、Embeddingだけをf32変換してMetalで処理
2. 次に、単純な行列演算をMetal化
3. 最後に、全Transformer blockをMetal化

**Phase 2B.1: Embedding + Matmul**
```rust
fn forward_metal(&self, input_ids: &[usize]) -> Result<Tensor<f64>> {
    // 1. Embedding lookup (CPU → f32)
    let embeddings_f32 = self.lookup_embeddings_f32(input_ids)?;

    // 2. Simple matmul test with Metal
    let executor = MetalKernelExecutor::get()?;
    let output_f32 = executor.lock().unwrap()
        .as_ref().unwrap()
        .matmul_f32(&embeddings_f32, &weights_f32, ...)?;

    // 3. Convert back to f64
    Ok(f32_vec_to_tensor_f64(output_f32, shape))
}
```

**Phase 2B.2: Full Transformer Block**
```rust
// Metal上でTransformer全体を実行
// - Attention: Q,K,V matmul + softmax + output matmul
// - FFN: fc1 matmul + gelu + fc2 matmul
// - LayerNorm: Metal kernel使用
```

## 技術的課題

### 1. 精度の違い (f64 vs f32)

**問題**:
- GPTModel設計がf64ベース
- Metal kernelがf32ベース

**解決策**:
- 短期: f64 ↔ f32 変換を許容（多少の精度低下）
- 長期: GPTModelをジェネリック化、またはf32版を別途作成

### 2. メモリ転送オーバーヘッド

**問題**:
- CPU ↔ GPU転送が遅い
- 毎回転送すると遅くなる

**解決策**:
- Phase 2B: 毎回転送（シンプル）
- Phase 3: 重みをGPUに常駐させる（最適化）

### 3. GGUFの量子化テンソル

**問題**:
- GGUFの重みはQ4_K, Q6_Kなど量子化済み
- Metal kernelはf32を期待

**解決策**:
- Phase 2B: CPU上でdequantize → f32変換 → Metal転送
- Phase 3: Metal側でdequantizeカーネル実装

## 実装優先度

### 必須（Phase 2B.1）- 今すぐ実装
1. ✅ テンソル変換ユーティリティ
   - `tensor_f64_to_f32_vec()`
   - `f32_vec_to_tensor_f64()`

2. ✅ 簡易Metal matmul テスト
   - Embedding lookup → f32
   - Metal matmul実行
   - 結果をf64に戻す

3. ✅ 動作確認
   - 出力が意味を持つか確認
   - CPU版と結果比較（精度チェック）

### 推奨（Phase 2B.2）- 次のステップ
4. Transformer block全体のMetal化
5. Layer normalizationの統合
6. GELU活性化関数の統合

### 最適化（Phase 3）- 将来
7. GPU-resident 重みキャッシュ
8. Metal側でのdequantization
9. Batch処理サポート

## 次のアクション

### 即座に実装
```rust
// src/models/gpt.rs に追加

#[cfg(feature = "metal")]
impl GPTModel {
    /// Convert Tensor<f64> to Vec<f32>
    fn tensor_to_f32_vec(tensor: &Tensor<f64>) -> Vec<f32> {
        tensor.data.iter().map(|&x| x as f32).collect()
    }

    /// Convert Vec<f32> to Tensor<f64>
    fn f32_vec_to_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f64> {
        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        Tensor::from_vec(data_f64, shape)
    }

    /// Metal-accelerated embedding lookup
    fn lookup_embeddings_metal(&self, input_ids: &[usize]) -> Result<Vec<f32>> {
        // 1. Get embedding weight
        let emb_weight = self.weights.get("token_embd.weight")
            .ok_or(...)?;

        // 2. Lookup embeddings for input_ids
        // 3. Convert to f32
        // 4. Return as flat Vec<f32>
    }
}
```

### テスト方法
```bash
# Metal版でテスト
./rustorch-cli --backend metal --model model.gguf --max-tokens 3

# 期待される出力
🚀 GPT forward pass using Metal GPU acceleration
🔧 Converting embeddings to f32...
🚀 Executing Metal matmul...
✅ Metal forward pass complete
```

## 実装完了の判断基準

**Phase 2B.1 完了**:
- [ ] テンソル変換が動作
- [ ] Metal matmul が動作
- [ ] 出力テンソルが意味を持つ
- [ ] CPU版と出力が近い（誤差<1%）

**Phase 2B.2 完了**:
- [ ] Transformer block全体がMetal化
- [ ] 生成テキストが意味を持つ
- [ ] CPU版と同等の品質

## 参考資料

- MetalKernelExecutor: `src/gpu/metal_kernels.rs`
- F32GPTModel実装: `src/hybrid_f32/models/gpt.rs`
- Metal Performance Shaders: https://developer.apple.com/metal/
