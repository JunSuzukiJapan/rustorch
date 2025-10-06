# RusTorch GGUF Llama Debugging Plan

## 現状まとめ (2025-10-06 更新)

### ✅ 検証済み・正常動作
1. **GGUF Weight Format** - 転置不要、全てのweightsが正しく読み込まれている
2. **Matrix Multiplication** - 次元が適合、エラーなし
3. **Transformer Layers** - 中間値は正常範囲
4. **Logits Computation** - ゼロではなく、適切な値(9.5-9.6範囲)
5. **Reference Implementation** - llama.cpp は同じモデルで正しく "Paris" を出力
6. **✅ Embedding Extraction** - Row extractionへの修正で正常動作
7. **✅ RoPE Implementation** - Position 0で cos=1, sin=0を正しく適用
8. **✅ Q/K/V Projections** - 全て非ゼロの正常値を出力
9. **✅ パフォーマンス** - タイムアウトなしで5トークン生成完了

### ❌ 問題
**出力品質**: 無意味なトークン列が生成される
- Input: "What is the capital of France?"
- Our output: "ruction werk werk werk werk" (トークン [13210, 9888, 9888, 9888, 9888])
- Expected (llama.cpp): "Paris" または妥当な応答

### 🔬 調査結果

#### Position 12-14 での観測
- Logits正常: top5=[(19388, 9.60), (15965, 9.58), ...]
- Last layer出力正常: [-0.72, -0.77, -0.18, 1.27, ...]
- RMSNorm出力正常: [-1.27, -1.29, -0.32, 2.31, ...]

#### Weight Shapes確認
- `token_embd.weight`: [2048, 32000]
- `output.weight`: [2048, 32000]
- Linear layers: [2048, 256] など - 全て正常

### 🎯 未解決の疑問点

1. **Embedding Extraction**
   - Column extraction使用中: `idx = i * vocab_size + token_id`
   - 形状 [2048, 32000] でのメモリレイアウトが正しいか?
   - llama.cppと同じ値を抽出しているか未確認

2. **Performance Issue**
   - 長い入力でタイムアウト発生
   - ログ無効化しても改善せず
   - 無限ループまたはdeadlockの可能性

3. **Sampling Strategy**
   - Temperature/top-k/top-pの設定確認必要
   - 同じlogit値を持つ複数トークンの選択ロジック

## 戦略的デバッグ計画

### Phase 1: Embedding検証 (最優先)

**仮説**: Column extractionが誤った値を抽出している

**検証方法**:
1. Token ID 1 (BOS) のembeddingを手動計算
2. llama.cppと比較 (--log-disable false で内部値確認)
3. 必要なら row extraction に変更してテスト

**コード変更箇所**: `src/hybrid_f32/models/llama.rs:467-481`

### Phase 2: パフォーマンス問題解決

**仮説**: KV cacheまたはメモリ管理に問題

**検証方法**:
1. 最大トークン数を制限 (--max-tokens 5)
2. KV cache更新ロジックを確認
3. Metal GPUメモリリークチェック

### Phase 3: Sampling戦略確認

**検証方法**:
1. Temperature=0.0 (greedy sampling) でテスト
2. Top token IDを直接確認
3. llama.cppのsampling設定と比較

## 次のアクション

### 即座に実行
1. Embedding extractionを row extraction に変更してテスト
2. 最小限の入力 ("Hi") で動作確認
3. KV cacheサイズ制限追加

### 比較検証
1. llama.cpp --log-enable で内部値取得
2. 同じtoken IDでのembedding値比較
3. 最初のlogits値を比較

## コード修正案

### 修正1: Row Extraction強制
```rust
// get_embedding() 内
// 常にrow extractionを使用
let start = token_id * hidden_size;
let end = start + hidden_size;
Ok(embed_data[start..end].to_vec())
```

### 修正2: Max Tokens制限
```rust
// CLI引数に--max-tokens追加済み
// デフォルトを5に変更してテスト
```

### 修正3: デバッグ最小化
```rust
// 全てのeprintln!を条件付きに
const DEBUG: bool = false;
if DEBUG { eprintln!(...); }
```

## 成功基準

✅ "What is the capital of France?" → "Paris" を含む適切な応答
✅ タイムアウトなし (< 10秒)
✅ llama.cppと同等の品質

---
Last updated: 2025-10-06
Status: Embedding extraction検証が次のステップ
