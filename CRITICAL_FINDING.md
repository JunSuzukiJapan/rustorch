# 🚨 重大な発見：Token 20780の謎

## 観察された現象

RusTorchのLlamaモデルは、**あらゆる入力に対してToken 20780を予測する**：

| テストケース | 入力 | RusTorchの予測 | llama.cppの予測 |
|------------|------|----------------|-----------------|
| 1. BOSのみ | `[1]` | Token 20780 (9.579) | "Air Force Rec" |
| 2. "What is" | `[1, 259, 5618, 338]` | Token 20780 (9.696) | - |
| 3. "The capital of France is" | `[1, 450, 7483, 310, 3444, 338]` | Token 20780 (9.696) | "The capital" |

## Token 20780とは何か？

トークンIDが確認できないため、llama-tokenizeで調査が必要。しかし、すべての異なる入力で同じトークンが予測されるのは**明らかに異常**。

## 可能性のある原因

### 1. LM Head Weightの問題 ⚠️
```
LM headのweight: [2048, 32000]
最後のhidden state: [1, 2048]
matmul: [1, 2048] @ [2048, 32000] → [1, 32000]
```

もし**LM head weightの特定の列（Token 20780に対応）が異常に大きい値**を持っていたら、入力に関係なく常にそのトークンが最高logitになる。

### 2. LM Head Weight読み込みの問題
- `output.weight` の読み込み時にtransposeが必要？
- 現在のコード: transpose=false
- 検証が必要

### 3. Matmulの実装問題
- Matmul自体は検証済み（test_exact_hidden_state.rsで100%一致）
- しかし、最終的なLM headでの使用時に問題がある可能性

## 決定的テスト

### テスト1: Token 20780のweight列を確認
```rust
// output.weightのToken 20780に対応する列を確認
// もし異常に大きい値があれば、それが原因
let token_20780_weights = &output_weight[..2048]; // 列20780
```

### テスト2: LM Head Weightのtransposeを試す
```rust
// 現在: needs_transpose = false
// 試す: needs_transpose = true (output.weightのみ)
```

### テスト3: 手動計算でlogitを検証
```rust
// hidden_state @ output_weight を手動計算
// Token 20780と450のlogitを比較
```

## 次のアクション

1. **最優先**: Token 20780のweight列を調査
2. LM head weightのtransposeを試す
3. 手動計算でlogit値を完全検証

この発見は、問題が**特定のトークンのweight列に集中している**可能性を示唆しています。
