# RoPE Position Verification Results

**日時**: 2025-10-08
**ステータス**: Position計算は正しい、RoPE実装の検証が必要

## 🎯 検証結果

### Position計算の検証 ✅

**Step 0 (プロンプト処理)**:
```
input_ids.len = 20 (prompt tokens)
start_position = 0
RoPE positions applied: 0, 1, 2, ..., 19
```

**確認項目**:
- ✅ `start_position` が正しく0に設定されている
- ✅ 各token_idxに対して正しいpositionが計算されている (position = start_position + token_idx)
- ✅ 22層すべてで同じpositionが使用されている（各層でQとKに適用、計44回）
- ✅ Debug logsが正しく出力されている

### RoPE関数呼び出しパターン

**Step 0での呼び出し**:
```
Layer 0: apply_rope(Q, start_position=0), apply_rope(K, start_position=0)
Layer 1: apply_rope(Q, start_position=0), apply_rope(K, start_position=0)
...
Layer 21: apply_rope(Q, start_position=0), apply_rope(K, start_position=0)

Total: 22 layers × 2 calls = 44 calls
```

各呼び出しで:
```
🔴 [ROPE ENTRY] start_position=0
🔴 [ROPE LOOP] token_idx=0, position=0
🔴 [ROPE LOOP] token_idx=1, position=1
...
🔴 [ROPE LOOP] token_idx=19, position=19
```

## 🔍 現在の状況

### 問題
- Position計算は正しいにもかかわらず、Token 15010 ("drew")が生成される
- llama.cppは同じモデルで正しいトークンを生成する

### 次の調査ポイント

1. **RoPE計算の実装検証**
   - `apply_rope()` 内のrotation計算が正しいか
   - cos/sin lookup tableが正しく初期化されているか
   - llama.cppのRoPE実装と数値的に一致するか

2. **Rotation matrix計算**
   ```rust
   // 現在の実装
   let rope_idx = position * (head_dim / 2) + i;
   let cos = self.rope_cos[rope_idx];
   let sin = self.rope_sin[rope_idx];

   let x0 = head_data[2 * i];
   let x1 = head_data[2 * i + 1];

   output.push(x0 * cos - x1 * sin);  // Real part
   output.push(x0 * sin + x1 * cos);  // Imaginary part
   ```

3. **Cos/Sin lookup table初期化**
   - `rope_cos`と`rope_sin`の値がllama.cppと一致するか検証
   - Theta計算 (`theta = 10000.0^(-2i/d)`) が正しいか確認

## 📊 Debug Log Analysis

### Step 0の詳細

**Input**:
- Prompt: "test" → 20 tokens (including chat template)
- Start position: 0
- Sequence length: 20

**RoPE Application**:
- All 22 layers: Position 0-19 correctly applied
- Each layer processes Q and K with RoPE

**Final Output**:
```
Top logit: token 15010 ("drew") = 11.9734
Expected tokens have much lower logits:
  - token 15043: -0.29
  - token 6324: -2.30
```

## 🎯 次のステップ

1. **RoPE cos/sin値の検証**
   - Position 0, 1, 2でのcos/sin値をログ出力
   - llama.cppの対応する値と比較

2. **Rotation結果の検証**
   - Layer 0, position 0での rotation前後のQ/K値を出力
   - llama.cppの対応する値と比較

3. **RoPE実装の数値検証**
   - 単一positionでの計算をllama.cppと完全に一致させる
   - 微小な差異が累積していないか確認

## 💡 仮説

**最も可能性が高い原因**: RoPE rotation計算の実装誤り

- Position計算は正しいが、rotation matrixの適用方法が間違っている
- Cos/sin lookup tableの初期化に問題がある
- Head dimensionの処理（64次元のpair処理）に誤りがある

**検証方法**: llama.cppのRoPE実装と1行ずつ比較し、数値的に完全一致を確認
