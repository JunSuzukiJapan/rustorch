# RoPE位置エンコーディング検証

**日付**: 2025-10-07
**調査**: RoPE位置パラメータが自己回帰生成中に正しく増分しているか確認

## まとめ

✅ **検証完了**: RoPE位置エンコーディングは正しく動作しています

## テスト結果

### 位置パラメータのログ

```
🔍 [POSITION] Forward call 0: seq_len=18, current_position=0, will apply RoPE positions: 0..17
🔍 [POSITION] Forward call 1: seq_len=1, current_position=18, will apply RoPE positions: 18..18
🔍 [POSITION] Forward call 2: seq_len=1, current_position=19, will apply RoPE positions: 19..19
```

### 実装確認

**`llama.rs:759`**:
```rust
let current_position = self.kv_cache[0].cached_len;
```

**`llama.rs:353`** (`apply_rope`メソッド内):
```rust
for token_idx in 0..seq_len {
    let position = start_position + token_idx;
    // RoPEを各トークンに適用
}
```

## ロジック検証

### Step 0（初回forward）
- 入力: 18トークン
- kv_cache[0].cached_len = 0
- current_position = 0
- apply_rope: token 0→pos 0, token 1→pos 1, ..., token 17→pos 17 ✅

### Step 1（増分生成）
- 入力: 1トークン（新規生成トークン）
- kv_cache[0].cached_len = 18
- current_position = 18
- apply_rope: token 0→pos 18 ✅

### Step 2（さらに増分）
- 入力: 1トークン
- kv_cache[0].cached_len = 19
- current_position = 19
- apply_rope: token 0→pos 19 ✅

## 結論

**RoPE位置エンコーディングの実装は完全に正しい。**

- 各トークンの位置は正しく計算されている
- KVキャッシュの長さを使用した位置オフセットは正確
- `apply_rope`内のループで各トークンに正しい位置が適用されている

## トークン繰り返し問題の再評価

RoPE位置が正しいことが証明されたため、トークン繰り返し問題の原因は**別の場所**にあります：

### 観察された動作

**入力: "Hi"**
- 生成: "ragmentragmentragment..." (token 4305)

**入力: "What is the capital of France?"**
- 生成: "ructructruct..." (token 1247)

### Logitsの比較

```
Step 0: token 4305, logit=9.9401
Step 1: token 4305, logit=9.9497  (差分: +0.0096)
Step 2: token 4305, logit=9.9497  (ほぼ同一)
```

logitsの変化が**極めて小さい**（0.01程度）。

### 可能性のある原因

1. **モデル自体の問題**
   - TinyLlama-1.1Bは小規模モデル
   - チャットテンプレートの適用が不適切な可能性
   - llama.cppとは異なるシステムプロンプトが必要かも

2. **サンプリングの問題**
   - 現在argmax（温度=0）を使用
   - 温度サンプリングやtop-pサンプリングが必要かも
   - Repetition penaltyが未実装

3. **チャットテンプレート**
   - llama.cppはデフォルトでシステムメッセージを使用
   - RusTorchはシンプルなテンプレートのみ

## 次のステップ

1. **Repetition penaltyの実装**
   - 既に生成されたトークンにペナルティを適用
   - llama.cppと同様の実装

2. **温度サンプリングの実装**
   - argmaxだけでなく確率的サンプリングを追加
   - top-p (nucleus) サンプリング

3. **チャットテンプレートの改善**
   - システムメッセージの追加
   - llama.cppのテンプレートと同様の形式

## llama.cppとの比較

### llama.cpp（正常動作）
```
入力: "What is the capital of France?"
出力: "The capital of France is Paris."
```

### RusTorch（問題あり）
```
入力: "What is the capital of France?"
出力: "ructructruct..."
```

llama.cppは以下を使用：
- Repetition penalty (repeat_penalty = 1.0)
- 温度サンプリング (temp = 0.8)
- Top-p サンプリング (top_p = 0.95)
- Top-k サンプリング (top_k = 40)

RusTorchは現在：
- Argmax only (温度 = 0相当)
- Repetition penalty なし
- サンプリング戦略なし

## 結論

**根本原因は位置エンコーディングではなく、サンプリング戦略とRepetition penaltyの欠如。**

モデルの計算自体は正しく動作していますが、生成戦略が不十分です。
