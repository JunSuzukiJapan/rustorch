# KVキャッシュ無効化テスト結果

**日時**: 2025-10-08
**ステータス**: ⚠️ **KVキャッシュは問題の原因ではない**

## 🎯 テスト目的

KVキャッシュの実装バグが原因でトークン繰り返し ("drew drew drew") が発生しているか検証する。

## 📊 テスト結果

### KVキャッシュ有効時 (Before)

```
Prompt: "Hello"
Output: "drew drew drew Superhé"
Tokens: [15010, 15010, 15010, 5670, 19880]
```

### KVキャッシュ無効化後 (After)

```
Prompt: "Hello"
Output: "drew drew drew Superhé"  ← 変化なし！
Tokens: [15010, 15010, 15010, 5670, 19880]
```

## 🔍 重要な発見

### 1. KVキャッシュは問題の原因ではない

**完全に同じ出力** → KVキャッシュを無効化しても問題は解決しない

### 2. Logit分布自体が間違っている

#### RusTorch (KVキャッシュ無効)
```
Top 10 logits:
  #1: token=15010 "drew"    logit=11.8283
  #2: token=5670  "Super"   logit=9.4061
  #3: token=19880 "hé"      logit=9.0219
```

#### llama.cpp (期待される動作)
```
Output: "Hello world\n\nAnd then"
```

→ **RusTorchのLogit計算自体が根本的に間違っている**

### 3. トークン繰り返しはLogit分布の崩壊

- 最初のトークン: 15010 ("drew")
- 2番目のトークン: 15010 ("drew") ← 同じ
- 3番目のトークン: 15010 ("drew") ← 同じ

**原因**: 各ステップで同じLogit分布を生成している可能性

## 🧠 新たな根本原因の仮説

### 仮説1: 位置エンコーディング (position) の計算ミス ⭐⭐⭐

**可能性**: 最高

**症状**:
- トークン繰り返し → 全てのトークンが同じ位置として処理されている
- KVキャッシュ無効でも同じ → キャッシュではなく位置の問題

**検証方法**:
```rust
// attention_layer関数で position パラメータを確認
eprintln!("🐛 [POSITION] layer={}, position={}", layer_idx, position);
```

**期待される動作**:
```
Step 0: position = <入力トークン数>
Step 1: position = <入力トークン数 + 1>
Step 2: position = <入力トークン数 + 2>
```

### 仮説2: 入力トークン数の誤計算 ⭐⭐

**可能性**: 高

**症状**:
- 生成ループで`position`が更新されていない
- 常に同じ位置でRoPEが適用される

**検証方法**:
- 生成ループのposition更新ロジックを確認

### 仮説3: RoPEインデックス計算のバグ ⭐

**可能性**: 中

**問題箇所**:
```rust
let rope_idx = position * (head_dim / 2) + i;
```

**llama.cppの実装**:
```c
const int64_t i0 = iq2 + i02_low;  // 位置インデックス
```

### 仮説4: Attention計算でのシーケンス長の誤解釈 ⭐

**可能性**: 中

**問題**:
- KVキャッシュなしでもトークン繰り返し
- 各生成ステップで前のトークンの情報が失われている

## 📋 次の検証手順

### Phase 1: Position パラメータの追跡 ⭐⭐⭐

1. **attention_layer, forward の position を出力**
   ```rust
   eprintln!("🐛 [STEP {}] position={}", step, position);
   ```

2. **期待される動作**:
   ```
   Step 0: position=20 (入力トークン数)
   Step 1: position=21 (入力 + 1)
   Step 2: position=22 (入力 + 2)
   ```

3. **もし position が更新されていなければ** → これが根本原因

### Phase 2: 生成ループの検証

1. **InferenceEngine の生成ループを確認**
   - `position`の計算ロジック
   - `kv_cache_len`の使用

2. **期待される実装**:
   ```rust
   for step in 0..max_tokens {
       let position = input_len + step;
       let logits = model.forward(&input_ids, position)?;
       // ...
   }
   ```

### Phase 3: RoPEインデックスの数値検証

1. **position=0での RoPE 値を出力**
   ```rust
   if position == 0 {
       eprintln!("RoPE cos[0..5]: {:?}", &self.rope_cos[0..5]);
   }
   ```

2. **llama.cppと比較**

## 💡 暫定的な結論

### 確認された事実

1. **KVキャッシュは問題の原因ではない**
   - 無効化しても同じ出力

2. **Logit分布が根本的に間違っている**
   - "drew" が最高logit (11.8283)
   - llama.cppとは全く異なる分布

3. **トークン繰り返しは副次的な症状**
   - 根本原因: 間違ったLogit分布
   - 各ステップで同じ分布 → 同じトークン選択

### 最も可能性の高い原因

**生成ループでの `position` パラメータの更新ミス** (確率: 80%)

**症状**:
- 全ての生成ステップで同じposition
- 同じRoPE値が適用される
- 同じAttention計算
- 同じLogit分布
- 同じトークン選択 → 繰り返し

**検証方法**:
1. `position`の値をステップごとに出力
2. 更新されていなければ確定

### 次のアクション

1. **positionパラメータのデバッグ出力追加** (最優先)
2. InferenceEngineの生成ループ確認
3. 修正後に再テスト
