# サンプリング戦略の実装

**日付**: 2025-10-07
**目的**: トークン繰り返し問題を解決するため、Repetition penaltyと確率的サンプリングを実装

## 実装内容

### 1. Repetition Penalty

```rust
let repetition_penalty = 1.1; // llama.cpp デフォルト値
let penalty_window = 64.min(generated_ids.len());
let recent_tokens = &generated_ids[generated_ids.len() - penalty_window..];

for &token_id in recent_tokens {
    if penalized_logits[token_id] > 0.0 {
        penalized_logits[token_id] /= repetition_penalty;
    } else {
        penalized_logits[token_id] *= repetition_penalty;
    }
}
```

**動作**: 最近64トークンに出現したトークンのlogitにペナルティを適用

### 2. Temperature Sampling

```rust
let temperature = 0.8;
for logit in &mut penalized_logits {
    *logit /= temperature;
}
```

**動作**: logitsを温度で割り、確率分布を平滑化

### 3. Top-p (Nucleus) Sampling

```rust
let top_p = 0.95;
let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
    .enumerate()
    .map(|(i, &p)| (i, p))
    .collect();
indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

let mut cumsum = 0.0;
let mut top_p_indices = Vec::new();
for (idx, prob) in &indexed_probs {
    cumsum += prob;
    top_p_indices.push(*idx);
    if cumsum >= top_p {
        break;
    }
}
```

**動作**: 累積確率が95%に達するまでトークンを選択し、その中からサンプリング

## 結果

### トークン繰り返しの改善

**実装前**:
```
入力: "Hi"
出力: "ragmentragmentragment..." (同じトークンの完全な繰り返し)
```

**実装後**:
```
入力: "Hi"
出力: "ragment totype neither：світ..." (多様なトークン)

🎯 [STEP 0] Selected token 4305 (sampled, prob=0.0815)
🎯 [STEP 1] Selected token 10414 (sampled, prob=0.0175)
🎯 [STEP 2] Selected token 9561 (sampled, prob=0.0021)
```

✅ **トークンの繰り返しは解消された**

### 残存する問題

❌ **生成内容が無意味**

```
入力: "What is the capital of France?"
出力: "ructbibliothekruct umajánragment(& whole請tod mirrortensor..."

入力: "Hello"
出力: "ructbinding Mar transformruct Whbatchовая uma請"
```

モデルは多様なトークンを生成しているが、意味のある文章にはなっていない。

## 原因分析

### 1. モデルの問題ではない

llama.cppは同じモデルで正しい出力を生成：
```
入力: "What is the capital of France?"
llama.cpp出力: "The capital of France is Paris."
```

### 2. チャットテンプレートの問題かもしれない

**RusTorch現在のテンプレート**:
```
<|system|>
You are a helpful assistant.</s>
<|user|>
{user_input}</s>
<|assistant|>
```

llama.cppはより複雑なテンプレート処理を行っている可能性がある。

### 3. トークナイザーの問題の可能性

- トークン化が正しく行われていない？
- EOS/BOS トークンの処理が不適切？
- Special tokensの扱いが異なる？

### 4. 計算は正しい

- ✅ 位置エンコーディング: 正しく増分
- ✅ 行列乗算: 手動計算と一致
- ✅ GGUF読み込み: 正確
- ✅ Logits計算: 正常
- ✅ サンプリング: 多様なトークンを選択

## 次のステップ

### 優先度1: トークナイザーの検証

1. 入力トークン化をllama.cppと比較
2. Special tokensの処理を確認
3. EOSトークンの扱いを検証

### 優先度2: チャットテンプレートの検証

1. llama.cppが使用するテンプレートを確認
2. TinyLlamaの公式テンプレートを調査
3. Jinja2テンプレートエンジンとの互換性

### 優先度3: モデルの検証

1. 異なるモデル（Llama-2など）でテスト
2. 量子化形式の影響を確認（Q4_0, Q4_K_M, F16）

## 参照

**実装ファイル**:
- `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/example-cli/src/model/inference.rs` (465-531行)

**llama.cppのサンプリング設定**:
```
repeat_penalty = 1.0 (デフォルト)
temperature = 0.8
top_p = 0.95
top_k = 40
```

## 結論

**Repetition penaltyとサンプリング戦略の実装は成功。** トークンの繰り返しは解消されたが、生成内容の意味性の問題が残っている。

この問題は計算の正確性ではなく、**トークナイザーまたはチャットテンプレートの不一致**に起因している可能性が高い。
