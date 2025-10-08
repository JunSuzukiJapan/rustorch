# RusTorch vs llama.cpp 出力比較分析

**日時**: 2025-10-08
**ステータス**: ⚠️ **RusTorchに実装バグを確認**

## 🎯 決定的な証拠

### 同一モデル・同一入力での出力比較

| 実装 | モデル | プロンプト | 生成テキスト | 評価 |
|------|--------|-----------|------------|------|
| llama.cpp | Q6_K | "Hello" | "Hello world\n\nAnd then" | ✅ 正常 |
| llama.cpp | Q4_K_M | "Hello" | "Hello there! I'm" | ✅ 正常 |
| **RusTorch** | **Q6_K** | **"Hello"** | **"drew drew drew Superhé"** | ❌ **異常** |
| **RusTorch** | **Q4_K_M** | **"Hello"** | **"diplom"** | ❌ **異常** |
| **RusTorch** | **Q4_0** | **"Hello"** | **"ER"** | ❌ **異常** |

## 📊 詳細分析

### llama.cpp出力（正常）

#### Q6_K
```
Prompt: "Hello"
Output: "Hello world\n\nAnd then"
```
- 文法的に正しい英語
- 意味のある継続
- トークン繰り返しなし

#### Q4_K_M
```
Prompt: "Hello"
Output: "Hello there! I'm"
```
- 自然な挨拶
- 文法的に正しい
- 適切な継続

### RusTorch出力（異常）

#### Q6_K (5トークン生成)
```
Prompt: "Hello"
Output: "drew drew drew Superhé"
Tokens: [15010, 15010, 15010, 5670, 19880]
```

**問題**:
1. **トークン繰り返し**: "drew" が3回連続
2. **意味不明**: "drew Superhé" は文法的に不正
3. **退化パターン**: 典型的な degeneration

#### Q4_K_M (1トークン)
```
Prompt: "Hello"
Output: "diplom"
Token: 13487
Logit rank: 6位
```

**問題**:
1. **不適切なトークン**: "diplom" は文脈に合わない
2. **ランクが低い**: Top-1ではなく6位のトークンを選択
3. **意味不明**: 挨拶の継続として不自然

#### Q4_0 (1トークン)
```
Prompt: "Hello"
Output: "ER"
Token: 1001
Logit rank: 1位
```

**問題**:
1. **不適切なトークン**: "ER" は文脈に合わない
2. **Top-1でも間違い**: 最高Logitでも不正なトークン

## 🔍 RusTorchの問題箇所

### Layer出力の成長パターン

#### llama.cpp (推定)
```
Layer 0 → Layer 21: 適度な成長、安定した値
```

#### RusTorch Q6_K
```
Layer 0:  RMS=0.011214, Mean=0.000046
Layer 5:  RMS=0.175816, Mean=0.002492  (15.7x)
Layer 10: RMS=0.349671, Mean=-0.004714 (31.2x)
Layer 15: RMS=0.573687, Mean=-0.010374 (51.2x)
Layer 21: RMS=1.259880, Mean=-0.009707 (112.4x)
Final:    RMS=1.918744, Mean=-0.039752
```

**異常な成長**:
- Layer 0 → 21で **112倍** のRMS成長
- Finalで **171倍** の成長

### 退化（Degeneration）の証拠

**Q6_K 5トークン生成:**
```
Token 0: 15010 -> 'drew'
Token 1: 15010 -> 'drew'  ← 同じトークン
Token 2: 15010 -> 'drew'  ← 同じトークン
Token 3: 5670  -> 'Super'
Token 4: 19880 -> 'hé'
```

**典型的な退化パターン**:
1. 同一トークンの連続生成
2. KVキャッシュの問題の可能性
3. Logit分布の崩壊

## 🧠 根本原因の推論

### 仮説1: RMSNormの数値誤差

**可能性**: 中
- RMSの計算で微小な誤差
- 22層で累積・増幅
- 最終的にLogit分布が歪む

**反証**:
- RMSNorm実装は標準的
- Epsilon値 (1e-5) は正しい

### 仮説2: RoPEの実装ミス

**可能性**: 高
- 位置エンコーディングの誤差
- 各トークンの位置情報が不正確
- Attentionの計算が狂う

**検証方法**:
- llama.cppのRoPE実装と比較
- 位置0での値を数値的に比較

### 仮説3: Attentionスコアの計算誤差

**可能性**: 高
- Softmaxの数値不安定性
- Attention weightsの計算ミス
- KVキャッシュの不正な更新

**検証方法**:
- Layer 0のAttention出力を比較
- Attention scoresの分布を確認

### 仮説4: KVキャッシュの実装バグ

**可能性**: **最高**
- トークン繰り返しは典型的なKVキャッシュバグ
- 位置インデックスのズレ
- キャッシュの更新ロジックミス

**検証方法**:
- KVキャッシュを無効にしてテスト
- キャッシュの内容をダンプして確認

### 仮説5: 最終Linear層の重み読み込みミス

**可能性**: 中
- LM headの重みが不正
- 量子化デコードのバグ
- テンソル形状の誤解釈

**検証方法**:
- output.weightの統計を比較
- 最初の数個の値を数値比較

## 🎯 最優先で確認すべき箇所

### 1. KVキャッシュの実装 ⭐⭐⭐

**ファイル**: `src/hybrid_f32/models/llama.rs`

**確認ポイント**:
- `grouped_query_attention`関数
- `cached_k`, `cached_v`の更新ロジック
- 位置インデックスの計算

**期待される動作**:
```rust
// 新しいトークン位置: position
// キャッシュ位置: 0..position-1
// 新しいK/V: position位置に追加
```

### 2. RoPEの位置計算 ⭐⭐

**ファイル**: `src/hybrid_f32/models/llama.rs`

**確認ポイント**:
- `apply_rope`関数
- `position`パラメータの使用
- `rope_idx`の計算式

**llama.cppとの比較**:
```c
// llama.cpp
int64_t i0 = position;
float freq = (float)rope_theta * powf(base, (float)(2*i)/n_rot);
float val = i0 * freq;
```

### 3. Attention計算 ⭐⭐

**確認ポイント**:
- Softmaxの数値安定性
- Attention scoresのスケーリング
- `1.0 / sqrt(head_dim)` の計算

### 4. 最終Logit計算 ⭐

**確認ポイント**:
- `output.weight`の読み込み
- LM headの計算
- 最終RMSNormの適用

## 📋 検証手順

### Phase 1: KVキャッシュ検証

1. **KVキャッシュを無効化**
   ```rust
   // cached_k, cached_v を常に None に設定
   let (attn_out, new_k, new_v) = self.grouped_query_attention(&q_proj, &k_proj, &v_proj, None, None)?;
   ```

2. **トークン繰り返しが解消されるか確認**
   - 解消される → KVキャッシュバグ確定
   - 解消されない → 他の原因

### Phase 2: Layer 0出力比較

1. **llama.cppでLayer 0出力をダンプ**
   - `--verbose`フラグで詳細ログ
   - Layer 0のAttention出力を記録

2. **RusTorchと数値比較**
   - 完全一致すべき（量子化誤差以内）
   - 不一致 → RoPEまたはAttentionのバグ

### Phase 3: RoPE数値検証

1. **Position 0でのRoPE値を出力**
   ```rust
   if position == 0 {
       eprintln!("RoPE cos[0..10]: {:?}", &self.rope_cos[0..10]);
       eprintln!("RoPE sin[0..10]: {:?}", &self.rope_sin[0..10]);
   }
   ```

2. **llama.cppと比較**

### Phase 4: 最終Linear層検証

1. **`output.weight`の統計を出力**
   ```rust
   let stats = Self::compute_stats(output_weight.as_slice());
   eprintln!("output.weight: rms={}, mean={}", stats.rms, stats.mean);
   ```

2. **llama.cppと比較**

## 💡 暫定的な回避策

### 短期的対応

1. **KVキャッシュを無効化**
   - トークン繰り返しが解消される可能性
   - パフォーマンスは低下するが正確性が向上

2. **Temperature = 1.0に設定**
   - Logit分布の歪みを軽減
   - Top-kサンプリングを有効化

3. **Repetition penaltyを強化**
   - 同じトークンの繰り返しを抑制

### 長期的対応

1. **llama.cppとの数値的完全一致を目指す**
   - Layer-by-layer比較
   - 全ての中間値を検証

2. **統合テストの追加**
   - 既知の入出力ペアでのテスト
   - llama.cppとの出力一致を保証

3. **数値精度の向上**
   - Float64での中間計算
   - 数値的に安定したアルゴリズム

## 🎯 結論

### 確認された事実

1. **llama.cppは正常に動作**
   - Q6_K, Q4_K_M共に正しい英語を生成
   - 同一モデルで一貫した結果

2. **RusTorchは異常な出力を生成**
   - 全ての量子化方式で問題発生
   - トークン繰り返し（退化）
   - 意味不明なトークン選択

3. **問題は実装バグ、量子化ではない**
   - llama.cppで正常動作 → モデル自体は正しい
   - 全量子化方式で問題発生 → 量子化の問題ではない

### 最も可能性の高い原因

**KVキャッシュの実装バグ** (確率: 70%)
- トークン繰り返しは典型的な症状
- 位置インデックスまたは更新ロジックのミス

**次点: RoPEの実装ミス** (確率: 20%)
- 位置エンコーディングの誤差が累積
- Attentionの計算が狂う

**その他の可能性** (確率: 10%)
- Attention計算のバグ
- 最終Linear層のバグ

### 推奨される次のアクション

1. **KVキャッシュを無効化してテスト** (最優先)
2. **llama.cppとのLayer 0出力比較**
3. **RoPE値の数値検証**
