# RusTorch 不正解出力デバッグ戦略

**最終更新**: 2025-10-10
**問題**: 入力 "1" に対して不正なtoken 25323 ("cogn") を生成

---

## 📋 検証済みコンポーネント（再検証不要）

### ✅ 1. Token Embedding
**検証日**: 2025-10-09
**方法**: `examples/verify_rms_norm_and_embeddings.rs`
**結果**: Token 29896の埋め込み値が正常範囲内
**結論**: 正しく動作

### ✅ 2. トークナイザー
**検証日**: 2025-10-09
**方法**: llama.cppとの出力比較
**結果**: 完全一致（チャットテンプレート修正後）
**結論**: 正しく動作

### ✅ 3. RoPE周波数事前計算
**検証日**: 2025-10-10
**場所**: `src/hybrid_f32/models/llama.rs:138-173`
**結果**:
- Position 0: cos=1.0, sin=0.0 ✅
- Position 1+: 正しい周波数値 ✅
**結論**: 正しく動作

### ✅ 4. RoPE回転適用
**検証日**: 2025-10-10
**場所**: `src/hybrid_f32/models/llama.rs:387-461`
**結果**:
- Token 0: 恒等変換（回転なし）✅
- Token 1+: 正しく回転適用 ✅
**結論**: 正しく動作、2D/3D layout差異は問題なし

### ✅ 5. Attention構造
**検証日**: 2025-10-10
**場所**: `src/hybrid_f32/models/llama.rs:468-594`
**結果**:
- Causal masking: 正しい ✅
- Softmax計算: 正しい ✅
- Attention weights正規化: 正しい ✅
**結論**: 構造は正しく動作

### ✅ 6. その他（過去に検証済み）
- Q/K/V Weight形状: 正しい
- Metal Matmul Kernel: 正しい
- RMS Norm hidden_size: 2048で正しい
- GGUF Memory Layout: 正しい

---

## 🔍 未検証コンポーネント（優先順位順）

### 🔥 優先度1: Q/K/V Projection値の範囲検証

**問題の兆候**:
- Attention raw scoresが異常に小さい（0.000019, 0.000649など）
- Q·K^T の内積値が期待より小さい可能性

**検証方法**:
1. Q/K/V projection直後の値を出力
2. 第一ヘッド、最初の10要素を確認
3. RMS（二乗平均平方根）と範囲を計算
4. llama.cppの同位置の値と比較

**実装場所**: `src/hybrid_f32/models/llama.rs` forward関数内
- Q projection後（Line 234付近）
- K projection後（Line 241付近）
- V projection後（Line 248付近）

**期待される出力**:
```rust
eprintln!("🎯 [Q PROJECTION] First 10: {:?}", &q_data[..10]);
eprintln!("🎯 [Q PROJECTION] RMS: {:.6}", rms);
```

**判定基準**:
- RMS値が 0.01 ～ 1.0 程度なら正常
- 極端に小さい（< 0.001）なら異常

---

### 🔥 優先度2: RMS Norm出力値の検証

**問題の兆候**:
- 過去セッションでRMS Norm出力が異常（2.14倍大きい）という報告あり
- ただしhidden_size=2048は正しいことを確認済み

**検証方法**:
1. Layer 0 Attention前のRMS Norm出力を確認
2. 入力RMS、Weight RMS、出力RMSを記録
3. 数式通り: `output_RMS ≈ weight_RMS` を確認

**実装場所**: `src/hybrid_f32/models/llama.rs` rms_norm_f32関数
- Line 1470付近にデバッグ出力追加

**判定基準**:
```
Normalized input RMS ≈ 1.0
Output RMS ≈ Weight RMS
Output RMS / Weight RMS の比率 ≈ 1.0（誤差±10%以内）
```

---

### 🔥 優先度3: llama.cppとの数値比較

**目的**: どの層で最初に発散するかを特定

**比較ポイント**:
1. Token Embedding出力（Token 7: ID=29896）
2. Layer 0 Attention前 RMS Norm出力
3. Layer 0 Q projection出力
4. Layer 0 Attention出力
5. Layer 0 FFN出力
6. Layer 0 最終出力（residual後）

**実装方法**:
- llama.cpp側: `llama_decode`にprintfを追加してビルド
- RusTorch側: 既存デバッグ出力を活用
- Python scriptで数値差分を計算

**判定基準**:
- 相対誤差 < 1e-3: 一致
- 相対誤差 1e-3 ～ 1e-2: 要注意
- 相対誤差 > 1e-2: 発散開始

---

### 🟡 優先度4: FFN計算の検証

**前提**: Priority 1-3で問題が見つからなかった場合のみ

**検証方法**:
1. Gate projection出力
2. Up projection出力
3. SiLU activation後
4. Down projection出力

**実装場所**: `src/hybrid_f32/models/llama.rs` forward関数内のFFN部分

---

### 🟡 優先度5: 最終Logits計算の検証

**前提**: Priority 1-4で問題が見つからなかった場合のみ

**検証方法**:
1. Output projection前のhidden state
2. Output projection後のlogits
3. Token 29896のlogit値とランキング

**実装場所**: `src/hybrid_f32/models/llama.rs` forward関数の最終部分

---

## 📐 検証の実施順序

### フェーズ1: 浅い層の検証（1-2時間）

```
1. Q/K/V Projection値の範囲確認
   ↓
2. RMS Norm出力の数値検証
   ↓
3. 判定: 異常があればそこを深掘り、なければフェーズ2へ
```

### フェーズ2: llama.cpp比較（2-3時間）

```
1. llama.cppのビルドとデバッグ出力追加
   ↓
2. Token Embedding～Layer 0の全ステップを比較
   ↓
3. 最初に発散する場所を特定
   ↓
4. 該当コンポーネントの実装を詳細調査
```

### フェーズ3: 深い層の検証（必要に応じて）

```
1. FFN計算の詳細検証
   ↓
2. 複数層での累積誤差の確認
   ↓
3. 最終Logits計算の検証
```

---

## 🚫 避けるべきこと（重複検証の防止）

### やってはいけない検証:
1. ❌ RoPEの再検証（既に完全検証済み）
2. ❌ Token Embeddingの再確認（検証済み）
3. ❌ Attention構造の再確認（causal masking, softmax検証済み）
4. ❌ Weight transposeの調査（不要と確認済み）
5. ❌ Metal Kernelの再テスト（検証済み）

### 検証前に必ず確認:
- `METAL_GPU_DEBUGGING_STATUS.md`の「✅ 検証済みコンポーネント」セクション
- このドキュメントの「検証済みコンポーネント」セクション
- 同じ検証をしようとしていないか

---

## 📝 デバッグ出力の命名規則

### 絵文字マーカー:
- 🔧: Weight情報
- 💫: Attention関連
- 🌀: RoPE関連
- 🎯: Projection/Matmul出力
- 📊: 統計情報（RMS, mean, range）
- ⚠️: 警告・異常値
- ✅: 検証完了・正常

### 出力フォーマット:
```rust
eprintln!("🎯 [COMPONENT] description: value");
eprintln!("📊 [STATS] rms={:.6}, range=[{:.6}, {:.6}]", rms, min, max);
```

---

## 🎯 成功条件

### デバッグ完了の定義:
1. 不正解の根本原因を特定
2. 修正を実装
3. 入力 "1" に対して正しい出力を生成
4. 複数の入力でllama.cppと同等の出力を確認

### 中間マイルストーン:
- [ ] Q/K/V projection値の範囲を確認
- [ ] RMS Norm出力の正常性を確認
- [ ] llama.cppとの数値比較で発散点を特定
- [ ] 根本原因を特定
- [ ] 修正を実装
- [ ] テストで検証

---

## 📚 関連ドキュメント

- `METAL_GPU_DEBUGGING_STATUS.md`: 全体的な調査履歴と検証済み項目
- `docs/core/METAL_GPU_VERIFICATION.md`: Metal GPU実装の検証レポート
- `/tmp/FINAL_DIAGNOSIS.md`: 過去セッションの診断レポート

---

## 🔄 このドキュメントの更新ルール

### 検証完了時:
1. 該当項目を「検証済みコンポーネント」に移動
2. 検証日、方法、結果、結論を記載
3. 「未検証コンポーネント」から削除

### 新たな問題発見時:
1. 「未検証コンポーネント」に追加
2. 優先度を設定（🔥高、🟡中、🟢低）
3. 検証方法と判定基準を明記

### 重複防止:
- 検証開始前に必ずこのドキュメントを確認
- 同じ検証は絶対に繰り返さない
- 疑問があれば`METAL_GPU_DEBUGGING_STATUS.md`も参照

---

*このドキュメントは調査の進捗に応じて随時更新されます*
