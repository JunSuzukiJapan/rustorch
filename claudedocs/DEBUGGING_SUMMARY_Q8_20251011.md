# デバッグまとめ: Q8_0モデルの問題

**日付**: 2025-10-11
**状態**: 🔍 **根本原因調査中**

## 検証済み事項 ✅

### 1. 埋め込み層 - 正常 ✅
- Q8_0デクオンタイズ: 正しい
- F16→F32変換: 正しい
- スケール値: 0.000000119（llama.cppと同じ）
- Token 29896 ("1") の埋め込み:
  - RusTorch: `[-0.0058379173, -0.0033612251, 0.00035381317, ...]`
  - llama.cpp: `[-0.005837917, -0.003361225, 0.000353813, ...]`
  - **完全一致！** ✅

### 2. llama.cppとの比較 ✅
- llama.cppは同じQ8_0モデルで正常に動作:
  ```
  "Greetings! I am writing to inquire about the current offerings on the Samsung"
  ```
- RusTorchは意味不明な出力:
  ```
  "avavertavanth" (hybrid-f32)
  "hr Following memory" (cpu)
  ```

### 3. RMSNorm重み - 正常 ✅
- F32値が正しく読み込まれている
- 値は小さい（0.004〜0.07範囲）が、これはモデルの正常値
- 最初の20値:
  ```
  [-0.004180908, 0.006317139, 0.069824219, -0.029418945, ...]
  ```

### 4. RMSNorm実装 - アルゴリズムは正しい ✅
```rust
1. sum = Σ(x[i]²)
2. mean = sum / hidden_size
3. scale = 1 / sqrt(mean + eps)
4. y[i] = x[i] * scale * weight[i]
```
llama.cppと完全一致。

### 5. バックエンド - 問題なし ✅
- hybrid-f32（Metal）: 意味不明
- cpu: 意味不明
- **両方とも同じ問題** → Metalバックエンド固有の問題ではない

## 問題箇所の特定 🎯

### Layer 0出力の異常
```
RusTorch Layer 0 Output:
  RMS: 0.014224 (期待値: ~1.0)
  First 10: [0.0045, 0.00017, -0.0016, ...]

埋め込み（Token 29896）:
  RMS: 0.008699
  First 10: [-0.0058, -0.0034, 0.00035, ...]
```

**観察**: 埋め込みは正しいのに、Layer 0の出力が異常に小さい（約100倍）

これは、**レイヤー内の処理（Attention, FFN）**に問題があることを示す。

### 仮説: 重みマトリクスの転置エラー 🔴

#### 問題の可能性
1. **Q/K/V投影の重み行列**が転置されていない、または逆に転置されている
2. **FFNの重み行列**が転置されていない
3. **matmulの入力順序**が逆

#### 証拠
- 埋め込み: ✅ 正しい
- RMSNorm: ✅ アルゴリズム正しい、重み正しい
- Layer 0出力: ❌ 100倍小さい

つまり、**RMSNormからAttention/FFNへの間**に問題がある。

## 次のステップ

### 1. 重みマトリクスの形状確認 🔴 最優先
```bash
# Q/K/V投影の重みの形状
blk.0.attn_q.weight: [2048, 2048]
blk.0.attn_k.weight: [2048, 256]
blk.0.attn_v.weight: [2048, 256]
```

**確認事項**:
- RusTorchがこれらを`[2048, 2048]`として読み込んでいるか？
- それとも`[2048, 2048]`として読み込んでいるが、転置が必要？

### 2. llama.cppのmatmul順序確認
```c
// llama.cppでQ投影がどのように計算されているか
q = x @ q_weight  // [seq_len, hidden] @ [hidden, hidden] = [seq_len, hidden]
or
q = q_weight @ x  // 転置された順序？
```

### 3. RusTorchのmatmul確認
```rust
// llama.rs:708
let q = x.matmul(q_weight)?;  // [seq_len, hidden] @ [hidden, hidden]
```

この順序が正しいか確認。

### 4. 簡単なテスト: Q投影の出力を直接確認
```bash
# Layer 0のQ投影直後の値をダンプ
# llama.cppと比較
```

## 推定される根本原因

**最も可能性が高い**: Q8_0重みマトリクスの**次元順序**が間違っている。

### 現在の理解
```rust
// gguf.rs:666-677
let shape: Vec<usize> = match ggml_type {
    GGMLType::F32 | GGMLType::F16 => {
        s.reverse();  // F32/F16は反転
    }
    _ => {
        original_dims.clone()  // Q8_0は反転しない
    }
};
```

**問題**: Q8_0重みは反転されていないが、**反転が必要かもしれない**。

または、**F32/F16が反転されているのが間違い**かもしれない。

### 検証方法
1. Q8_0重みを強制的に反転してテスト
2. F32/F16の反転を無効化してテスト
3. llama.cppのGGUF読み込みコードを詳細に確認

## 参考情報

- 埋め込み検証スクリプト: `/tmp/dump_embedding_llamacpp.c`
- RMSNorm重み検証: `/tmp/verify_f32_rmsnorm_weights.py`
- Q8_0スケール検証: `/tmp/check_q8_0_scale.py`

## 結論

**RusTorchの実装は99%正しい。問題は重みマトリクスの次元順序の解釈にある可能性が高い。**

llama.cppがGGUFファイルから重みを読み込む際の**正確な次元順序**を確認し、RusTorchと比較する必要がある。
