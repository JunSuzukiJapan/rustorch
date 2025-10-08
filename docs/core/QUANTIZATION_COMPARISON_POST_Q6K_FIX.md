# Quantization Format Comparison After Q6_K Fix

**Date**: 2025-10-08
**Status**: Q6_K dequantization fixed, but token generation still incorrect across all formats

## Test Results Summary

### llama.cpp (正解)
すべての量子化フォーマットで正しい出力：
```
Prompt: "Hello world"
Q4_K_M: "I am glad to"
Q6_K:   "Write a Python program"
```

### RusTorch (Q6_K fix後)

| Format | Status | Output | Token IDs | Notes |
|--------|--------|--------|-----------|-------|
| Q4_K_M | ❌ Wrong | "тивmiglicognregretmask" | [3499, 20379, 25323, 28883, 13168] | Changed from "drew drew drew" |
| Q5_K_M | ⏳ Testing | - | - | Model downloaded, pending test |
| Q6_K | ❌ Wrong | "leiчётleiчётlei" | [16301, 28651, 16301, 28651, 16301] | Repeating pattern |
| Q8_0 | ❌ Not supported | "Tensor type Q8_0 not yet supported" | - | Implementation missing |

## Key Observations

### 1. Q6_K修正の影響
- ✅ 逆量子化の数値範囲が正常化（~0.01）
- ❌ トークン生成は依然として間違い
- 📊 出力が変化した（Q4_K_M: "drew drew drew" → "тивmiglicognregretmask"）
  - これは修正が何らかの影響を与えたことを示す
  - しかし根本原因は解決されていない

### 2. 量子化フォーマット間の違い
- Q4_K_M: ランダムなトークン（多言語混在）
- Q6_K: 反復パターン（2つのトークンを交互に）
- どちらも正しい英語にならない

### 3. Q8_0フォーマット未サポート
Q8_0量子化形式は現在RusTorchでサポートされていません：
```
⚠️  Failed to load tensor 'token_embd.weight':
    Parse error: Tensor type Q8_0 not yet supported for loading
```

すべての重みテンソルがQ8_0形式のため、モデルがロードできず、
"Embedding weight not found"エラーとなる。

## 結論

### Q6_K修正の効果
1. ✅ **数値精度は改善**: 逆量子化の値が正常範囲に
2. ❌ **トークン生成は未解決**: 異なる誤ったトークンを生成
3. 📊 **修正の影響確認**: 出力が変化したことから、修正は何らかの影響を与えている

### 根本原因の特定
Q6_K逆量子化の修正だけでは不十分。問題は以下にあると推測：

1. **レイヤー間の値の成長** ([LAYER_VALUE_GROWTH_ANALYSIS.md](LAYER_VALUE_GROWTH_ANALYSIS.md))
   - レイヤー0出力 RMS: 0.015
   - レイヤー21出力 RMS: 1.124 (75倍成長)
   - 最終正規化後 RMS: 1.920 (128倍成長)

2. **他の量子化形式の問題**
   - Q4_K: サポート済みだがトークン生成誤り
   - Q5_K_M: 未テスト（ダウンロード中）
   - Q6_K: 修正済みだがトークン生成誤り
   - Q8_0: 未サポート

3. **可能性のある原因**
   - Q4_Kにも同様のインデックス問題がある可能性
   - RMSNorm、Attention、FFNの実装に問題がある可能性
   - 量子化形式に依存しない、より深い実装の問題

## 次のステップ

### 優先度1: Q4_K逆量子化の検証
Q6_Kと同様のインデックス問題がないか確認：
```bash
# Q4_Kのインターリーブパターンを調査
# llama.cppのdequantize_row_q4_Kと比較
```

### 優先度2: Q5_K_Mのテスト
ダウンロード完了後、Q5_K_Mの動作確認

### 優先度3: レイヤー値成長の調査
llama.cppとの中間値比較で、どこから値が発散するか特定

### 優先度4: Q8_0サポート追加
他の量子化形式が解決したら、Q8_0のサポートを追加

## 参考資料

- [Q6K_DEQUANTIZATION_FIX_RESULTS.md](Q6K_DEQUANTIZATION_FIX_RESULTS.md) - Q6_K修正の詳細
- [LAYER_VALUE_GROWTH_ANALYSIS.md](LAYER_VALUE_GROWTH_ANALYSIS.md) - 値の成長分析
- [RUSTORCH_VS_LLAMACPP_COMPARISON.md](RUSTORCH_VS_LLAMACPP_COMPARISON.md) - llama.cpp比較
