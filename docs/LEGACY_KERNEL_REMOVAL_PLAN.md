# Legacy Kernel Removal Plan

## Overview
バッチ対応カーネル実装完了後、レガシー（単一シーケンス専用）カーネルを段階的に削除する計画。

## 削除対象カーネル

1. `apply_rope_single_f32` - [metal_shaders.metal:420-456](../src/gpu/metal_shaders.metal#L420-L456)
2. `compute_attention_scores_f32` - [metal_shaders.metal:502-533](../src/gpu/metal_shaders.metal#L502-L533)
3. `softmax_max_f32` - [metal_shaders.metal:569-595](../src/gpu/metal_shaders.metal#L569-L595)
4. `softmax_exp_sum_f32` - 既存（バッチ未対応版）
5. `softmax_normalize_f32` - 既存（バッチ未対応版）
6. `apply_attention_to_values_f32` - [metal_shaders.metal:690-718](../src/gpu/metal_shaders.metal#L690-L718)

**合計**: 約150行のコード削減見込み

## 削除前提条件（Checklist）

### Phase 3: 統合完了
- [ ] Rustカーネルラッパー実装（metal_kernels.rs）
- [ ] forward_batch_metal最適化実装
- [ ] すべての既存コードがバッチカーネル使用に移行

### Phase 4: 検証完了
- [ ] ユニットテスト：バッチカーネル vs レガシーカーネル結果比較
- [ ] パフォーマンステスト：batch_size=1でレガシーと同等以上
- [ ] 統合テスト：全quantization types（Q4_K_M, Q5_K_M, Q6_K, Q8_0）で動作確認
- [ ] 本番環境：最低2週間の安定動作

### 安全性確認
- [ ] ロールバック手順文書化
- [ ] 削除前にブランチ作成（backup/legacy-kernels）
- [ ] CI/CDでバッチカーネル専用ビルドテスト

## 削除実行手順

### Step 1: Deprecation Warning追加（Phase 3開始時）

```metal
// ⚠️ DEPRECATED: Use apply_rope_f32 with batch_size=1 instead
// This kernel will be removed in v0.7.0
kernel void apply_rope_single_f32(...) {
    // existing implementation
}
```

### Step 2: 使用箇所を段階的に置き換え（Phase 3-4）

```rust
// Before (レガシー使用)
apply_rope_single_f32(x, start_pos, seq_len, ...);

// After (バッチカーネル使用、batch_size=1)
apply_rope_f32(x, /*batch_size=*/1, start_pos, seq_len, ...);
```

### Step 3: レガシーカーネル削除（Phase 4完了後）

1. バックアップブランチ作成
   ```bash
   git checkout -b backup/legacy-kernels
   git push origin backup/legacy-kernels
   ```

2. メインブランチで削除
   ```bash
   git checkout main
   # metal_shaders.metalから削除
   git commit -m "refactor: Remove legacy single-sequence kernels"
   ```

3. 削除後テスト
   - 全ビルドテスト
   - 全ユニットテスト
   - パフォーマンスベンチマーク

## 削除タイムライン（予想）

- **Phase 3完了**: 2-3週間後 → Deprecation warning追加
- **Phase 4完了**: 4-6週間後 → 本番環境テスト開始
- **最終削除**: 6-8週間後 → 安定動作確認後に削除

## ロールバック計画

問題発生時の対応：

1. **即座のロールバック**
   ```bash
   git revert <deletion_commit>
   git push origin main
   ```

2. **バックアップからの復元**
   ```bash
   git checkout backup/legacy-kernels -- src/gpu/metal_shaders.metal
   git commit -m "revert: Restore legacy kernels due to <issue>"
   ```

3. **フィーチャーフラグで制御**（理想）
   ```rust
   #[cfg(feature = "use-legacy-kernels")]
   fn use_legacy_rope() { ... }
   ```

## メリット vs デメリット

### 削除のメリット ✅
- コードベース簡素化（~150行削減）
- メンテナンス負荷軽減
- 混乱の回避（どちらを使うべきか）
- コンパイル時間短縮

### 削除のデメリット ❌
- ロールバック困難
- バッチカーネルにバグがあった場合の代替手段なし
- パフォーマンス退化のリスク（batch_size=1で）

## 結論

**現時点（Phase 2完了）での削除は時期尚早**。Phase 3-4完了後、本ドキュメントのチェックリストを満たした時点で削除を実行すべき。

**推奨行動**：
1. Phase 3-4実装を優先
2. 統合テスト・パフォーマンステスト実施
3. 本番環境で2週間安定動作確認
4. チェックリスト完了後に削除実行

---

**Status**: Deletion Plan Documented
**Target Deletion Date**: Phase 4完了 + 2週間後
**Last Updated**: 2025-10-10
