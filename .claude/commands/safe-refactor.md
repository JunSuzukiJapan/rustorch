# Safe Refactor Agent
# 既存項目保護リファクタリングエージェント

<!--
USAGE / 使用方法:
/safe-refactor [target_file]     - 指定ファイルを安全にリファクタリング
/safe-refactor baseline          - 現在の状態をベースラインとして記録
/safe-refactor verify            - 削除された項目がないか検証のみ実行

PROTECTION / 保護機能:
- 既存のpub fn/struct/enum/traitの削除を絶対禁止
- 関数シグネチャの変更を禁止
- 内部実装の改善のみ許可
- 削除が検出されたら即座に復元指示

SAFE CHANGES / 許可される変更:
✅ 内部アルゴリズムの最適化
✅ 新しいprivateヘルパー関数の追加
✅ エラーハンドリングの改善
✅ 新しいpublic関数の追加（既存に影響しない）

FORBIDDEN / 禁止事項:
❌ 既存public項目の削除
❌ 関数パラメータの変更
❌ 戻り値型の変更
❌ 破壊的変更
-->

## Command
`/safe-refactor [target]`

## Purpose
既存のAPI・関数・構造体を削除せずに、内部実装の改善とコード品質向上を行う専用エージェント。

## Core Principles

### 絶対的制約
1. **既存パブリック項目の削除禁止** - 例外なし
2. **シグネチャ変更禁止** - パラメータ・戻り値型不変
3. **内部実装のみ改善** - 外部インターフェース保持
4. **追加のみ許可** - 新機能は既存に影響しない方法で

## Agent Workflow

### Phase 1: Pre-Analysis (必須)
```bash
# 1. 既存項目の完全スキャン
echo "🔍 Scanning existing public items..."
rg "pub (fn|struct|enum|trait|const|impl)" --type rust -n > baseline_items.txt

# 2. 関数シグネチャの記録
rg "pub fn [^{]*" --type rust -o > function_signatures.txt

# 3. 保護対象の特定
echo "🛡️ Protection targets identified:"
wc -l baseline_items.txt function_signatures.txt
```

### Phase 2: Safe Refactoring Rules
```rust
// ✅ 許可される変更例
pub fn existing_function(&self) -> Result<T> {
    // Before: 非効率な実装
    old_slow_implementation()

    // After: 高速な内部実装 (インターフェース不変)
    new_fast_implementation()
}

// ✅ 新機能追加 (既存に影響なし)
pub fn existing_function_v2(&self) -> Result<T> {
    // 新しい改良版、既存関数は保持
    improved_implementation()
}

// ❌ 禁止される変更
// pub fn existing_function() の削除 → 絶対禁止
// pub fn existing_function(new_param: T) のシグネチャ変更 → 禁止
```

### Phase 3: During Refactoring
```markdown
## Refactoring Guidelines

### DO (実行可能)
- 内部ロジックの最適化
- アルゴリズムの改善
- エラーハンドリング強化
- ドキュメント追加
- 新しいヘルパー関数追加

### DON'T (禁止)
- 既存パブリック関数の削除
- 関数パラメータの変更
- 戻り値型の変更
- 既存構造体フィールドの削除
- トレイトメソッドの削除
```

### Phase 4: Post-Verification (必須)
```bash
# 1. 変更後スキャン
rg "pub (fn|struct|enum|trait|const|impl)" --type rust -n > current_items.txt

# 2. 削除検証
echo "🔍 Checking for deletions..."
if ! comm -23 baseline_items.txt current_items.txt | head; then
    echo "✅ No existing items deleted"
else
    echo "🚨 CRITICAL: Items were deleted!"
    echo "Must restore immediately:"
    comm -23 baseline_items.txt current_items.txt
fi

# 3. シグネチャ検証
rg "pub fn [^{]*" --type rust -o > current_signatures.txt
if ! diff function_signatures.txt current_signatures.txt; then
    echo "⚠️ Function signature changes detected"
fi
```

## Usage Examples

### Example 1: Performance Optimization
```bash
/safe-refactor src/tensor/operations.rs

# Agent will:
1. Scan existing public functions in operations.rs
2. Allow internal algorithm improvements
3. Prevent deletion of any existing public functions
4. Verify all original functions remain accessible
```

### Example 2: Code Organization
```bash
/safe-refactor src/gpu/kernels.rs

# Agent will:
1. Protect all existing pub fn, pub struct, pub trait
2. Allow splitting large functions into smaller private helpers
3. Allow adding new public functions (non-breaking additions)
4. Ensure all original public interfaces remain unchanged
```

### Example 3: Error Handling Improvement
```bash
/safe-refactor src/hybrid_f32/tensor/core.rs

# Agent will:
1. Preserve all existing method signatures
2. Allow improving internal error handling
3. Allow adding new Result types for new functions
4. Prevent changing existing function return types
```

## Emergency Response

### If Deletion Detected
```bash
🚨 DELETION DETECTED - IMMEDIATE ACTION REQUIRED

Deleted items:
- pub fn gpu_sum(&self) -> Result<Tensor>
- pub fn __add__(&self, other: &Tensor) -> Result<Tensor>

RESTORATION COMMANDS:
1. git checkout HEAD~1 -- [affected_files]
2. Manually restore deleted functions
3. Re-apply only internal improvements
4. Verify with /safe-refactor verify
```

## Agent Parameters

### /safe-refactor [target]
- `target`: File path or module to refactor safely
- If no target specified, operates on current working directory

### /safe-refactor verify
- Runs post-refactoring verification only
- Checks for any deletions since last baseline

### /safe-refactor baseline
- Creates new baseline of existing items
- Use before starting major refactoring session

## Agent Success Criteria

### ✅ Successful Refactoring
- All baseline public items remain accessible
- Internal implementation improved
- Code quality metrics improved
- No breaking changes introduced

### ❌ Failed Refactoring (Must Fix)
- Any existing public item deleted
- Function signature changed
- Breaking changes introduced
- User code would break after changes

## Integration with Development Workflow

```bash
# Before any refactoring work
/safe-refactor baseline

# During refactoring
/safe-refactor [target_file]

# After refactoring
/safe-refactor verify

# If working correctly:
git add .
git commit -m "refactor: internal improvements (no breaking changes)"

# If deletions detected:
# 1. Restore deleted items
# 2. Re-apply only safe changes
# 3. Re-verify
```

This agent ensures that refactoring improves code quality while maintaining 100% backward compatibility by preventing any deletion of existing public interfaces.