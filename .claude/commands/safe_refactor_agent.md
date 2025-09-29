# Safe Refactoring Agent
# 安全リファクタリングエージェント

## 🎯 Purpose / 目的
既存項目（API、関数、構造体、トレイト等）の削除を防ぎながら、内部実装の改善とコード品質向上を行う専用エージェント。

## 🛡️ Core Principles / 基本原則

### 絶対的制約
1. **既存のパブリック項目は削除禁止** - 例外なし
2. **シグネチャ変更禁止** - パラメータ・戻り値型の変更不可
3. **内部実装のみ改善** - 外部インターフェースは不変
4. **追加のみ許可** - 新機能は既存機能に影響しない方法で追加

### 作業前必須チェック
1. **既存項目の完全スキャン** - 削除対象がないか事前確認
2. **依存関係マッピング** - 既存項目の使用状況把握
3. **影響範囲評価** - 変更が既存項目に与える影響を評価

## 🔧 Agent Implementation / エージェント実装

### Phase 1: Pre-Refactoring Analysis / リファクタリング前分析

```bash
#!/bin/bash
# safe_refactor_analysis.sh

echo "🔍 Safe Refactoring Pre-Analysis"
echo "================================"

# 1. 既存パブリック項目の完全リスト作成
echo "📋 Step 1: Scanning existing public items"
cargo doc --no-deps --document-private-items 2>/dev/null
find target/doc -name "*.html" | xargs grep -l "pub fn\|pub struct\|pub enum\|pub trait" > existing_items.log

# 2. パブリックAPIの抽出
echo "📋 Step 2: Extracting public API signatures"
cargo expand | grep -E "pub (fn|struct|enum|trait|const|static|mod)" > api_baseline.txt

# 3. 関数・メソッドシグネチャの記録
echo "📋 Step 3: Recording function signatures"
rg "pub fn [^{]*" --type rust -o > function_signatures.txt

# 4. トレイト・構造体の記録
echo "📋 Step 4: Recording types and traits"
rg "pub (struct|enum|trait|const) [^{]*" --type rust -o > types_baseline.txt

echo "✅ Pre-analysis complete. Baseline established."
echo "📁 Files created:"
echo "   - existing_items.log"
echo "   - api_baseline.txt"
echo "   - function_signatures.txt"
echo "   - types_baseline.txt"
```

### Phase 2: Safe Refactoring Guidelines / 安全リファクタリングガイドライン

```rust
// Safe Refactoring Checklist
// 安全リファクタリングチェックリスト

pub struct SafeRefactorGuard {
    original_items: HashSet<String>,
    protected_signatures: HashMap<String, String>,
}

impl SafeRefactorGuard {
    /// リファクタリング開始前の状態保存
    pub fn capture_baseline() -> Self {
        // 1. 既存パブリック項目をスキャン
        // 2. シグネチャを記録
        // 3. 保護対象を特定
        todo!("実装: 既存項目の完全キャプチャ")
    }

    /// 変更後の検証
    pub fn verify_no_deletion(&self) -> Result<(), Vec<String>> {
        // 1. 現在の状態をスキャン
        // 2. ベースラインと比較
        // 3. 削除された項目があればエラー
        todo!("実装: 削除検出と報告")
    }

    /// 許可された変更のみチェック
    pub fn validate_changes(&self) -> RefactoringSummary {
        RefactoringSummary {
            added_items: vec![], // 新規追加項目
            modified_internals: vec![], // 内部実装変更
            forbidden_changes: vec![], // 禁止された変更
        }
    }
}

#[derive(Debug)]
pub struct RefactoringSummary {
    added_items: Vec<String>,
    modified_internals: Vec<String>,
    forbidden_changes: Vec<String>,
}
```

### Phase 3: Automated Protection / 自動保護システム

```bash
#!/bin/bash
# safe_refactor_guard.sh

echo "🛡️ Safe Refactoring Guard"
echo "========================"

# リファクタリング前
./safe_refactor_analysis.sh

echo ""
echo "⚠️  REFACTORING CONSTRAINTS:"
echo "   - NO deletion of existing public items"
echo "   - NO signature changes"
echo "   - Internal improvements ONLY"
echo "   - New features must not affect existing APIs"
echo ""
echo "Continue? (y/N)"
read -r response

if [[ "$response" != "y" ]]; then
    echo "❌ Refactoring cancelled"
    exit 1
fi

echo "✅ Starting protected refactoring session"
echo "📝 Guidelines:"
echo "   1. Keep all existing public functions/structs/traits"
echo "   2. Improve internal implementations only"
echo "   3. Add new features without breaking existing ones"
echo "   4. Run verification after changes"

# 作業後検証用のフック設定
trap 'echo "🔍 Running post-refactoring verification..."; ./verify_no_deletion.sh' EXIT
```

### Phase 4: Post-Refactoring Verification / リファクタリング後検証

```bash
#!/bin/bash
# verify_no_deletion.sh

echo "🔍 Post-Refactoring Verification"
echo "==============================="

# 1. 新しい状態をスキャン
echo "📋 Step 1: Scanning current state"
cargo expand | grep -E "pub (fn|struct|enum|trait|const|static|mod)" > api_current.txt

# 2. ベースラインと比較
echo "📋 Step 2: Comparing with baseline"
if ! diff -u api_baseline.txt api_current.txt > api_changes.diff; then
    echo "⚠️  API changes detected:"
    cat api_changes.diff

    # 削除をチェック
    if grep "^-" api_changes.diff | grep -v "^---"; then
        echo "🚨 CRITICAL: Existing items were DELETED!"
        echo "❌ The following items were removed:"
        grep "^-" api_changes.diff | grep -v "^---"
        echo ""
        echo "🔧 ACTION REQUIRED: Restore all deleted items immediately"
        exit 1
    fi

    # 追加のみかチェック
    if grep "^+" api_changes.diff | grep -v "^+++"; then
        echo "✅ New items added (acceptable):"
        grep "^+" api_changes.diff | grep -v "^+++"
    fi
else
    echo "✅ No API changes detected"
fi

# 3. 関数シグネチャの検証
echo "📋 Step 3: Verifying function signatures"
rg "pub fn [^{]*" --type rust -o > function_signatures_current.txt

if ! diff -q function_signatures.txt function_signatures_current.txt >/dev/null; then
    echo "⚠️  Function signature changes detected"
    diff -u function_signatures.txt function_signatures_current.txt

    # 削除された関数をチェック
    if comm -23 function_signatures.txt function_signatures_current.txt | grep -q .; then
        echo "🚨 CRITICAL: Functions were deleted or signatures changed!"
        echo "Missing functions:"
        comm -23 function_signatures.txt function_signatures_current.txt
        exit 1
    fi
fi

echo "✅ Verification complete - No existing items deleted"
```

## 🚀 Agent Usage / エージェント使用方法

### SlashCommand Integration / スラッシュコマンド統合

```bash
# /safe-refactor を実行した時の処理
/safe-refactor [target_file_or_module]

# 実行内容:
1. 既存項目の完全スキャン・保護
2. リファクタリング制約の確認
3. 安全な変更作業の実行
4. 変更後の自動検証
5. 削除項目があれば即座復元指示
```

### Example Safe Refactoring Session / 安全リファクタリングセッション例

```rust
// ❌ 危険なリファクタリング例 (Agent prevents this)
// pub fn old_function() -> i32 { ... } を削除 → 禁止

// ✅ 安全なリファクタリング例 (Agent allows this)

// Before: 内部実装が非効率
pub fn gpu_sum(&self, axis: Option<usize>) -> RusTorchResult<Self> {
    // 古い実装
    slow_implementation()
}

// After: 内部実装を改善、インターフェースは不変
pub fn gpu_sum(&self, axis: Option<usize>) -> RusTorchResult<Self> {
    // 新しい高速実装
    fast_gpu_implementation()
}

// ✅ 新機能追加も安全
pub fn gpu_sum_optimized(&self, axis: Option<usize>) -> RusTorchResult<Self> {
    // 既存のgpu_sum()に影響しない新機能
    self.gpu_sum(axis) // 内部で既存機能を活用
}
```

## 🛠️ Implementation Priority / 実装優先順位

### High Priority / 高優先度
1. **既存項目スキャン機能** - 削除対象の事前特定
2. **自動検証システム** - 変更後の完全性チェック
3. **復元指示システム** - 削除された場合の即座復元

### Medium Priority / 中優先度
1. **インタラクティブ保護** - 削除しようとした時点で警告
2. **依存関係追跡** - 既存項目の使用状況把握
3. **安全な変更提案** - 削除なしでの改善方法提示

## 📋 Agent Checklist / エージェントチェックリスト

### Before Starting / 開始前
- [ ] 既存パブリック項目の完全リスト作成
- [ ] 関数・メソッドシグネチャの記録
- [ ] 構造体・トレイト・列挙型の記録
- [ ] 依存関係マップの作成

### During Refactoring / リファクタリング中
- [ ] 削除操作の禁止
- [ ] シグネチャ変更の禁止
- [ ] 内部実装改善のみ実行
- [ ] 新機能は非破壊的に追加

### After Refactoring / リファクタリング後
- [ ] 既存項目の完全性検証
- [ ] シグネチャの変更なし確認
- [ ] 新機能の非干渉性確認
- [ ] 削除された項目があれば即座復元

このエージェントにより、リファクタリング時の既存項目削除を根本的に防げます。