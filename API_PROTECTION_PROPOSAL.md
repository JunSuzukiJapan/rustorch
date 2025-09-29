# API Protection Implementation Proposal
# API保護実装提案

## 🎯 目的
RusTorchプロジェクトでのAPI消失問題を根本的に解決し、将来の破壊的変更を防止する包括的システムを構築する。

## 🔍 問題分析結果

### 発見された根本原因
1. **API破壊検知メカニズムの不在**: パブリックAPI変更の自動検知なし
2. **リファクタリング時の無計画性**: 既存API保持の考慮なし
3. **後方互換性ポリシーの欠如**: API変更時のガイドライン不足
4. **テストでのAPI保護不足**: 既存API継続性のテストなし

### 具体的な問題事例
- `aa204b1b6`: hybrid_f32全面書き換えで`gpu_sum()`, `__add__()`等が消失
- `5fdb99c6e`: 品質改善でunused関数として誤削除
- リファクタリング計画に後方互換性の考慮なし

## 🛡️ 提案する解決策

### Phase 1: 即座実装可能な保護策

#### 1.1 自動API互換性チェッカー
```bash
#!/bin/bash
# scripts/check_api_compatibility.sh

echo "🔍 API互換性チェック開始..."

# 現在のパブリックAPIを抽出
cargo public-api --simplified > api_current.txt

# 前バージョンとの差分チェック
if [ -f "api_baseline.txt" ]; then
    echo "📊 API変更の検出:"
    if diff -u api_baseline.txt api_current.txt > api_changes.diff; then
        echo "✅ API変更なし"
    else
        echo "⚠️  API変更を検出:"
        cat api_changes.diff

        # 破壊的変更の自動検知
        if grep -E "^-.*pub (fn|struct|enum|trait)" api_changes.diff; then
            echo "🚨 破壊的変更を検出！レビューが必要です"
            exit 1
        fi
    fi
fi

# 成功時は新しいベースラインを保存
cp api_current.txt api_baseline.txt
echo "✅ API互換性チェック完了"
```

#### 1.2 包括的API保護テスト
```rust
// tests/api_protection_tests.rs
//! 既存APIの保護とシグネチャ検証

use rustorch::hybrid_f32::tensor::F32Tensor;

/// hybrid_f32 GPU操作APIの存在確認
#[test]
fn test_hybrid_f32_gpu_apis_exist() {
    let tensor = F32Tensor::zeros(&[2, 2]).unwrap();

    // GPU演算APIの存在確認
    assert!(has_method!(tensor, "gpu_sum"));
    assert!(has_method!(tensor, "gpu_mean"));
    assert!(has_method!(tensor, "gpu_min"));
    assert!(has_method!(tensor, "gpu_max"));
    assert!(has_method!(tensor, "gpu_std"));
    assert!(has_method!(tensor, "gpu_var"));

    // Python互換APIの存在確認
    assert!(has_method!(tensor, "__add__"));
    assert!(has_method!(tensor, "__mul__"));
}

/// APIシグネチャの破壊的変更検知
#[test]
fn test_api_signatures_unchanged() {
    verify_method_signature!(
        F32Tensor::gpu_sum,
        "(&self, axis: Option<usize>) -> RusTorchResult<Self>"
    );

    verify_method_signature!(
        F32Tensor::__add__,
        "(&self, other: &Self) -> RusTorchResult<Self>"
    );
}

/// 古いAPIの非推奨化テスト
#[test]
fn test_deprecated_apis_still_work() {
    // 非推奨でも動作することを確認
    let tensor = F32Tensor::zeros(&[2, 2]).unwrap();

    // 将来非推奨予定のAPIでも現在は動作
    let result = tensor.gpu_sum(None);
    assert!(result.is_ok());
}

// テスト用マクロ
macro_rules! has_method {
    ($obj:expr, $method:literal) => {
        // コンパイル時にメソッド存在を確認
        std::mem::forget(|| { $obj.$method });
        true
    };
}

macro_rules! verify_method_signature {
    ($method:path, $expected:literal) => {
        // シグネチャ情報をコンパイル時に検証
        // 実装詳細は省略
    };
}
```

#### 1.3 強化された開発ワークフロー
```bash
#!/bin/bash
# scripts/enhanced_pre_push.sh
# 強化されたプッシュ前チェック

echo "🚀 RusTorch Enhanced Pre-Push Validation"
echo "========================================"

# 1. API互換性チェック
echo "📋 Step 1: API Compatibility Check"
./scripts/check_api_compatibility.sh || exit 1

# 2. API保護テスト
echo "📋 Step 2: API Protection Tests"
cargo test api_protection_tests --quiet || {
    echo "🚨 API保護テストが失敗しました"
    exit 1
}

# 3. 従来の品質チェック
echo "📋 Step 3: Code Quality Checks"
cargo clippy --lib --no-default-features -- -D warnings || exit 1
cargo fmt || exit 1
cargo check --lib --no-default-features || exit 1

# 4. 包括テスト実行
echo "📋 Step 4: Comprehensive Testing"
cargo test --features hybrid-f32 --quiet || exit 1

echo "✅ All checks passed! Ready to push."
```

### Phase 2: 長期的な改善策

#### 2.1 API管理ポリシーの策定
```markdown
# API Management Policy v1.0

## セマンティックバージョニング
- **MAJOR (X.0.0)**: 破壊的変更（API削除、シグネチャ変更）
- **MINOR (0.X.0)**: 後方互換の機能追加
- **PATCH (0.0.X)**: バグ修正のみ

## API変更プロセス
1. **新機能**: MINORバージョンで追加
2. **非推奨化**: `#[deprecated]`でマーク、1バージョン継続
3. **削除**: 次のMAJORバージョンでのみ許可

## 例外的削除条件
- クリティカルセキュリティ問題
- ライセンス法的問題
- データ破損を引き起こす明らかなバグ
```

#### 2.2 リファクタリングガイドライン
```markdown
# Refactoring Guidelines

## 必須プロセス
1. **事前調査**: 既存パブリックAPIの完全リスト化
2. **影響評価**: 各APIの使用状況と重要度評価
3. **移行設計**: 既存API保持またはスムーズな移行パス
4. **テスト保護**: API保護テストの事前作成
5. **段階実装**: 非破壊的な段階的移行

## 禁止事項
- パブリックAPIの予告なし削除
- シグネチャの破壊的変更（MAJOR以外）
- テストされていない大規模変更
- ドキュメント更新なしのAPI変更
```

#### 2.3 継続的なAPI健全性監視
```rust
// scripts/api_health_monitor.rs
//! 定期的なAPI健全性チェック

use std::collections::HashMap;

#[derive(Debug)]
struct ApiHealthReport {
    total_apis: usize,
    deprecated_apis: usize,
    unused_apis: usize,
    compatibility_issues: Vec<String>,
}

impl ApiHealthReport {
    fn generate() -> Self {
        // 1. パブリックAPI一覧の取得
        // 2. 非推奨API数の計算
        // 3. 未使用API検出
        // 4. 互換性問題の特定

        Self {
            total_apis: 0,
            deprecated_apis: 0,
            unused_apis: 0,
            compatibility_issues: vec![],
        }
    }

    fn print_summary(&self) {
        println!("📊 API Health Report");
        println!("==================");
        println!("Total APIs: {}", self.total_apis);
        println!("Deprecated: {}", self.deprecated_apis);
        println!("Unused: {}", self.unused_apis);

        if !self.compatibility_issues.is_empty() {
            println!("⚠️  Issues found:");
            for issue in &self.compatibility_issues {
                println!("  - {}", issue);
            }
        } else {
            println!("✅ No compatibility issues detected");
        }
    }
}

fn main() {
    let report = ApiHealthReport::generate();
    report.print_summary();
}
```

## 🎯 実装タイムライン

### Week 1: 緊急対策
- [ ] API互換性チェッカーの実装
- [ ] API保護テストスイートの作成
- [ ] 強化された開発ワークフローの導入

### Week 2-3: ポリシー策定
- [ ] API管理ポリシーの文書化
- [ ] リファクタリングガイドラインの作成
- [ ] チーム内でのプロセス教育

### Week 4: 自動化・監視
- [ ] CI/CDパイプラインにAPI保護を統合
- [ ] 定期的なAPI健全性監視の設定
- [ ] レポート・アラートシステムの構築

## 📈 期待される効果

### 短期効果（1-2週間）
- API消失の即座防止
- 破壊的変更の事前検知
- 開発者の意識向上

### 中期効果（1-3ヶ月）
- 安定したパブリックAPI
- ユーザーからの信頼向上
- 後方互換性の確保

### 長期効果（6ヶ月以上）
- 成熟したAPI管理文化
- 予測可能なリリースサイクル
- エコシステムの健全な発展

## 🚨 リスク軽減策

### 実装リスク
- **過度な制約**: 開発速度の低下 → 柔軟性を持った例外プロセス
- **誤検知**: 正当な変更の阻害 → 人間による最終判断プロセス
- **学習コスト**: 新プロセスの習得 → 段階的導入と教育

### 運用リスク
- **メンテナンス負荷**: 追加のツール管理 → 自動化による軽減
- **誤用**: ポリシーの誤解 → 明確なドキュメントと例示

この提案により、RusTorchプロジェクトでのAPI消失問題を根本的に解決し、持続可能な開発プロセスを確立できます。