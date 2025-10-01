# RusTorch Release Workflow

## リリース前の準備

### 1. mainブランチを最新に更新
```bash
git checkout main
git pull origin main
```

### 2. リリースブランチ作成
```bash
# バージョン番号は適宜変更
git checkout -b release/v0.6.28
```

## バージョン更新

### 3. バージョン番号の更新
以下のファイルを更新：

- `Cargo.toml` (version = "0.6.28")
- `README.md` (インストール例)
- `Cargo.lock` (`cargo update -p rustorch`で自動更新)
- Jupyter notebooks (8ファイル)
  - `notebooks/rustorch_rust_kernel_demo_ja.ipynb`
  - `notebooks/rustorch_rust_kernel_demo.ipynb`
  - `notebooks/en/rustorch_rust_kernel_demo_en.ipynb`
  - `notebooks/es/rustorch_rust_kernel_demo_es.ipynb`
  - `notebooks/fr/rustorch_rust_kernel_demo_fr.ipynb`
  - `notebooks/it/rustorch_rust_kernel_demo_it.ipynb`
  - `notebooks/ko/rustorch_rust_kernel_demo_ko.ipynb`
  - `notebooks/zh/rustorch_rust_kernel_demo_zh.ipynb`

### 4. 変更をコミット
```bash
git add -A
git commit -m "chore: bump version to 0.6.28"
```

## Pre-publish チェックリスト

### 5. テスト実行
```bash
# 基本テスト
cargo test --lib --no-default-features
cargo test --lib --features metal
cargo test --lib --features coreml
cargo test --lib --features "metal coreml"

# サンプル実行
cargo run --example readme_basic_usage_demo

# Doctests
cargo test --doc

# コード品質
cargo clippy --all-targets --no-default-features -- -W clippy::all
cargo fmt --all -- --check

# ドキュメント生成
cargo doc --no-deps --no-default-features

# Docker ビルド
docker build -f docker/Dockerfile .
```

### 6. 変更をプッシュ
```bash
git push origin release/v0.6.28
```

## プルリクエスト作成

### 7. PRを作成
```bash
gh pr create \
  --title "Release v0.6.28: [主な変更内容]" \
  --body "$(cat <<'EOF'
## Summary
[リリース内容の要約]

## Changes
- [変更点1]
- [変更点2]

## Testing
- ✅ Tests passed
- ✅ Examples verified
- ✅ Docker build successful

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 8. CI/CDの完了を待つ
すべてのチェックがパスするまで待機

## マージと公開

### 9. PRをマージ
```bash
gh pr merge [PR番号] --squash \
  --subject "Release v0.6.28: [主な変更内容]" \
  --body "[リリースノート]"
```

### 10. mainブランチに切り替え
```bash
git checkout main
git pull origin main
```

### 11. Gitタグの作成
```bash
git tag v0.6.28
git push origin v0.6.28
```

### 12. GitHubリリース作成
```bash
gh release create v0.6.28 \
  --title "RusTorch v0.6.28 - [タイトル]" \
  --notes "[リリースノート]"
```

### 13. crates.io公開
```bash
cargo publish
```

## クリーンアップ

### 14. リリースブランチの削除
```bash
# ローカル
git branch -d release/v0.6.28

# リモート
git push origin --delete release/v0.6.28
```

## コンフリクト防止のポイント

### ✅ 推奨事項

1. **常にmainから最新をpull**: リリースブランチ作成前に必ず実行
2. **リリースブランチの削除**: マージ後は即座に削除
3. **cargo fmtの実行**: コミット前に必ず実行
4. **PRマージ方式**: 常にsquashマージを使用

### ❌ 避けるべき行動

1. mainブランチへの直接コミット
2. マージ済みリリースブランチの再利用
3. フォーマットチェックなしでのコミット
4. 複数のリリースブランチを同時に作成

## トラブルシューティング

### コンフリクトが発生した場合

```bash
# mainから最新を取得
git checkout main
git pull origin main

# リリースブランチにrebase
git checkout release/v0.6.28
git rebase main

# コンフリクト解決後
git add .
git rebase --continue
git push origin release/v0.6.28 --force-with-lease
```

### CI/CDが失敗した場合

1. ローカルで該当テストを実行
2. 問題を修正
3. `cargo fmt --all`を実行
4. コミット＆プッシュ

## 自動化スクリプト（オプション）

将来的に`scripts/release.sh`を作成して自動化することを推奨：

```bash
#!/bin/bash
# 引数: ./scripts/release.sh 0.6.28 "主な変更内容"

VERSION=$1
TITLE=$2

# バージョン更新、テスト、PR作成、マージ、タグ作成、公開を自動化
```

---

このワークフローに従うことで、次回以降のリリースでコンフリクトを最小限に抑えることができます。
