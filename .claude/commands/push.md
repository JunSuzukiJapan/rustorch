# Push コマンド

## 基本的な使用方法

### 1. 自動修正（clippy + format + push）
```bash
# すべての修正と整形を自動で実行
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && \
cargo fmt --all && \
git add -A && \
git commit -m "style: Apply clippy fixes and formatting" && \
git push origin $(git branch --show-current)
```

### 2. 手動での段階的実行
```bash
# Step 1: Clippy による修正
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged

# Step 2: フォーマット
cargo fmt --all

# Step 3: 確認
git diff
git status

# Step 4: コミット
git add -A
git commit -m "style: Apply clippy fixes and formatting

- Resolve clippy suggestions
- Apply rustfmt formatting
- Ensure code quality standards

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Step 5: プッシュ
git push origin $(git branch --show-current)
```

## プラットフォーム別の実行方法

### Ubuntu/Linux
```bash
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged
cargo fmt --all
```

### macOS
```bash
cargo clippy --all-targets --features "linalg-netlib" --target x86_64-apple-darwin --fix --allow-dirty --allow-staged
cargo fmt --all
```

### Windows
```bash
cargo clippy --all-targets --no-default-features --fix --allow-dirty --allow-staged
cargo fmt --all
```

## テストを含むプッシュ

```bash
# すべての検証を実行してからプッシュ
cargo clippy --all-targets --features "linalg-netlib" -- -D warnings && \
cargo fmt --all --check && \
cargo test --features "linalg-netlib" && \
git add -A && \
git commit -m "chore: Code quality improvements" && \
git push origin $(git branch --show-current)
```

## エイリアスの設定

`.bashrc` または `.zshrc` に追加:

```bash
alias rustpush='cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged && cargo fmt --all && git add -A && git commit -m "style: Apply clippy fixes and formatting" && git push origin $(git branch --show-current)'
```

使用例:
```bash
rustpush
```

## トラブルシューティング

### Clippy の警告を確認したい場合
```bash
# 修正前に確認
cargo clippy --all-targets --features "linalg-netlib"

# 警告をエラーとして扱う
cargo clippy --all-targets --features "linalg-netlib" --fix --allow-dirty --allow-staged
```

### フォーマットのみ確認
```bash
# フォーマットの差分確認
cargo fmt --all --check
```

### 特定のファイルのみ処理
```bash
# 特定ファイルのみフォーマット
cargo fmt -- src/lib.rs src/tensor/mod.rs

# 特定ファイルのみコミット
git add src/lib.rs src/tensor/mod.rs
git commit -m "style: Format specific modules"
git push origin $(git branch --show-current)
```