# Git Push Workflow - 必須手順

## プッシュ前の必須チェックリスト

**すべてのプッシュ前に以下を必ず実行すること：**

### 1. Clippy チェック
```bash
cargo clippy --lib --no-default-features -- -D warnings
```
- 全ての警告がエラーとして扱われる設定でテスト
- CodeQL CI環境と同じ条件でチェック
- 警告が出た場合は修正してからプッシュ

### 2. フォーマット
```bash
cargo fmt
```
- 一貫したコードスタイルを適用
- インデント、改行、スペーシングを統一

### 3. コンパイル確認
```bash
cargo check --lib --no-default-features
```
- 基本的なコンパイルエラーがないことを確認

### 4. 実行順序
1. **Clippy** → 警告修正 → 再チェック
2. **Format** → スタイル統一
3. **Compile** → 最終確認
4. **Git add + commit + push**

## 重要なポイント

- **CodeQL CI失敗の主要原因**: clippy警告の見落とし
- **フォーマット**: コードレビューを円滑にするため必須
- **段階的実行**: clippy → fmt → check の順序を守る

## テンプレートコマンド

```bash
# 1. Clippy チェック
cargo clippy --lib --no-default-features -- -D warnings

# 2. フォーマット適用
cargo fmt

# 3. 最終コンパイル確認
cargo check --lib --no-default-features

# 4. Git操作
git add .
git commit -m "適切なコミットメッセージ"
git push
```

この手順を守ることでCodeQL CI通過率100%を維持する。

## 注意事項

- 必ずこの順序で実行する
- 各ステップでエラーが出たら修正してから次に進む
- 急いでいても省略しない
- CI失敗を防ぐための必須プロセス