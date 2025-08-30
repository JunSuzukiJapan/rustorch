# Pre-publish checklist

## 1.ビルドとテスト
```bash
cargo build --all-targets --no-default-features"
cargo build --all-targets --features "linalg-netlib"
cargo test --all-targets --features "linalg-netlib"
```

## 2. WASMのテスト

すべてのエラーと警告をなくして。

```bash
cargo test --all-targets --features "linalg-netlib,wasm" --target wasm32-unknown-unknown
```

## 3. ドックテストの実行
```bash
cargo doc --test
```

## 4. ドキュメントの生成
```bash
cargo doc --all-targets --features "linalg-netlib"
```

## 5. ベンチマークの実行
```bash
cargo bench --all-targets --features "linalg-netlib"
```

## 6. examplesの実行
```bash
cargo run --example autograd_demo
```

## 7. clippyの実行
```bash
cargo clippy --all-targets --features "linalg-netlib" -D warnings
```

## 8. フォーマットの実行
```bash
cargo fmt --all --check
```

## 9. ビルドの実行
```bash
cargo build --all-targets --features "linalg-netlib"
```

## 10. WASMのビルド
```bash
cargo build --all-targets --features "linalg-netlib,wasm" --target wasm32-unknown-unknown
```

<!-- ## 11. プッシュ
```bash
git add -A
git commit -m "chore: Prepare for release"
git push origin $(git branch --show-current)
```

## 11. リリースの作成
```bash
cargo release -- --no-verify
```

## 12. リリースの確認
```bash
cargo release -- --no-verify
```

## 13. リリースのプッシュ
```bash
git push origin $(git branch --show-current)
git push origin $(git branch --show-current) --tags
``` -->



