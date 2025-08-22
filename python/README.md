# RusTorch Python Bindings

PyO3を使用したRusTorchのPythonバインディングです。

## macOSでのビルド

### Framework Python (Homebrew/Python.org) を使用している場合

macOSでHomebrew (`brew install python`) またはPython.orgからインストールしたPythonを使用している場合、以下の設定が必要です：

#### 1. 使用しているPythonの種類を確認

```bash
python3 -c "import sysconfig; print('Framework:', sysconfig.get_config_var('PYTHONFRAMEWORK'))"
```

`Framework: Python` と表示される場合は、Framework Pythonを使用しています。

#### 2. `.cargo/config.toml` ファイルの作成

プロジェクトルートに以下の内容で `.cargo/config.toml` を作成してください：

```toml
[build]
rustflags = [
    "-C", "link-arg=-undefined",
    "-C", "link-arg=dynamic_lookup",
]

[env]
PYO3_PYTHON = "/usr/local/bin/python3.9"  # あなたのPythonパス
```

#### 3. ビルド

```bash
cargo clean
cargo build --release
```

#### 4. 拡張モジュールのコピー

```bash
cp target/release/lib_rustorch_py.dylib _rustorch_py.so
```

### よくあるエラーと解決方法

#### エラー: `symbol(s) not found for architecture x86_64`

**原因:** Framework PythonでPyO3が正しくリンクできていない

**解決方法:** 上記の `.cargo/config.toml` 設定を追加

#### エラー: `Python interpreter version (3.6) is lower than PyO3's minimum`

**原因:** デフォルトのPythonバージョンが古い

**解決方法:** 
```bash
export PYO3_PYTHON=/usr/local/bin/python3.9  # 新しいPythonを指定
cargo build --release
```

### 代替手段

Framework Pythonで問題が続く場合は、以下の代替手段があります：

1. **pyenv を使用:**
   ```bash
   brew install pyenv
   pyenv install 3.9.18
   pyenv local 3.9.18
   export PYO3_PYTHON=$(pyenv which python)
   ```

2. **conda/miniconda を使用:**
   ```bash
   conda create -n rustorch python=3.9
   conda activate rustorch
   export PYO3_PYTHON=$(which python)
   ```

3. **システムPython を使用 (利用可能な場合):**
   ```bash
   export PYO3_PYTHON=/usr/bin/python3
   ```

## テスト

```bash
python3 test_minimal.py
```

成功すると以下のような出力が表示されます：

```
Testing Python->Rust communication...
✓ Found library files: ['target/release/deps/lib_rustorch_py.dylib']
✓ Successfully imported _rustorch_py
✓ hello_from_rust() = Hello from RusTorch!
✓ get_version() = 0.3.3
✓ add_numbers(1.5, 2.5) = 4.0
```

## 参考リンク

- [PyO3 User Guide - Building and Distribution](https://pyo3.rs/v0.20.3/building_and_distribution)
- [PyO3 FAQ - macOS](https://pyo3.rs/v0.20.3/faq#i-cant-run-cargo-test-im-having-linker-issues-like-symbol-not-found-or-undefined-reference-to)