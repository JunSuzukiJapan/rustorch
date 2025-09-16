# RusTorch WASM Jupyter Notebook ガイド

初心者向けに設計された、Jupyter Notebook で RusTorch WASM を簡単に使用するためのステップバイステップガイド。

## 📚 目次

1. [必要な環境](#必要な環境)
2. [セットアップ手順](#セットアップ手順)
3. [基本的な使い方](#基本的な使い方)
4. [実用的な例](#実用的な例)
5. [トラブルシューティング](#トラブルシューティング)
6. [よくある質問](#よくある質問)

## 必要な環境

### 最低要件
- **Python 3.8+**
- **Jupyter Notebook** または **Jupyter Lab**
- **Node.js 16+** (WASM ビルド用)
- **Rust** (最新安定版)
- **wasm-pack** (Rust コードを WASM に変換)

### 推奨環境
- メモリ: 8GB 以上
- ブラウザ: Chrome、Firefox、Safari の最新版
- OS: Windows 10/11、macOS 10.15+、Ubuntu 20.04+

## セットアップ手順

### 🚀 クイックスタート（推奨）

**最も簡単な方法**: 一つのコマンドで Jupyter Lab を起動
```bash
./start_jupyter.sh
```

このスクリプトは自動的に以下を実行します:
- 仮想環境の作成と有効化
- 依存関係のインストール (numpy, jupyter, matplotlib)
- RusTorch Python バインディングのビルド
- デモノートブックを開いた状態で Jupyter Lab を起動

### 手動セットアップ

#### ステップ 1: 基本ツールのインストール

```bash
# Python バージョンの確認
python --version

# Jupyter Lab のインストール
pip install jupyterlab

# Node.js のインストール (macOS with Homebrew)
brew install node

# Rust のインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# wasm-pack のインストール
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### ステップ 2: RusTorch WASM のビルド

```bash
# プロジェクトのクローン
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# WASM ターゲットの追加
rustup target add wasm32-unknown-unknown

# wasm-pack でビルド
wasm-pack build --target web --out-dir pkg
```

#### ステップ 3: Jupyter の起動

```bash
# Jupyter Lab の起動
jupyter lab
```

## 基本的な使い方

### テンソルの作成

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 1次元テンソル
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('1次元テンソル:', vec.to_array());

    // 2次元テンソル (行列)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // 形状: 2行3列
    );
    console.log('2次元テンソルの形状:', matrix.shape());
});
```

### 基本操作

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);

    // 加算
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());

    // 行列の乗算
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### 自動微分

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 勾配追跡付きテンソルの作成
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true

    // 計算: y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);

    // 逆伝播
    y.backward();

    // 勾配の取得 (dy/dx = 2x + 3 = 7 when x=2)
    console.log('勾配:', x.grad().to_array());
});
```

## 実用的な例

### 線形回帰

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // データの準備
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x

    // パラメータの初期化
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);

    const lr = 0.01;

    // 学習ループ
    for (let epoch = 0; epoch < 100; epoch++) {
        // 予測: y_pred = wx + b
        const y_pred = X.mul(w).add(b);

        // 損失: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();

        // 勾配の計算
        loss.backward();

        // パラメータの更新
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));

        // 勾配のリセット
        w.zero_grad();
        b.zero_grad();

        if (epoch % 10 === 0) {
            console.log(`エポック ${epoch}: 損失 = ${loss.item()}`);
        }
    }

    console.log(`最終 w: ${w.item()}, b: ${b.item()}`);
});
```

## トラブルシューティング

### 🚀 Rust カーネルの高速化（推奨）
初回実行が遅い場合、キャッシュを有効にすることで大幅な性能向上が可能です:

```bash
# キャッシュディレクトリの作成
mkdir -p ~/.config/evcxr

# 500MBキャッシュの有効化
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**効果:**
- 初回: 通常のコンパイル時間
- 以降の実行: 依存関係の再コンパイルなし（数倍高速）
- `rustorch` ライブラリも初回使用後にキャッシュされる

**注意:** ライブラリ更新後は `:clear_cache` でキャッシュを更新してください

### よくあるエラー

#### "RusTorch is not defined" エラー
**解決策**: 常に RusTorchReady を待つ
```javascript
window.RusTorchReady.then((rt) => {
    // ここで RusTorch を使用
});
```

#### "Failed to load WASM module" エラー
**解決策**:
1. `pkg` ディレクトリが正しく生成されているか確認
2. ブラウザコンソールでエラーメッセージを確認
3. WASM ファイルのパスが正しいか確認

#### メモリ不足エラー
**解決策**:
```javascript
// メモリを明示的に解放
tensor.free();

// より小さなバッチサイズを使用
const batchSize = 32;  // 1000 の代わりに 32 を使用
```

### パフォーマンスのヒント

1. **バッチ処理の使用**: ループの代わりにデータをバッチで処理
2. **メモリ管理**: 大きなテンソルは明示的に解放
3. **適切なデータ型**: 高精度が不要な場合は f32 を使用

## よくある質問

### Q: Google Colab で使用できますか？
**A**: はい、WASM ファイルをアップロードし、カスタム JavaScript ローダーを使用してください。

### Q: Python と WASM コードを混在できますか？
**A**: はい、IPython.display.Javascript を使用して Python と JavaScript 間でデータを受け渡しできます。

### Q: デバッグ方法は？
**A**: ブラウザの開発者ツール（F12）を使用し、コンソールタブでエラーを確認してください。

### Q: どのような高度な機能が利用できますか？
**A**: 現在は基本的なテンソル操作、自動微分、シンプルなニューラルネットワークをサポートしています。CNN と RNN レイヤーは予定されています。

## 次のステップ

1. 📖 [詳細な RusTorch WASM API](../wasm.md)
2. 🔬 [高度な例](../examples/)
3. 🚀 [パフォーマンス最適化ガイド](../wasm-memory-optimization.md)

## コミュニティとサポート

- GitHub: [RusTorch リポジトリ](https://github.com/JunSuzukiJapan/rustorch)
- Issues: GitHub でバグ報告と機能リクエストを行ってください

---

RusTorch WASM で楽しく学習しましょう！ 🦀🔥📓