# RusTorch WASM Jupyter Notebook ガイド

初心者でも簡単にJupyter NotebookでRusTorchのWASM版を使えるようになるためのステップバイステップガイドです。

## 📚 目次

1. [必要な環境](#必要な環境)
2. [セットアップ手順](#セットアップ手順)
3. [基本的な使い方](#基本的な使い方)
4. [実践例](#実践例)
5. [トラブルシューティング](#トラブルシューティング)
6. [よくある質問](#よくある質問)

## 必要な環境

### 最低限必要なもの
- **Python 3.8以上**
- **Jupyter Notebook** または **Jupyter Lab**
- **Node.js 16以上**（WASMビルド用）
- **Rust**（最新の安定版）
- **wasm-pack**（RustコードをWASMに変換）

### 推奨環境
- メモリ: 8GB以上
- ブラウザ: Chrome、Firefox、Safari の最新版
- OS: Windows 10/11、macOS 10.15以上、Ubuntu 20.04以上

## セットアップ手順

### 🚀 クイックスタート（推奨）

**最も簡単な方法**: 1つのコマンドでJupyter Labを起動
```bash
./start_jupyter.sh
```

このスクリプトは以下を自動実行します：
- 仮想環境の作成・アクティベート
- 依存関係のインストール（numpy, jupyter, matplotlib）
- RusTorch Pythonバインディングのビルド
- Jupyter Lab起動とデモノートブック表示

### ステップ1: 基本ツールのインストール（手動セットアップの場合）

#### 1.1 Pythonとpipの確認
```bash
# Pythonバージョンの確認
python --version
# または
python3 --version

# pipの確認
pip --version
# または
pip3 --version
```

#### 1.2 Jupyter Notebookのインストール
```bash
# Jupyter Notebookをインストール
pip install notebook

# または、Jupyter Lab（より高機能版）をインストール
pip install jupyterlab
```

#### 1.3 Node.jsのインストール
```bash
# macOSの場合（Homebrewを使用）
brew install node

# Windowsの場合
# https://nodejs.org からダウンロードしてインストール

# Ubuntuの場合
sudo apt update
sudo apt install nodejs npm

# バージョン確認
node --version
npm --version
```

#### 1.4 Rustのインストール
```bash
# Rustupをインストール（公式推奨方法）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# インストール後、パスを通す
source $HOME/.cargo/env

# バージョン確認
rustc --version
cargo --version
```

#### 1.5 wasm-packのインストール
```bash
# wasm-packをインストール
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# または、cargoを使用
cargo install wasm-pack

# バージョン確認
wasm-pack --version
```

### ステップ2: RusTorchプロジェクトの準備

#### 2.1 RusTorchのクローン
```bash
# プロジェクトディレクトリを作成
mkdir ~/rustorch-jupyter
cd ~/rustorch-jupyter

# RusTorchをクローン
git clone https://github.com/yourusername/rustorch.git
cd rustorch
```

#### 2.2 WASM用にビルド
```bash
# WASMターゲットを追加
rustup target add wasm32-unknown-unknown

# wasm-packでビルド
wasm-pack build --target web --out-dir pkg

# ビルドが成功すると、pkgディレクトリが作成されます
ls pkg/
# 出力例: rustorch.js  rustorch_bg.wasm  package.json  など
```

### ステップ3: Jupyter環境の設定

#### 3.1 Jupyterカーネル拡張の作成

`jupyter_setup.py`という新しいファイルを作成:

```python
# jupyter_setup.py
import os
import shutil
from pathlib import Path

def setup_jupyter_wasm():
    """Jupyter用のWASM環境をセットアップ"""
    
    # 1. Jupyter設定ディレクトリを確認
    jupyter_dir = Path.home() / '.jupyter'
    jupyter_dir.mkdir(exist_ok=True)
    
    # 2. カスタムディレクトリを作成
    custom_dir = jupyter_dir / 'custom'
    custom_dir.mkdir(exist_ok=True)
    
    # 3. custom.jsファイルを作成
    custom_js = custom_dir / 'custom.js'
    
    js_content = """
// RusTorch WASM自動ロード設定
require.config({
    paths: {
        'rustorch': '/files/rustorch/pkg/rustorch'
    }
});

// グローバル変数として利用可能にする
window.RusTorchReady = new Promise((resolve, reject) => {
    require(['rustorch'], function(rustorch) {
        rustorch.default().then(() => {
            window.RusTorch = rustorch;
            console.log('✅ RusTorch WASM loaded successfully!');
            resolve(rustorch);
        }).catch(reject);
    });
});
"""
    
    # 4. ファイルに書き込み
    with open(custom_js, 'w') as f:
        f.write(js_content)
    
    print(f"✅ Jupyter設定が完了しました: {custom_js}")
    
    # 5. シンボリックリンクを作成（開発用）
    notebook_dir = Path.home() / 'rustorch'
    if not notebook_dir.exists():
        current_dir = Path.cwd()
        notebook_dir.symlink_to(current_dir)
        print(f"✅ シンボリックリンクを作成: {notebook_dir} -> {current_dir}")

if __name__ == "__main__":
    setup_jupyter_wasm()
```

セットアップを実行:
```bash
python jupyter_setup.py
```

#### 3.2 Jupyterサーバーの起動

```bash
# Jupyter Notebookを起動
jupyter notebook

# または、Jupyter Labを起動
jupyter lab
```

### ステップ4: 動作確認

#### 4.1 ブラウザでJupyterを開く

Jupyterサーバーを起動すると、自動的にブラウザが開きます。開かない場合は：

1. **ターミナルに表示されたURLをコピー**
   ```
   [I 12:34:56.789 NotebookApp] Serving notebooks from local directory: /Users/username/rustorch-jupyter
   [I 12:34:56.789 NotebookApp] Jupyter Notebook 6.4.12 is running at:
   [I 12:34:56.789 NotebookApp] Local URL: http://localhost:8888/?token=abc123...
   ```

2. **ブラウザで手動で開く**
   - URLをコピーしてブラウザのアドレスバーに貼り付け
   - または `http://localhost:8888` にアクセスしてトークンを入力

3. **推奨ブラウザ**
   - **Chrome**: WASMとWebGPUの最高サポート ✅
   - **Firefox**: 安定したWASMサポート ✅
   - **Safari**: 基本的なWASMサポート ⚠️
   - **Edge**: Chromiumベースで良好なサポート ✅

#### 4.2 新しいNotebookを作成

1. Jupyterのブラウザ画面で「New」→「Python 3」をクリック
2. 新しいNotebookが開きます

#### 4.3 RusTorch WASMの初期化

最初のセルに以下のコードを入力して実行（Shift + Enter）:

```javascript
%%javascript
// RusTorch WASMが読み込まれるまで待つ
window.RusTorchReady.then((rustorch) => {
    console.log('RusTorch is ready!');
    
    // バージョン確認
    const version = rustorch.get_version();
    console.log(`RusTorch version: ${version}`);
    
    // 簡単なテスト
    const tensor = rustorch.create_tensor([1, 2, 3, 4], [2, 2]);
    console.log('Created tensor:', tensor);
});
```

## 基本的な使い方

### テンソルの作成

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 1次元テンソル
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('1D Tensor:', vec.to_array());
    
    // 2次元テンソル（行列）
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // shape: 2行3列
    );
    console.log('2D Tensor shape:', matrix.shape());
});
```

### 基本演算

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // テンソルの作成
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // 加算
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // 乗算
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
    
    // 転置
    const transposed = a.transpose();
    console.log('A^T =', transposed.to_array());
});
```

### 自動微分

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 勾配追跡を有効にしてテンソルを作成
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // 計算: y = x^2 + 3x + 1
    const x_squared = x.mul(x);
    const three_x = x.mul_scalar(3.0);
    const y = x_squared.add(three_x).add_scalar(1.0);
    
    // 逆伝播
    y.backward();
    
    // 勾配を取得（dy/dx = 2x + 3 = 7 when x=2）
    console.log('Gradient:', x.grad().to_array());
});
```

## 実践例

### 例1: 線形回帰

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // データの準備
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // パラメータの初期化
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    // 学習率
    const lr = 0.01;
    
    // 訓練ループ
    for (let epoch = 0; epoch < 100; epoch++) {
        // 予測: y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // 損失: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // 勾配を計算
        loss.backward();
        
        // パラメータ更新
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // 勾配をリセット
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${loss.item()}`);
        }
    }
    
    console.log(`Final w: ${w.item()}, b: ${b.item()}`);
});
```

### 例2: ニューラルネットワーク

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // 簡単な2層ニューラルネットワーク
    class SimpleNN {
        constructor(inputSize, hiddenSize, outputSize) {
            // 重みの初期化（Xavier初期化）
            const scale1 = Math.sqrt(2.0 / inputSize);
            const scale2 = Math.sqrt(2.0 / hiddenSize);
            
            this.W1 = rt.randn([inputSize, hiddenSize]).mul_scalar(scale1);
            this.b1 = rt.zeros([hiddenSize]);
            this.W2 = rt.randn([hiddenSize, outputSize]).mul_scalar(scale2);
            this.b2 = rt.zeros([outputSize]);
            
            // 勾配追跡を有効化
            this.W1.requires_grad_(true);
            this.b1.requires_grad_(true);
            this.W2.requires_grad_(true);
            this.b2.requires_grad_(true);
        }
        
        forward(x) {
            // 第1層: ReLU活性化
            let h = x.matmul(this.W1).add(this.b1);
            h = h.relu();
            
            // 第2層: 線形
            const output = h.matmul(this.W2).add(this.b2);
            return output;
        }
    }
    
    // モデルの作成
    const model = new SimpleNN(2, 4, 1);
    
    // XORデータセット
    const X = rt.create_tensor([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ], [4, 2]);
    
    const y = rt.create_tensor([0, 1, 1, 0], [4, 1]);
    
    // 訓練
    const lr = 0.1;
    for (let epoch = 0; epoch < 1000; epoch++) {
        // 順伝播
        const output = model.forward(X);
        
        // 損失計算
        const loss = output.sub(y).pow(2).mean();
        
        // 逆伝播
        loss.backward();
        
        // パラメータ更新
        model.W1 = model.W1.sub(model.W1.grad().mul_scalar(lr));
        model.b1 = model.b1.sub(model.b1.grad().mul_scalar(lr));
        model.W2 = model.W2.sub(model.W2.grad().mul_scalar(lr));
        model.b2 = model.b2.sub(model.b2.grad().mul_scalar(lr));
        
        // 勾配リセット
        model.W1.zero_grad();
        model.b1.zero_grad();
        model.W2.zero_grad();
        model.b2.zero_grad();
        
        if (epoch % 100 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${loss.item()}`);
        }
    }
    
    // テスト
    const predictions = model.forward(X);
    console.log('Predictions:', predictions.to_array());
});
```

### 例3: データ可視化との統合

```python
# Pythonセル: matplotlibでの可視化
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML, Javascript

# JavaScriptでデータを生成してPythonに渡す
display(Javascript("""
window.RusTorchReady.then((rt) => {
    // データ生成
    const x = rt.linspace(-5, 5, 100);
    const y = x.mul(x);  // y = x^2
    
    // Pythonに渡すためにJSONに変換
    const data = {
        x: x.to_array(),
        y: y.to_array()
    };
    
    // IPython.kernelを使ってPythonに送信
    IPython.notebook.kernel.execute(
        `plot_data = ${JSON.stringify(data)}`
    );
});
"""))
```

```python
# 次のセル: データをプロット
import json
import time

# JavaScriptからデータが来るまで少し待つ
time.sleep(1)

# データをプロット
if 'plot_data' in globals():
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['x'], plot_data['y'])
    plt.title('y = x²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
```

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. "RusTorch is not defined"エラー

**原因**: WASMモジュールがまだ読み込まれていない

**解決方法**:
```javascript
// 必ずRusTorchReadyを待つ
window.RusTorchReady.then((rt) => {
    // ここでRusTorchを使用
});
```

#### 2. "Failed to load WASM module"エラー

**原因**: WASMファイルのパスが間違っている

**解決方法**:
1. `pkg`ディレクトリが正しく生成されているか確認
2. Jupyterのファイルブラウザで`pkg/rustorch_bg.wasm`が見えるか確認
3. ブラウザのコンソールでエラーメッセージを確認

#### 3. メモリ不足エラー

**原因**: 大きなテンソルを作成しようとしている

**解決方法**:
```javascript
// メモリを解放
tensor.free();  // 不要になったテンソルを明示的に解放

// または、小さいバッチサイズを使用
const batchSize = 32;  // 1000ではなく32に
```

#### 4. 勾配が計算されない

**原因**: `requires_grad`が設定されていない

**解決方法**:
```javascript
// テンソル作成時に指定
const x = rt.create_tensor([1, 2, 3], null, true);  // requires_grad=true

// または後から設定
x.requires_grad_(true);
```

### パフォーマンス最適化のヒント

#### 1. バッチ処理を活用
```javascript
// 悪い例: ループで個別に処理
for (let i = 0; i < 1000; i++) {
    const result = tensor.mul_scalar(2.0);
}

// 良い例: ベクトル化された操作
const batch = rt.create_tensor(data, [1000, 10]);
const result = batch.mul_scalar(2.0);  // 一度に全て処理
```

#### 2. メモリ管理
```javascript
// 大きな計算の後はガベージコレクションを促す
if (typeof gc !== 'undefined') {
    gc();
}

// 明示的にテンソルを解放
largeTensor.free();
```

#### 3. 適切なデータ型を使用
```javascript
// 精度が不要な場合はf32を使用
const tensor_f32 = rt.create_tensor_f32(data);

// 高精度が必要な場合のみf64を使用
const tensor_f64 = rt.create_tensor_f64(data);
```

## よくある質問

### Q1: Google ColabやKaggle Notebookでも使えますか？

**A**: はい、使えます。ただし、以下の手順が必要です：

1. WASMファイルをアップロード
2. カスタムJavaScriptローダーを設定
3. CORSの制限に注意

詳細な手順:
```python
# Google Colab用のセットアップ
from google.colab import files
import os

# WASMファイルをアップロード
uploaded = files.upload()  # rustorch_bg.wasmとrustorch.jsを選択

# HTMLとJavaScriptを表示
from IPython.display import HTML

HTML("""
<script type="module">
    import init, * as rustorch from './rustorch.js';
    
    await init();
    window.RusTorch = rustorch;
    console.log('RusTorch loaded in Colab!');
</script>
""")
```

### Q2: PythonコードとWASMコードを混在させられますか？

**A**: はい、可能です。以下の方法があります：

```python
# Python側でデータを準備
import numpy as np
data = np.random.randn(100, 10).tolist()

# JavaScriptに渡す
from IPython.display import Javascript
Javascript(f"""
window.pythonData = {data};
window.RusTorchReady.then((rt) => {{
    const tensor = rt.create_tensor(window.pythonData, [100, 10]);
    // 処理...
}});
""")
```

### Q3: デバッグはどうすればいいですか？

**A**: ブラウザの開発者ツールを活用します：

1. **Chrome/Firefox**: F12キーで開発者ツールを開く
2. **Console**タブでエラーメッセージを確認
3. **Network**タブでWASMファイルの読み込みを確認
4. **Source**タブでブレークポイントを設定

デバッグ用のヘルパー関数:
```javascript
// デバッグ情報を出力
function debugTensor(tensor, name) {
    console.log(`=== ${name} ===`);
    console.log('Shape:', tensor.shape());
    console.log('Data:', tensor.to_array());
    console.log('Requires grad:', tensor.requires_grad());
    console.log('Device:', tensor.device());
}
```

### Q4: より高度な機能（CNN、RNNなど）は使えますか？

**A**: 現在のWASM版では基本的な機能に限定されています。高度な機能については：

1. **利用可能**: 基本的なテンソル演算、自動微分、簡単なNN
2. **制限あり**: GPU演算、大規模モデル
3. **今後追加予定**: CNN層、RNN層、最適化アルゴリズム

### Q5: エラーが出て動かない場合は？

チェックリスト：

1. ✅ Rustがインストールされているか: `rustc --version`
2. ✅ wasm-packがインストールされているか: `wasm-pack --version`
3. ✅ WASMビルドが成功したか: `ls pkg/`
4. ✅ Jupyterが最新版か: `jupyter --version`
5. ✅ ブラウザが対応しているか（Chrome/Firefox/Safari推奨）

それでも解決しない場合は、以下の情報と共にIssueを作成してください：
- OS とバージョン
- ブラウザとバージョン
- エラーメッセージの全文
- 実行したコマンドの履歴

## 次のステップ

1. 📖 [RusTorch WASMの詳細なAPI](./wasm.md)
2. 🔬 [高度な例題集](../examples/)
3. 🚀 [パフォーマンス最適化ガイド](./wasm-memory-optimization.md)
4. 🧪 [テスト方法](./wasm-testing.md)

## コミュニティとサポート

- GitHub: [RusTorch Repository](https://github.com/yourusername/rustorch)
- Discord: [RusTorch Community](https://discord.gg/rustorch)
- Stack Overflow: タグ `rustorch-wasm` を使用

---

Happy Learning with RusTorch WASM! 🦀🔥📓