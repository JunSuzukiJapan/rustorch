# RusTorch API ドキュメント

## 📚 完全API リファレンス

このドキュメントは、RusTorch v0.5.15の包括的なAPIドキュメントを、モジュールと機能別に整理して提供しています。すべての1060以上のテストにわたって一貫したエラー管理のために、`RusTorchError`と`RusTorchResult<T>`による統一されたエラーハンドリング機能があります。**Phase 8完了**により、条件演算、インデックス操作、統計関数を含む高度なテンソルユーティリティが追加されました。**Phase 9完了**により、モデル保存/読み込み、JITコンパイル、PyTorch互換性を含む複数形式サポートの包括的シリアライゼーションシステムが導入されました。

## 🏗️ コアアーキテクチャ

### モジュール構成

```
rustorch/
├── tensor/              # コアテンソル操作とデータ構造
├── nn/                  # ニューラルネットワークレイヤーと関数
├── autograd/            # 自動微分エンジン
├── optim/               # オプティマイザーと学習率スケジューラー
├── special/             # 特殊数学関数
├── distributions/       # 統計分布
├── vision/              # コンピュータビジョン変換
├── linalg/              # 線形代数演算 (BLAS/LAPACK)
├── gpu/                 # GPU加速 (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # スパーステンソル演算とプルーニング (Phase 12)
├── serialization/       # モデルシリアライゼーションとJITコンパイル (Phase 9)
└── wasm/                # WebAssemblyバインディング ([WASM APIドキュメント](WASM_API_DOCUMENTATION.md)を参照)
```

## 📊 Tensorモジュール

### コアテンソル作成

```rust
use rustorch::tensor::Tensor;

// 基本作成
let tensor = Tensor::new(vec![2, 3]);               // 形状ベース作成
let tensor = Tensor::from_vec(data, vec![2, 3]);    // データベクターから作成
let tensor = Tensor::zeros(vec![10, 10]);           // ゼロ埋めテンソル
let tensor = Tensor::ones(vec![5, 5]);              // ワン埋めテンソル
let tensor = Tensor::randn(vec![3, 3]);             // ランダム正規分布
let tensor = Tensor::rand(vec![3, 3]);              // ランダム一様分布 [0,1)
let tensor = Tensor::eye(5);                        // 単位行列
let tensor = Tensor::full(vec![2, 2], 3.14);       // 特定値で埋める
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // 範囲テンソル
let tensor = Tensor::linspace(0.0, 1.0, 100);      // 線形スペーシング
```

### テンソル演算

```rust
// 算術演算
let result = a.add(&b);                             // 要素ごとの加算
let result = a.sub(&b);                             // 要素ごとの減算
let result = a.mul(&b);                             // 要素ごとの乗算
let result = a.div(&b);                             // 要素ごとの除算
let result = a.pow(&b);                             // 要素ごとの累乗
let result = a.rem(&b);                             // 要素ごとの余り

// 行列演算
let result = a.matmul(&b);                          // 行列積
let result = a.transpose();                         // 行列転置
let result = a.dot(&b);                             // ドット積

// 数学関数
let result = tensor.exp();                          // 指数
let result = tensor.ln();                           // 自然対数
let result = tensor.log10();                        // 常用対数
let result = tensor.sqrt();                         // 平方根
let result = tensor.abs();                          // 絶対値
let result = tensor.sin();                          // サイン関数
let result = tensor.cos();                          // コサイン関数
let result = tensor.tan();                          // タンジェント関数
let result = tensor.asin();                         // アークサイン
let result = tensor.acos();                         // アークコサイン
let result = tensor.atan();                         // アークタンジェント
let result = tensor.sinh();                         // 双曲線サイン
let result = tensor.cosh();                         // 双曲線コサイン
let result = tensor.tanh();                         // 双曲線タンジェント
let result = tensor.floor();                        // フロア関数
let result = tensor.ceil();                         // シーリング関数
let result = tensor.round();                        // ラウンド関数
let result = tensor.sign();                         // 符号関数
let result = tensor.max();                          // 最大値
let result = tensor.min();                          // 最小値
let result = tensor.sum();                          // 全要素の合計
let result = tensor.mean();                         // 平均値
let result = tensor.std();                          // 標準偏差
let result = tensor.var();                          // 分散

// 形状操作
let result = tensor.reshape(vec![6, 4]);            // テンソル再形成
let result = tensor.squeeze();                      // サイズ1次元の除去
let result = tensor.unsqueeze(1);                   // 指定インデックスに次元追加
let result = tensor.permute(vec![1, 0, 2]);         // 次元の順列
let result = tensor.expand(vec![10, 10, 5]);        // テンソル次元の拡張

// 高度な形状操作 (Phase 1)
let result = tensor.squeeze_dim(1);                 // 特定のサイズ1次元の除去
let result = tensor.flatten_owned();                // 1Dテンソルに平坦化
let result = tensor.flatten_range(1, Some(3));      // 次元1-3の平坦化
let result = tensor.unflatten(0, &[2, 3]);         // 平坦化の逆操作
```

### テンソル結合操作 (Phase 2)

```rust
// テンソル連結
let result = Tensor::cat(&[a, b, c], 0)?;           // 軸0でテンソル連結
let result = Tensor::stack(&[a, b, c], 1)?;         // 軸1でテンソルスタック
let (a, b) = tensor.chunk(2, 0)?;                  // テンソルを2つに分割
let (a, b, c) = tensor.split(&[2, 3, 5], 0)?;     // 指定サイズで分割

// 高度な結合操作
let result = tensor.repeat(&[2, 3]);               // 指定回数で繰り返し
let result = tensor.tile(&[2, 3]);                 // タイル配置
```

### 高度なテンソル操作 (Phase 8)

```rust
use rustorch::tensor::{IndexSelect, ConditionOps};

// 条件操作
let result = a.where_condition(&mask, &b)?;        // 条件ベース選択
let mask = tensor.gt(0.5)?;                        // 条件マスク作成
let indices = tensor.nonzero()?;                   // ゼロ以外のインデックス
let result = tensor.masked_select(&mask)?;         // マスクによる選択

// インデックス操作
let result = tensor.index_select(0, &indices)?;    // インデックス選択
let result = tensor.gather(1, &indices)?;          // インデックス収集
let result = tensor.scatter(1, &indices, &values)?; // インデックス散布

// 統計関数
let result = tensor.median(0)?;                    // 中央値計算
let result = tensor.mode(0)?;                      // 最頻値計算
let (values, indices) = tensor.sort(0, false)?;    // ソート操作
let (values, indices) = tensor.topk(3, 0, true)?;  // トップK値
```

## 🧠 Neural Network (nn) モジュール

### 基本レイヤー

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// 線形レイヤー
let linear = Linear::new(784, 256)?;                // 入力784、出力256
let output = linear.forward(&input)?;

// 畳み込みレイヤー
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// バッチ正規化
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// ドロップアウト
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### 活性化関数

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// 基本活性化関数
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// パラメータ付き活性化関数
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// 使用例
let activated = relu.forward(&input)?;
```

### 損失関数

```rust
use rustorch::nn::{CrossEntropyLoss, MSELoss, BCELoss};

// 分類用損失関数
let ce_loss = CrossEntropyLoss::new(None, None)?;
let loss = ce_loss.forward(&predictions, &targets)?;

// 回帰用損失関数
let mse_loss = MSELoss::new("mean")?;
let loss = mse_loss.forward(&predictions, &targets)?;

// バイナリ分類用損失関数
let bce_loss = BCELoss::new(None, "mean")?;
let loss = bce_loss.forward(&predictions, &targets)?;
```

### Transformerアーキテクチャ (Phase 6)

```rust
use rustorch::nn::{MultiHeadAttention, TransformerBlock, PositionalEncoding};

// マルチヘッドアテンション
let attention = MultiHeadAttention::new(512, 8, 0.1)?; // d_model=512, num_heads=8
let output = attention.forward(&query, &key, &value, None)?;

// Transformerブロック
let transformer = TransformerBlock::new(512, 2048, 8, 0.1)?;
let output = transformer.forward(&input, None)?;

// 位置エンコーディング
let pos_encoding = PositionalEncoding::new(512, 1000)?;
let encoded = pos_encoding.forward(&input)?;
```

## 🔄 自動微分 (Autograd) モジュール

### 基本微分操作

```rust
use rustorch::autograd::{Variable, backward};

// 変数作成（勾配計算を有効化）
let x = Variable::new(Tensor::randn(vec![5, 5]), true)?;
let y = Variable::new(Tensor::randn(vec![5, 5]), true)?;

// 前進計算
let z = x.matmul(&y)?;
let loss = z.sum()?;

// 逆伝播
backward(&loss, true)?;

// 勾配アクセス
let x_grad = x.grad()?;
let y_grad = y.grad()?;
```

### 勾配チェック機能

```rust
use rustorch::autograd::gradcheck;

// 数値勾配との比較による勾配チェック
let inputs = vec![
    Variable::new(Tensor::randn(vec![3, 3]), true)?,
];

let check_passed = gradcheck(
    |inputs| inputs[0].matmul(&inputs[0].transpose()),
    &inputs,
    1e-5,  // 相対許容誤差
    1e-4,  // 絶対許容誤差
)?;
```

## 🎯 オプティマイザー (Optim) モジュール

### 基本オプティマイザー

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Adamオプティマイザー
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// SGDオプティマイザー  
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// 最適化ステップ
optimizer.zero_grad()?;
// ... 前進計算と逆伝播 ...
optimizer.step()?;
```

### 学習率スケジューラー

```rust
use rustorch::optim::scheduler::{StepLR, CosineAnnealingLR, ReduceLROnPlateau};

// ステップ学習率減衰
let step_scheduler = StepLR::new(&mut optimizer, 10, 0.1)?;

// コサインアニーリング
let cosine_scheduler = CosineAnnealingLR::new(&mut optimizer, 100)?;

// プラトー時学習率減少
let plateau_scheduler = ReduceLROnPlateau::new(&mut optimizer, "min", 0.1, 10)?;

// スケジューラー使用
step_scheduler.step()?;
plateau_scheduler.step(validation_loss)?;
```

## 🔢 特殊数学関数 (Special) モジュール

### ガンマ関数とベータ関数

```rust
use rustorch::special::{gamma, lgamma, beta, digamma, polygamma};

let result = gamma(&tensor)?;                       // ガンマ関数
let result = lgamma(&tensor)?;                      // 対数ガンマ関数
let result = beta(&a, &b)?;                         // ベータ関数
let result = digamma(&tensor)?;                     // ディガンマ関数
let result = polygamma(2, &tensor)?;                // ポリガンマ関数
```

### ベッセル関数

```rust
use rustorch::special::{i0, i1, j0, j1, y0, y1};

let result = i0(&tensor)?;                          // 第1種変形ベッセル関数 I0
let result = i1(&tensor)?;                          // 第1種変形ベッセル関数 I1
let result = j0(&tensor)?;                          // 第1種ベッセル関数 J0
let result = j1(&tensor)?;                          // 第1種ベッセル関数 J1
let result = y0(&tensor)?;                          // 第2種ベッセル関数 Y0
let result = y1(&tensor)?;                          // 第2種ベッセル関数 Y1
```

## 📊 分布 (Distributions) モジュール

### 基本分布

```rust
use rustorch::distributions::{Normal, Uniform, Categorical, Bernoulli};

// 正規分布
let normal = Normal::new(0.0, 1.0)?;
let sample = normal.sample(&[100])?;
let log_prob = normal.log_prob(&sample)?;

// 一様分布
let uniform = Uniform::new(0.0, 1.0)?;
let sample = uniform.sample(&[50])?;

// カテゴリカル分布
let probs = Tensor::new(vec![0.3, 0.4, 0.3])?;
let categorical = Categorical::new_probs(&probs)?;
let sample = categorical.sample(&[10])?;

// ベルヌーイ分布
let bernoulli = Bernoulli::new_probs(&Tensor::new(vec![0.7])?)?;
let sample = bernoulli.sample(&[20])?;
```

## 🖼️ コンピュータビジョン (Vision) モジュール

### 画像変換

```rust
use rustorch::vision::transforms::{
    Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip,
    Normalize, ToTensor, ColorJitter
};

// 変換パイプライン
let transform = Compose::new(vec![
    Box::new(Resize::new(256)),
    Box::new(RandomCrop::new(224)),
    Box::new(RandomHorizontalFlip::new(0.5)),
    Box::new(ToTensor::new()),
    Box::new(Normalize::new(
        vec![0.485, 0.456, 0.406],  // mean
        vec![0.229, 0.224, 0.225],  // std
    )),
]);

// 変換適用
let transformed = transform.forward(&input_tensor)?;
```

### データセット

```rust
use rustorch::vision::datasets::{CIFAR10, MNIST, ImageFolder};

// CIFAR-10データセット
let cifar10 = CIFAR10::new("./data", true, Some(transform))?;  // train=true
let (image, label) = cifar10.get_item(0)?;

// MNISTデータセット
let mnist = MNIST::new("./data", false, Some(transform))?;     // train=false

// カスタム画像フォルダ
let dataset = ImageFolder::new("./custom_data", Some(transform))?;
```

## 🔢 線形代数 (Linalg) モジュール

### 分解操作

```rust
use rustorch::linalg::{svd, qr, eig, cholesky, lu};

// 特異値分解
let (u, s, vt) = svd(&tensor, true)?;              // full_matrices=true

// QR分解
let (q, r) = qr(&tensor, "reduced")?;

// 固有値分解
let (eigenvalues, eigenvectors) = eig(&tensor)?;

// コレスキー分解
let l = cholesky(&tensor)?;

// LU分解
let (p, l, u) = lu(&tensor)?;
```

### ノルムと距離

```rust
use rustorch::linalg::{norm, vector_norm, matrix_norm};

let result = norm(&tensor, None, None, false)?;     // フロベニウスノルム
let result = vector_norm(&tensor, 2.0, &[0], false)?; // L2ノルム
let result = matrix_norm(&tensor, "fro", &[0, 1])?; // 行列ノルム
```

## 🚀 GPU加速モジュール

### デバイス管理

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// 利用可能デバイス確認
let device_count = get_device_count()?;
let device = Device::best_available()?;            // 最適デバイス選択

// デバイス設定
set_device(&device)?;

// テンソルをGPUに移動
let gpu_tensor = tensor.to_device(&device)?;
```

### CUDA操作

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// CUDAデバイス操作
let cuda_device = CudaDevice::new(0)?;              // GPU 0使用
let stats = memory_stats(0)?;                      // メモリ統計
println!("使用メモリ: {} MB", stats.used_memory / (1024 * 1024));
```

### Metal操作 (macOS)

```rust
#[cfg(feature = "metal")]
use rustorch::gpu::metal::MetalDevice;

// Metalデバイス操作
let metal_device = MetalDevice::new()?;
let gpu_tensor = tensor.to_metal(&metal_device)?;
```

## 🗜️ スパーステンソル (Sparse) モジュール (Phase 12)

### COOスパーステンソル

```rust
use rustorch::sparse::{SparseTensor, SparseFormat};

// COO形式スパーステンソル作成
let indices = Tensor::from_vec(vec![0, 1, 2, 0, 1], vec![5])?;
let values = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
let sparse = SparseTensor::new_coo(indices, values, vec![3, 4])?;

// 密テンソルとの演算
let dense_result = sparse.to_dense()?;
let sparse_result = dense_tensor.to_sparse(SparseFormat::COO)?;
```

### プルーニング操作

```rust
use rustorch::sparse::pruning::{magnitude_pruning, structured_pruning};

// 大きさベースプルーニング
let pruned = magnitude_pruning(&tensor, 0.5)?;     // 50%プルーニング

// 構造化プルーニング
let pruned = structured_pruning(&tensor, &[1], 0.25)?; // 軸1で25%プルーニング
```

## 💾 シリアライゼーション (Serialization) モジュール (Phase 9)

### モデル保存・読み込み

```rust
use rustorch::serialization::{save_model, load_model, ModelFormat};

// モデル保存
save_model(&model, "model.pt", ModelFormat::PyTorch)?;
save_model(&model, "model.rustorch", ModelFormat::Native)?;

// モデル読み込み
let loaded_model = load_model("model.pt", ModelFormat::PyTorch)?;
let native_model = load_model("model.rustorch", ModelFormat::Native)?;
```

### JITコンパイル

```rust
use rustorch::serialization::jit::{trace, script, JitModule};

// トレーシングベースJIT
let traced_module = trace(&model, &example_input)?;
let output = traced_module.forward(&input)?;

// スクリプトベースJIT
let scripted = script(&model)?;
let optimized_output = scripted.forward(&input)?;
```

## 🌐 WebAssembly (WASM) モジュール

WebAssemblyサポートの詳細については、[WASM API ドキュメント](WASM_API_DOCUMENTATION.md)を参照してください。

### 基本WASM使用法

```rust
use rustorch::wasm::{WasmTensor, wasm_ops};

// WASM環境でのテンソル作成
let wasm_tensor = WasmTensor::new(vec![2, 3])?;
let result = wasm_ops::matmul(&a, &b)?;
```

## 🔧 ユーティリティとエラーハンドリング

### エラー型

```rust
use rustorch::error::{RusTorchError, RusTorchResult};

// エラーハンドリング例
match tensor_operation() {
    Ok(result) => println!("成功: {:?}", result),
    Err(RusTorchError::InvalidShape(msg)) => eprintln!("形状エラー: {}", msg),
    Err(RusTorchError::IncompatibleDevice(msg)) => eprintln!("デバイスエラー: {}", msg),
    Err(RusTorchError::ComputationError(msg)) => eprintln!("計算エラー: {}", msg),
    Err(e) => eprintln!("その他のエラー: {}", e),
}
```

### パフォーマンスヒント

```rust
// 効率的なテンソル演算
let a = Tensor::zeros(vec![1000, 1000]);
let b = Tensor::ones(vec![1000, 1000]);

// in-place演算（メモリ効率良）
a.add_(&b)?;                                        // a += b
a.mul_(&scalar_tensor)?;                            // a *= scalar

// GPU使用時のベストプラクティス
let device = Device::best_available()?;
let gpu_a = a.to_device(&device)?;                  // 一度GPU転送
let gpu_b = b.to_device(&device)?;
let result = gpu_a.matmul(&gpu_b)?;                 // GPU上で計算
let cpu_result = result.to_cpu()?;                  // 必要時のみCPU転送
```

## 📝 実用例とチュートリアル

### 線形回帰

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// データ準備
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// モデル定義
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// 訓練ループ
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("エポック {}: 損失 = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

### 画像分類CNN

```rust
use rustorch::nn::{Conv2d, MaxPool2d, Linear, ReLU, Dropout};

pub struct SimpleCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
    relu: ReLU,
    dropout: Dropout,
}

impl SimpleCNN {
    pub fn new() -> RusTorchResult<Self> {
        Ok(Self {
            conv1: Conv2d::new(3, 32, 3, Some(1), None)?,
            conv2: Conv2d::new(32, 64, 3, Some(1), None)?,
            pool: MaxPool2d::new(2, Some(2))?,
            fc1: Linear::new(64 * 8 * 8, 128)?,
            fc2: Linear::new(128, 10)?,
            relu: ReLU::new(),
            dropout: Dropout::new(0.5)?,
        })
    }
    
    pub fn forward(&self, x: &Variable) -> RusTorchResult<Variable> {
        let x = self.relu.forward(&self.conv1.forward(x)?)?;
        let x = self.pool.forward(&x)?;
        let x = self.relu.forward(&self.conv2.forward(&x)?)?;
        let x = self.pool.forward(&x)?;
        let x = x.reshape(vec![-1, 64 * 8 * 8])?;
        let x = self.relu.forward(&self.fc1.forward(&x)?)?;
        let x = self.dropout.forward(&x, true)?;
        self.fc2.forward(&x)
    }
}
```

## 🔧 高度な機能

### カスタム演算子定義

```rust
use rustorch::tensor::Tensor;

impl Tensor<f32> {
    pub fn custom_activation(&self) -> RusTorchResult<Self> {
        // カスタム活性化関数：Swish (x * sigmoid(x))
        let sigmoid_x = self.sigmoid()?;
        self.mul(&sigmoid_x)
    }
    
    pub fn gelu_precise(&self) -> RusTorchResult<Self> {
        // 精密GELU実装
        let half = Tensor::from(0.5)?;
        let one = Tensor::from(1.0)?;
        let sqrt_2_pi = Tensor::from((2.0 / std::f32::consts::PI).sqrt())?;
        
        let tanh_input = &sqrt_2_pi * self * (one + Tensor::from(0.044715)? * self.pow(&Tensor::from(3.0)?)?)?;
        half * self * (one + tanh_input.tanh()?)
    }
}
```

### 分散学習サポート

```rust
use rustorch::distributed::{init_process_group, all_reduce, DistributedMode};

// 分散学習初期化
init_process_group("nccl", 0, 4)?;                  // rank=0, world_size=4

// AllReduce演算
let reduced = all_reduce(&gradients, DistributedMode::Sum)?;
let averaged = reduced.div(&Tensor::from(4.0)?)?;   // 勾配平均化
```

## 🎯 パフォーマンス最適化

### メモリプール使用

```rust
use rustorch::utils::memory::{MemoryPool, set_memory_strategy};

// メモリプール設定
set_memory_strategy(MemoryStrategy::Pool(1024 * 1024 * 1024))?; // 1GB プール

// 効率的なメモリ使用
let pool = MemoryPool::new(512 * 1024 * 1024)?;    // 512MB プール
let tensor = pool.allocate_tensor(vec![1000, 1000])?;
```

### SIMD最適化

```rust
use rustorch::simd::{simd_add, simd_mul, enable_simd};

// SIMD有効化
enable_simd(true);

// SIMD演算（自動的に使用される）
let result = a.add(&b)?;                            // 内部でSIMD最適化
```

## 📖 API バージョニング

### 安定API vs 実験的API

```rust
// 安定API（推奨）
use rustorch::tensor::Tensor;                       // v0.1+で安定
use rustorch::nn::Linear;                          // v0.2+で安定

// 実験的API（注意して使用）
use rustorch::experimental::quantization::*;       // 実験的量子化
use rustorch::experimental::pruning::*;            // 実験的プルーニング
```

### 非推奨API

```rust
// v0.6.0で削除予定
// let tensor = Tensor::legacy_create(data);        // 非推奨：Tensor::from_vecを使用
// let result = tensor.old_matmul(&other);          // 非推奨：tensor.matmulを使用
```

## 🛠️ デバッグとプロファイリング

### デバッグモード

```rust
use rustorch::debug::{set_debug_mode, print_tensor_info, check_gradients};

// デバッグモード有効化
set_debug_mode(true);

// テンソル情報出力
print_tensor_info(&tensor);

// 勾配チェック
check_gradients(&model, &input, &target)?;
```

### プロファイリング

```rust
use rustorch::profiler::{start_profiling, stop_profiling, get_profile_report};

// プロファイリング開始
start_profiling("gpu")?;

// 測定対象コード
let result = model.forward(&input)?;
let loss = criterion.forward(&result, &target)?;

// プロファイリング終了とレポート取得
let report = stop_profiling()?;
println!("実行プロファイル: {}", report);
```

## ⚠️ 既知の制限事項

1. **GPU メモリ制限**: 大型テンソル（>8GB）では明示的なメモリ管理が必要
2. **WebAssembly制限**: 一部のBLAS演算はWASM環境では利用不可
3. **分散学習**: NCCLバックエンドはLinux環境でのみサポート
4. **Metal制限**: 一部の高度な演算はCUDAバックエンドでのみ利用可能

## 🔗 関連リンク

- [メインREADME](../README.md)
- [WASM API ドキュメント](WASM_API_DOCUMENTATION.md)
- [Jupyterガイド](jupyter-guide.md)
- [GitHub リポジトリ](https://github.com/JunSuzukiJapan/RusTorch)
- [Crates.io パッケージ](https://crates.io/crates/rustorch)

---

**最終更新**: v0.5.15 | **ライセンス**: MIT | **作者**: Jun Suzuki