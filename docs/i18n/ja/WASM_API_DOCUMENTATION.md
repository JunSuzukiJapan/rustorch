# 🌐 WebAssembly API ドキュメント

> 📚 **メインドキュメント**: [API ドキュメント](API_DOCUMENTATION.md)  
> 🔗 **関連ガイド**: [WASM機能強化ロードマップ](WASM_API_Enhancement_Roadmap.md)

このドキュメントは、RusTorchを使用したブラウザベース機械学習のための包括的なWebAssembly (WASM) APIリファレンスを含んでいます。

## 🚀 実装状況

**✅ 完了フェーズ** (95%実装価値):
- **フェーズ1** (60%): 特殊関数、分布、FFT、損失関数
- **フェーズ2** (25%): コンピュータビジョン、簡素化自動微分、ブラウザストレージ
- **フェーズ3** (10%): WASM制約下での線形代数

**🌟 主要機能**:
- ブラウザ互換テンソル演算
- 簡素化自動微分を持つニューラルネットワークレイヤー
- コンピュータビジョン演算 (Harris corners, morphology, LBP)
- モデル永続化 (IndexedDB, LocalStorage, 圧縮)
- 線形代数 (QR, LU, SVD分解、固有値)
- ChromeブラウザでのWebGPU加速
- JavaScript相互運用性と型変換

## 目次

- [🌐 WebAssemblyサポート](#-webassemblyサポート)
- [🧮 WASMテンソル演算](#-wasmテンソル演算)
- [🧠 WASMニューラルネットワークレイヤー](#-wasmニューラルネットワークレイヤー)
- [🌍 ブラウザ統合](#-ブラウザ統合)
- [⚡ WebGPU加速](#-webgpu加速)
- [🔧 高度なWASM機能](#-高度なwasm機能)
- [💾 メモリ管理](#-メモリ管理)
- [📡 信号処理](#-信号処理)
- [🔧 WASMユーティリティとヘルパー](#-wasmユーティリティとヘルパー)
- [🔄 後方互換性](#-後方互換性)

## 🌐 WebAssemblyサポート

### WASMモジュール構造

```
src/
└── wasm/                # WebAssemblyバインディング
    ├── core/           # コアテンソル演算
    ├── data/           # データ分布とサンプリング
    ├── math/           # 数学関数とFFT
    ├── ml/             # 機械学習コンポーネント
    ├── vision/         # コンピュータビジョン演算
    ├── gpu/            # WebGPU統合
    └── storage/        # ブラウザストレージと永続化
```

### 機能フラグ

`Cargo.toml`にWASMサポートを含めてください：

```toml
[features]
wasm = ["wasm-bindgen", "wasm-bindgen-futures", "web-sys", "js-sys", "console_error_panic_hook"]
webgpu = ["wasm", "dep:wgpu", "dep:wgpu-hal", "dep:wgpu-core", "dep:wgpu-types"]
```

### WASMバインディング

```rust
use rustorch::wasm::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<f32>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor {
            inner: Tensor::from_vec(data.to_vec(), shape.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn add(&self, other: &WasmTensor) -> WasmTensor {
        WasmTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f32> {
        self.inner.to_vec()
    }
}

// WASMでのニューラルネットワーク
#[wasm_bindgen]
pub struct WasmModel {
    model: Sequential<f32>,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmModel {
        let model = Sequential::<f32>::new()
            .add_layer(Box::new(Linear::<f32>::new(2, 10)))
            .add_activation(Box::new(ReLU::<f32>::new()))
            .add_layer(Box::new(Linear::<f32>::new(10, 1)));
        
        WasmModel { model }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let input_tensor = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
        let output = self.model.forward(&input_tensor);
        output.to_vec()
    }
}
```

## 🧮 WASMテンソル演算

### 基本テンソル演算

```rust
use rustorch::wasm::{WasmTensor, WasmLinear};

// WASM互換テンソル作成
let data = vec![1.0, 2.0, 3.0, 4.0];
let shape = vec![2, 2];
let wasm_tensor = WasmTensor::new(data, shape);

// 基本テンソル演算
let tensor_a = WasmTensor::new(vec![1.0, 2.0], vec![2, 1]);
let tensor_b = WasmTensor::new(vec![3.0, 4.0], vec![2, 1]);
let result = tensor_a.add(&tensor_b)?;

// 行列演算
let result = tensor_a.matmul(&tensor_b)?;
let transposed = tensor_a.transpose();

// 要素ごと演算
let squared = tensor_a.square();
let sqrt_result = tensor_a.sqrt();
let sum = tensor_a.sum();
let mean = tensor_a.mean();
```

### JavaScript相互運用

```javascript
import init, { WasmTensor, WasmModel } from './pkg/rustorch.js';

async function runML() {
    await init();
    
    // テンソル作成
    const tensor1 = new WasmTensor([1, 2, 3, 4], [2, 2]);
    const tensor2 = new WasmTensor([5, 6, 7, 8], [2, 2]);
    
    // 演算実行
    const result = tensor1.add(tensor2);
    const resultArray = result.to_array();
    
    console.log('結果:', resultArray); // [6, 8, 10, 12]
    
    // ニューラルネットワーク
    const model = new WasmModel();
    const prediction = model.predict([0.5, 0.3]);
    console.log('予測:', prediction);
}
```

## 🧠 WASMニューラルネットワークレイヤー

### 基本レイヤー

```rust
use rustorch::wasm::nn::{WasmLinear, WasmConv2d, WasmReLU};

// 線形レイヤー
#[wasm_bindgen]
pub struct WasmLinear {
    layer: Linear<f32>,
}

#[wasm_bindgen]
impl WasmLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_features: usize, out_features: usize) -> WasmLinear {
        WasmLinear {
            layer: Linear::<f32>::new(in_features, out_features).unwrap(),
        }
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let output = self.layer.forward(&input.inner).unwrap();
        WasmTensor { inner: output }
    }
}
```

### 活性化関数

```rust
use rustorch::wasm::nn::{WasmReLU, WasmSigmoid, WasmTanh};

#[wasm_bindgen]
pub struct WasmReLU;

#[wasm_bindgen]
impl WasmReLU {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmReLU {
        WasmReLU
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let output = input.inner.relu().unwrap();
        WasmTensor { inner: output }
    }
}
```

## 🌍 ブラウザ統合

### モデル永続化

```rust
use rustorch::wasm::storage::{save_to_indexeddb, load_from_indexeddb};

// IndexedDBへの保存
#[wasm_bindgen]
pub async fn save_model_to_browser(model: &WasmModel, name: &str) -> Result<(), JsValue> {
    let serialized = model.serialize()?;
    save_to_indexeddb(name, &serialized).await
}

// IndexedDBからの読み込み
#[wasm_bindgen]
pub async fn load_model_from_browser(name: &str) -> Result<WasmModel, JsValue> {
    let serialized = load_from_indexeddb(name).await?;
    WasmModel::deserialize(&serialized)
}
```

### ブラウザストレージ

```javascript
// LocalStorageでの軽量保存
const saveModelLocally = (model, name) => {
    const serialized = model.to_json();
    localStorage.setItem(`rustorch_model_${name}`, serialized);
};

// IndexedDBでの重いデータ保存
const saveModelToIndexedDB = async (model, name) => {
    await save_model_to_browser(model, name);
};
```

## ⚡ WebGPU加速

### WebGPU初期化

```rust
use rustorch::wasm::gpu::{init_webgpu, WasmGpuDevice};

#[wasm_bindgen]
pub async fn initialize_webgpu() -> Result<WasmGpuDevice, JsValue> {
    let device = init_webgpu().await?;
    Ok(WasmGpuDevice::new(device))
}

#[wasm_bindgen]
impl WasmGpuDevice {
    pub fn gpu_matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, JsValue> {
        let result = self.execute_matmul(&a.inner, &b.inner)?;
        Ok(WasmTensor { inner: result })
    }
}
```

### WebGPU使用例

```javascript
// WebGPU初期化と使用
const initializeGPU = async () => {
    const gpuDevice = await initialize_webgpu();
    
    const tensorA = new WasmTensor([1, 2, 3, 4], [2, 2]);
    const tensorB = new WasmTensor([5, 6, 7, 8], [2, 2]);
    
    // GPU上で行列積計算
    const result = gpuDevice.gpu_matmul(tensorA, tensorB);
    console.log('GPU結果:', result.to_array());
};
```

## 🔧 高度なWASM機能

### 信号処理

```rust
use rustorch::wasm::signal::{fft, ifft, spectrogram};

#[wasm_bindgen]
pub struct WasmSignalProcessor;

#[wasm_bindgen]
impl WasmSignalProcessor {
    #[wasm_bindgen]
    pub fn fft(&self, input: &WasmTensor) -> WasmTensor {
        let result = fft(&input.inner).unwrap();
        WasmTensor { inner: result }
    }
    
    #[wasm_bindgen]
    pub fn spectrogram(&self, signal: &WasmTensor, window_size: usize) -> WasmTensor {
        let result = spectrogram(&signal.inner, window_size).unwrap();
        WasmTensor { inner: result }
    }
}
```

### コンピュータビジョン

```rust
use rustorch::wasm::vision::{harris_corners, morphology, lbp};

#[wasm_bindgen]
pub struct WasmVision;

#[wasm_bindgen]
impl WasmVision {
    #[wasm_bindgen]
    pub fn detect_corners(&self, image: &WasmTensor, threshold: f32) -> WasmTensor {
        let corners = harris_corners(&image.inner, threshold).unwrap();
        WasmTensor { inner: corners }
    }
    
    #[wasm_bindgen]
    pub fn morphological_opening(&self, image: &WasmTensor, kernel_size: usize) -> WasmTensor {
        let result = morphology::opening(&image.inner, kernel_size).unwrap();
        WasmTensor { inner: result }
    }
    
    #[wasm_bindgen]
    pub fn local_binary_pattern(&self, image: &WasmTensor, radius: f32, neighbors: usize) -> WasmTensor {
        let lbp_result = lbp(&image.inner, radius, neighbors).unwrap();
        WasmTensor { inner: lbp_result }
    }
}
```

## 💾 メモリ管理

### 効率的メモリ使用

```rust
use rustorch::wasm::memory::{WasmMemoryPool, optimize_memory};

#[wasm_bindgen]
pub struct WasmMemoryManager {
    pool: WasmMemoryPool,
}

#[wasm_bindgen]
impl WasmMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new(pool_size: usize) -> WasmMemoryManager {
        WasmMemoryManager {
            pool: WasmMemoryPool::new(pool_size),
        }
    }
    
    #[wasm_bindgen]
    pub fn allocate_tensor(&mut self, shape: &[usize]) -> WasmTensor {
        let tensor = self.pool.allocate_tensor(shape).unwrap();
        WasmTensor { inner: tensor }
    }
    
    #[wasm_bindgen]
    pub fn optimize_memory(&mut self) {
        optimize_memory(&mut self.pool);
    }
}
```

### ガベージコレクション最適化

```javascript
// メモリ効率的なWASM使用
class RusTorchWasmManager {
    constructor() {
        this.memoryManager = new WasmMemoryManager(1024 * 1024); // 1MB pool
        this.tensorCache = new Map();
    }
    
    createTensor(data, shape, cacheKey = null) {
        const tensor = this.memoryManager.allocate_tensor(shape);
        if (cacheKey) {
            this.tensorCache.set(cacheKey, tensor);
        }
        return tensor;
    }
    
    cleanup() {
        this.tensorCache.clear();
        this.memoryManager.optimize_memory();
    }
}
```

## 📡 信号処理

### FFTと周波数解析

```rust
use rustorch::wasm::signal::{FFTProcessor, SpectralAnalyzer};

#[wasm_bindgen]
pub struct WasmFFT {
    processor: FFTProcessor,
}

#[wasm_bindgen]
impl WasmFFT {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> WasmFFT {
        WasmFFT {
            processor: FFTProcessor::new(size),
        }
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let result = self.processor.fft(&input.inner).unwrap();
        WasmTensor { inner: result }
    }
    
    #[wasm_bindgen]
    pub fn inverse(&self, input: &WasmTensor) -> WasmTensor {
        let result = self.processor.ifft(&input.inner).unwrap();
        WasmTensor { inner: result }
    }
}
```

## 🔄 実用的なWASMワークフロー

### ブラウザでの機械学習パイプライン

```javascript
class MLPipeline {
    constructor() {
        this.model = null;
        this.preprocessor = null;
    }
    
    async initialize() {
        await init(); // WASM初期化
        
        // モデル作成
        this.model = new WasmModel();
        this.preprocessor = new WasmVision();
    }
    
    async processImage(imageData) {
        // 画像前処理
        const imageTensor = new WasmTensor(imageData, [224, 224, 3]);
        const corners = this.preprocessor.detect_corners(imageTensor, 0.1);
        
        // 特徴抽出
        const features = this.preprocessor.local_binary_pattern(corners, 1.0, 8);
        
        // 予測
        const prediction = this.model.predict(features.to_array());
        
        return prediction;
    }
    
    async saveModel(name) {
        await save_model_to_browser(this.model, name);
    }
    
    async loadModel(name) {
        this.model = await load_model_from_browser(name);
    }
}
```

## ⚠️ WASMでの制限事項

1. **メモリ制限**: ブラウザのメモリ制約により大きなモデル（>1GB）は制限される
2. **並列実行**: Web Workersは部分的サポート、完全な並列性は将来実装予定
3. **ファイルI/O**: ローカルファイルアクセス制限、ブラウザAPIのみ
4. **デバッグ**: 限定的なデバッグツール、主にconsole.logとブラウザdevtools

## 🔧 パフォーマンス最適化

### WASM最適化のヒント

```rust
// SIMD最適化使用
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

// 効率的なメモリアクセスパターン
impl WasmTensor {
    pub fn optimized_matmul(&self, other: &WasmTensor) -> Result<WasmTensor, JsValue> {
        // キャッシュフレンドリーな行列積
        let result = self.inner.matmul_optimized(&other.inner)?;
        Ok(WasmTensor { inner: result })
    }
}
```

## 🔗 関連リンク

- [メインAPI ドキュメント](../API_DOCUMENTATION.md)
- [Jupyterガイド](jupyter-guide.md)
- [GitHub リポジトリ](https://github.com/JunSuzukiJapan/RusTorch)
- [npm パッケージ](https://www.npmjs.com/package/rustorch-wasm)
- [WebGPU仕様](https://gpuweb.github.io/gpuweb/)

---

**最終更新**: v0.5.15 | **ライセンス**: MIT | **作者**: Jun Suzuki