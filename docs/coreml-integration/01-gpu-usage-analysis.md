# RusTorch GPU使用箇所 完全分析レポート

> **作成日**: 2025-09-19
> **対象ブランチ**: feature/coreml-integration
> **分析範囲**: RusTorchプロジェクト全体のGPU関連コード

## 📋 Executive Summary

RusTorchプロジェクトにおけるGPU使用箇所を包括的に調査し、CoreML統合の対象となる機能領域を特定しました。合計**300以上のGPU関連シンボル**が検出され、以下の主要カテゴリに分類されます。

### 🎯 主要GPU使用カテゴリ

| カテゴリ | 機能数 | 重要度 | CoreML対応優先度 |
|----------|---------|--------|-----------------|
| **デバイス管理** | 25 | 🔴 高 | Phase 1 |
| **テンソル演算** | 85 | 🔴 高 | Phase 1-2 |
| **ニューラルネット** | 45 | 🔴 高 | Phase 2 |
| **メモリ管理** | 35 | 🟡 中 | Phase 3 |
| **分散処理** | 40 | 🟡 中 | Phase 3 |
| **プロファイリング** | 30 | 🟢 低 | Phase 4 |
| **WebGPU/WASM** | 25 | 🟢 低 | 除外 |
| **検証・テスト** | 55 | 🟢 低 | 除外 |

---

## 🏗️ 1. デバイス管理層 (Core Infrastructure)

### 1.1 基本デバイス管理
**ファイル**: `src/gpu/mod.rs`

#### 主要コンポーネント

##### DeviceType (Enum)
```rust
pub enum DeviceType {
    Cpu,
    Cuda(usize),      // ←CoreML対応対象
    Metal(usize),     // ←CoreML統合候補
    OpenCL(usize),    // ←CoreML代替候補
}
```

##### GpuContext (Struct)
```rust
pub struct GpuContext {
    device: DeviceType,
    memory_pool_size: usize,
    stream_count: usize,
}
```

##### DeviceManager (Struct)
```rust
pub struct DeviceManager {
    contexts: Vec<GpuContext>,
    current_device: usize,
}
```

#### 主要メソッド
- `is_gpu_available()`: GPU利用可能性検査
- `set_device()`: デバイス切り替え
- `current_device()`: 現在のデバイス取得

### 1.2 デバイス検出・管理
**ファイル**: `src/tensor/device.rs`

- `Device` enum: テンソルレベルのデバイス管理
- デバイス間転送処理
- フォールバック機構

---

## ⚡ 2. テンソル演算層 (Compute Engine)

### 2.1 基本演算 (Element-wise Operations)
**ファイル**: `src/gpu/verification_tests.rs`, `src/gpu/performance_benchmark.rs`

#### GPU加速対象演算
```rust
// 要素ごと演算
gpu_elementwise_add()    // ✅ CoreML対応可能
gpu_elementwise_sub()    // ✅ CoreML対応可能
gpu_elementwise_mul()    // ✅ CoreML対応可能
gpu_elementwise_div()    // ✅ CoreML対応可能

// 活性化関数
gpu_relu()               // ✅ CoreML対応可能
gpu_gelu()              // ✅ CoreML対応可能
gpu_softmax()           // ✅ CoreML対応可能
```

### 2.2 線形代数演算
**ファイル**: `src/gpu/matrix_ops.rs`

#### 行列演算
```rust
trait GpuLinearAlgebra<T> {
    fn gpu_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;        // ✅ CoreML対応必須
    fn gpu_batch_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;  // ✅ CoreML対応必須
    fn gpu_matvec(&self, vec: &Tensor<T>) -> Result<Tensor<T>>;         // ✅ CoreML対応可能
}
```

#### 実行エンジン
- `GpuMatrixExecutor<T>`: 行列演算実行器
- `GpuBatchMatrixExecutor<T>`: バッチ行列演算実行器

### 2.3 リダクション演算
**ファイル**: `src/gpu/reduction_ops.rs`

```rust
trait GpuReduction<T> {
    fn gpu_sum(&self, dim: Option<usize>) -> Result<Tensor<T>>;    // ✅ CoreML対応可能
    fn gpu_mean(&self, dim: Option<usize>) -> Result<Tensor<T>>;   // ✅ CoreML対応可能
    fn gpu_max(&self, dim: Option<usize>) -> Result<Tensor<T>>;    // ✅ CoreML対応可能
    fn gpu_min(&self, dim: Option<usize>) -> Result<Tensor<T>>;    // ✅ CoreML対応可能
    fn gpu_std(&self, dim: Option<usize>) -> Result<Tensor<T>>;    // ⚠️  CoreML制限あり
    fn gpu_var(&self, dim: Option<usize>) -> Result<Tensor<T>>;    // ⚠️  CoreML制限あり
}
```

### 2.4 パラレル処理
**ファイル**: `src/tensor/gpu_parallel.rs` (1,104行の大規模ファイル)

#### 中心的なtrait
```rust
trait GpuParallelOp<T> {
    fn gpu_elementwise_op(&self, other: &Tensor<T>, op: ElementwiseOp) -> Result<Tensor<T>>;
    fn gpu_matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    fn gpu_reduce(&self, op: ReductionOp, dim: Option<usize>) -> Result<Tensor<T>>;
}

trait GpuBatchParallelOp<T> {
    fn gpu_batch_normalize(&self, mean: &Tensor<T>, var: &Tensor<T>) -> Result<Tensor<T>>;
    fn gpu_batch_conv2d(&self, kernel: &Tensor<T>, bias: Option<&Tensor<T>>) -> Result<Tensor<T>>;
    fn gpu_batch_attention(&self, query: &Tensor<T>, key: &Tensor<T>, value: &Tensor<T>) -> Result<Tensor<T>>;
}
```

#### 実行戦略
```rust
pub enum GpuExecutionStrategy {
    GpuOnly,                                    // GPU専用実行
    CpuOnly,                                   // CPU専用実行
    Hybrid { gpu_threshold: usize },           // ハイブリッド実行
}
```

---

## 🧠 3. ニューラルネットワーク層

### 3.1 畳み込み演算
**ファイル**: `src/gpu/conv_ops.rs`

#### 畳み込み演算trait
```rust
trait GpuConvolution<T> {
    fn gpu_conv2d(&self, kernel: &Tensor<T>, stride: (usize, usize),
                  padding: (usize, usize)) -> Result<Tensor<T>>;              // ✅ CoreML対応必須
    fn gpu_batch_conv2d(&self, kernel: &Tensor<T>) -> Result<Tensor<T>>;      // ✅ CoreML対応必須
    fn gpu_conv2d_transpose(&self, kernel: &Tensor<T>) -> Result<Tensor<T>>;  // ✅ CoreML対応可能
}

trait GpuPooling<T> {
    fn gpu_max_pool2d(&self, kernel_size: (usize, usize),
                      stride: (usize, usize)) -> Result<Tensor<T>>;           // ✅ CoreML対応必須
    fn gpu_avg_pool2d(&self, kernel_size: (usize, usize),
                      stride: (usize, usize)) -> Result<Tensor<T>>;           // ✅ CoreML対応必須
}
```

#### 実行エンジン
- `GpuConvolutionExecutor<T>`: 畳み込み実行器
- `GpuPoolingExecutor<T>`: プーリング実行器

---

## 💾 4. メモリ管理層

### 4.1 GPU メモリ管理
**ファイル**: `src/gpu/memory_integration.rs`, `src/gpu/memory_ops/`

#### 統計・カウンター
```rust
struct AccessCounters {
    gpu_accesses: AtomicU64,          // GPU アクセス回数
    last_gpu_access: AtomicU64,       // 最後のGPUアクセス時刻
}

struct FaultStatistics {
    gpu_faults: AtomicU64,            // GPU メモリフォルト回数
}
```

#### メモリ操作
- **CUDA**: `src/gpu/memory_ops/cuda.rs`
- **Metal**: `src/gpu/memory_ops/metal.rs`  ← **CoreML統合最有力候補**
- **OpenCL**: `src/gpu/memory_ops/opencl.rs`
- **転送**: `src/gpu/memory_ops/transfer.rs`

### 4.2 メモリ転送
**ファイル**: `src/gpu/memory_transfer.rs`

GPU⇔Host間のデータ転送処理。CoreML統合時の重要なボトルネック。

---

## 🌐 5. 分散処理層

### 5.1 マルチGPU管理
**ファイル**: `src/gpu/multi_gpu.rs` (1,252行の大規模ファイル)

#### トポロジー管理
```rust
struct GpuTopology {
    num_gpus: usize,                  // GPU数
    memory_per_gpu: Vec<usize>,       // GPU別メモリ容量
}

struct CommunicationGroup {
    gpu_ids: Vec<usize>,              // 通信グループのGPU ID
}
```

#### 分散訓練
```rust
struct DataParallelTrainer {
    num_gpus: usize,                  // データ並列GPU数
}
```

### 5.2 同期プリミティブ
**ファイル**: `src/gpu/sync_primitives.rs`

```rust
struct MultiGpuBarrier {
    num_gpus: usize,                  // 同期対象GPU数
    gpu_barriers: Vec<Barrier>,       // GPU別バリア
}
```

### 5.3 分散処理統合
**ファイル**: `src/distributed/multi_gpu_validation.rs`

- `MultiGpuValidator<T>`: マルチGPU検証
- `benchmark_single_gpu()` vs `benchmark_multi_gpu()` 性能比較

---

## 📊 6. プロファイリング・モニタリング層

### 6.1 性能プロファイラ
**ファイル**: `src/profiler/`, `src/gpu/multi_gpu_profiler.rs`

#### GPU性能メトリクス
```rust
struct GpuBenchmarkMetrics {
    gpu_utilization_percent: f64,     // GPU使用率
    gpu_memory_used_bytes: u64,       // GPU使用メモリ
    gpu_temperature_celsius: f64,     // GPU温度
}
```

#### マルチGPUプロファイラ
```rust
struct MultiGpuProfiler {
    gpu_metrics: HashMap<usize, GpuMetrics>,  // GPU別メトリクス
}
```

### 6.2 ベンチマーク
**ファイル**: `src/gpu/performance_benchmark.rs`

#### GPU性能測定
- `benchmark_gpu_elementwise_*()`: 要素演算性能
- `benchmark_gpu_matmul()`: 行列乗算性能
- `benchmark_gpu_conv2d()`: 畳み込み性能
- `benchmark_gpu_*_pool2d()`: プーリング性能
- `benchmark_gpu_host_to_device()`: 転送性能

---

## 🌍 7. WebGPU/WASM層 (CoreML対象外)

### 7.1 ブラウザGPU
**ファイル**: `src/wasm/gpu/`

WebGPU関連機能。macOSネイティブではないため、CoreML統合対象外。

- `check_webgpu_support()`: WebGPU対応検査
- `webgpu_tensor_*()`: WebGPUテンソル演算

---

## 🧪 8. 検証・テスト層

### 8.1 GPU検証テスト
**ファイル**: `src/gpu/verification_tests.rs`

#### 主要検証項目
- 要素演算の正確性検証
- 行列演算の正確性検証
- ニューラルネット演算の正確性検証
- 性能回帰テスト

### 8.2 統合テスト
**ファイル**: `src/gpu/integration_tests.rs`

GPUカーネルの統合テスト。

---

## 📈 9. 量的分析

### 9.1 ファイル別GPU関連コード量

| ファイル | 行数 | GPU関連度 | CoreML優先度 |
|----------|------|-----------|-------------|
| `src/gpu/mod.rs` | 600+ | 100% | 🔴 最高 |
| `src/tensor/gpu_parallel.rs` | 1,104 | 100% | 🔴 最高 |
| `src/gpu/multi_gpu.rs` | 1,252 | 100% | 🟡 中 |
| `src/gpu/performance_benchmark.rs` | 1,400+ | 90% | 🟢 低 |
| `src/gpu/verification_tests.rs` | 1,000+ | 90% | 🟢 低 |
| `src/distributed/multi_gpu_validation.rs` | 704 | 80% | 🟡 中 |

### 9.2 機能カテゴリ別シンボル数

```
デバイス管理: 25個
├─ DeviceType関連: 8個
├─ GpuContext関連: 7個
└─ DeviceManager関連: 10個

テンソル演算: 85個
├─ 要素演算: 25個
├─ 線形代数: 20個
├─ リダクション: 15個
└─ パラレル処理: 25個

ニューラルネット: 45個
├─ 畳み込み: 25個
└─ プーリング: 20個

メモリ管理: 35個
分散処理: 40個
プロファイリング: 30個
WebGPU: 25個
検証・テスト: 55個
```

---

## 🎯 10. CoreML統合対象 優先度マトリクス

### 🔴 Phase 1: 最優先 (必須実装)

| 機能 | ファイル | 理由 |
|------|----------|------|
| DeviceType::CoreML | `src/gpu/mod.rs` | デバイス管理の中核 |
| gpu_matmul | `src/gpu/matrix_ops.rs` | ML推論の基本演算 |
| gpu_elementwise_* | `src/gpu/verification_tests.rs` | 要素演算の基礎 |
| gpu_conv2d | `src/gpu/conv_ops.rs` | CNN推論の中核 |
| gpu_*_pool2d | `src/gpu/conv_ops.rs` | CNN推論の必須要素 |

### 🟡 Phase 2: 高優先 (性能向上)

| 機能 | ファイル | 理由 |
|------|----------|------|
| gpu_batch_* | `src/tensor/gpu_parallel.rs` | バッチ処理性能 |
| gpu_reduce_* | `src/gpu/reduction_ops.rs` | 統計演算 |
| gpu_batch_normalize | `src/tensor/gpu_parallel.rs` | 正規化処理 |
| gpu_attention | `src/tensor/gpu_parallel.rs` | Transformer対応 |

### 🟢 Phase 3: 中優先 (システム最適化)

| 機能 | ファイル | 理由 |
|------|----------|------|
| メモリ管理統合 | `src/gpu/memory_ops/metal.rs` | メモリ効率 |
| プロファイリング統合 | `src/profiler/` | 性能モニタリング |
| 分散処理連携 | `src/gpu/multi_gpu.rs` | スケーラビリティ |

### ❌ 対象外

- WebGPU関連: ブラウザ環境のため
- 検証・テスト: 機能テストのため
- OpenCL関連: CoreMLと重複のため

---

## 🔍 11. 技術的考察

### 11.1 CoreML統合の技術的課題

#### メモリ管理
- **課題**: GPU⇔CoreML間のメモリ共有
- **対策**: Metal共有メモリプールの活用

#### 実行戦略
- **課題**: GPU vs CoreMLの使い分け
- **対策**: ハイブリッド実行戦略の拡張

#### API設計
- **課題**: 既存GPUアーキテクチャとの統合
- **対策**: trait-based設計の活用

### 11.2 既存コードとの互換性

#### 良好な統合ポイント
- ✅ DeviceType enumの拡張性
- ✅ trait-based演算設計
- ✅ ハイブリッド実行戦略

#### 課題となるポイント
- ⚠️ メモリレイアウトの違い
- ⚠️ 数値精度の管理
- ⚠️ エラーハンドリングの統合

---

## 📝 12. 結論・提言

### 12.1 CoreML統合の実現可能性: 🟢 **高い**

1. **アーキテクチャ適合性**: 既存のマルチGPU設計が CoreML統合に適している
2. **段階的実装**: Phase分けによる リスク分散が可能
3. **既存機能保持**: 既存のGPU機能を損なわない拡張が可能

### 12.2 次のステップ

1. **Phase 1実装計画の詳細化**
2. **CoreML対応方針ドキュメントの作成**
3. **プロトタイプ実装による技術検証**
4. **性能ベンチマークの設計**

### 12.3 期待効果

- **性能向上**: Apple Silicon での推論性能 20-40%向上期待
- **メモリ効率**: CoreML最適化による メモリ使用量削減
- **エコシステム統合**: macOSネイティブML機能の完全活用

---

*このドキュメントは、RusTorch プロジェクトのCoreML統合戦略策定の基礎資料として作成されました。詳細な実装計画は別途「CoreML対応方針ドキュメント」にて提供されます。*