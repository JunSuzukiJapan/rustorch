# Metal GPU Integration Plan

**重要な制約**: hybrid-f32は使用しない。rustorchの既存Metal実装（MetalKernelExecutor）を直接使用する。

## 現状分析

### 既存のMetal実装
- ✅ `MetalKernelExecutor` (src/gpu/metal_kernels.rs) - シングルトンパターンで実装済み
- ✅ `MetalBuffer` - GPU/CPUメモリ転送機能
- ✅ Metal Performance Shaders - 基本演算カーネル実装済み
  - Element-wise operations (add, mul, etc.)
  - Matrix multiplication
  - Reduction operations

### 現在の問題点
- ❌ `GPTModel::forward()` はMetalを使用せず、CPUフォールバック
- ❌ `LlamaModel` にはMetal統合がない
- ❌ `BackendLoader` はhybrid-f32経由でしかMetalを使用できない

## Metal統合戦略

### フェーズ1: GPTModel Metal統合 (優先度: 高)

#### 目標
`rustorch::models::GPTModel` を `MetalKernelExecutor` で直接加速

#### 実装手順

1. **GPTModelにMetal対応フィールド追加**
```rust
pub struct GPTModel {
    // 既存フィールド
    device_type: DeviceType,

    // 新規: Metal executor
    #[cfg(feature = "metal")]
    metal_executor: Option<Arc<Mutex<MetalKernelExecutor>>>,
}
```

2. **forward()メソッドをMetal対応に変更**
```rust
impl GPTModel {
    pub fn forward(&mut self, input_ids: &[usize]) -> Result<Tensor> {
        match self.device_type {
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    self.forward_metal(input_ids)
                }
                #[cfg(not(feature = "metal"))]
                {
                    eprintln!("⚠️  Metal not available, falling back to CPU");
                    self.forward_cpu(input_ids)
                }
            }
            _ => self.forward_cpu(input_ids),
        }
    }

    #[cfg(feature = "metal")]
    fn forward_metal(&mut self, input_ids: &[usize]) -> Result<Tensor> {
        // MetalKernelExecutorを使用したGPU加速実装
        // 1. 入力をGPUバッファに転送
        // 2. MetalカーネルでTransformer演算実行
        // 3. 結果をCPUに戻す
    }
}
```

3. **Metal カーネル実装の拡張**
- Embedding lookup (Metal buffer)
- Multi-head attention (Metal matmul + softmax)
- Feed-forward network (Metal matmul + activation)
- Layer normalization (Metal reduce + normalize)

### フェーズ2: BackendLoader Metal直接統合

#### 目標
hybrid-f32を経由せずにMetalBackendを直接使用

#### 実装手順

1. **backend_loader.rsにMetal直接ロードを追加**
```rust
impl BackendLoader {
    /// Load model with Metal backend (direct, no hybrid-f32)
    #[cfg(feature = "metal")]
    pub fn load_metal_direct(
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        tracing::info!("Loading model with direct Metal GPU backend");

        // MetalKernelExecutorを初期化
        let executor = MetalKernelExecutor::get()?;

        // GPTModelをMetal有効で読み込み
        let mut model = GPTModel::from_gguf_with_backend(
            model_path,
            DeviceType::Metal
        )?;

        // MetalExecutorを設定
        model.set_metal_executor(executor);

        engine.set_gpt_model(model);
        Ok(())
    }
}
```

2. **CliBackendにMetal専用オプション追加**
```rust
pub enum Backend {
    Metal,      // Metal直接（hybrid-f32なし）
    HybridF32,  // 既存のhybrid-f32 (廃止予定)
    // ...
}
```

### フェーズ3: 最適化とテスト

#### パフォーマンス最適化
- [ ] KV Cacheの GPU resident化
- [ ] Batch処理の効率化
- [ ] メモリ転送の最小化

#### テスト項目
- [ ] Metal直接実行での推論精度検証
- [ ] hybrid-f32との速度比較
- [ ] メモリ使用量測定
- [ ] 複数量子化形式でのテスト (Q4_K_M, Q6_K, Q8_0)

## 実装優先度

### 必須 (Phase 1)
1. ✅ リファクタリング完了
2. GPTModel Metal統合
3. 基本的なforward()実装

### 推奨 (Phase 2)
4. BackendLoaderへの統合
5. パフォーマンステスト

### 将来 (Phase 3)
6. LlamaModel Metal対応
7. KV Cache最適化
8. Batch処理サポート

## 技術的課題

### メモリ管理
- Metal Bufferのライフタイム管理
- CPU ↔ GPU データ転送の最適化
- Unified Memory vs Shared Memory

### 量子化対応
- GGUFの量子化テンソル → Metal bufferへの変換
- Metal側での dequantization カーネル実装
- 精度とパフォーマンスのトレードオフ

### エラーハンドリング
- Metal初期化失敗時のCPUフォールバック
- GPUメモリ不足時の対処
- デバイス互換性チェック

## 次のステップ

1. **GPTModel::forward_metal() 実装開始**
   - 簡単な演算から始める（embedding, matmul）
   - 段階的にTransformer層を追加

2. **テストケース作成**
   - 小さいモデルで動作確認
   - CPU実装との出力比較

3. **ドキュメント更新**
   - Metal使用方法の説明
   - パフォーマンス測定結果

## 参考資料

- rustorch Metal実装: `src/gpu/metal_kernels.rs`
- Metal Performance Shaders: https://developer.apple.com/metal/
- GGUF形式仕様: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
