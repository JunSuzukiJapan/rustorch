# Known Issues / 既知の問題

## Metal Context Leak Warning

### 症状 (Symptoms)
```
Context leak detected, msgtracer returned -1
```

### 原因 (Root Cause)
- `hybrid-f32`フィーチャーは自動的に`metal`フィーチャーを有効化
- `F32Tensor::matmul`がMetal GPU加速を試みる際、Metalデバイスを初期化
- macOSの診断システム (msgtracer) がリソース追跡の警告を出力
- **実際のメモリリークではなく、診断メッセージ**

### 影響 (Impact)
- ⚠️ **実行には影響なし** - 警告のみ
- パフォーマンス低下なし
- メモリ使用量の異常増加なし

### 回避策 (Workaround)

#### オプション1: 環境変数で抑制
```bash
export MTL_DEBUG_LAYER=0
cargo run --features hybrid-f32
```

#### オプション2: Metalデバッグを無効化
```bash
# 一時的に無効化
MTL_DEBUG_LAYER=0 ./your_binary

# または.zshrc/.bashrcに追加
echo 'export MTL_DEBUG_LAYER=0' >> ~/.zshrc
```

### 技術的詳細 (Technical Details)

**関連コード:**
- `src/hybrid_f32/tensor/core.rs:539-564` - Metal matmul実装
- `src/gpu/metal_kernels.rs:1776` - MetalKernelExecutor初期化
- `Cargo.toml:178` - `hybrid-f32 = ["metal"]`

**発生メカニズム:**
1. `F32Tensor::matmul`が`#[cfg(feature = "metal")]`ブロックでMetal加速を試行
2. `MetalKernelExecutor::new()`が毎回Metalデバイスを初期化
3. macOSのmetaltracerがコンテキスト追跡の警告を出力
4. Metalフレームワーク自体はARCでリソース管理しており、実際のリークはない

### 長期的解決策 (Long-term Solution)

**Phase 1: デバイスタイプベースの制御**
- `F32Tensor`に`device_type`フィールドを追加
- CPUモード時はMetal加速を無効化

**Phase 2: リソースプーリング**
- `MetalKernelExecutor`をシングルトンまたはスレッドローカルに変更
- デバイス/キューの再利用でコンテキスト作成を削減

**Phase 3: 明示的クリーンアップ**
- `Drop` traitでMetalリソースの明示的解放

## その他の既知の問題

### Token Repetition in Generation
**症状**: モデルが同じトークンを繰り返し生成
**原因**: KVキャッシュまたはAttention maskの問題
**ステータス**: 調査中

### Inference Speed (1 token/sec)
**症状**: 推論速度が遅い（目標: 10 tokens/sec）
**原因**: 未最適化
**ステータス**: Phase 2で対応予定
