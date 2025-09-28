# F32統一ハイブリッド実験的実装計画

## 概要

f32精度に統一したハイブリッド実行システムの実験的実装を行い、変換コスト削減効果とNeural Engine性能影響を実測評価します。

**ブランチ**: `experimental/hybrid-f32`

## 実装目標

### 主要目標
1. **変換コスト完全削除**: f32テンソルでの変換レス実行
2. **統一メモリ管理**: Metal-Neural Engine間のゼロコピー転送
3. **性能測定**: 現行システムとの詳細比較

### 検証項目
- 変換コスト削減効果（10-30%期待）
- Neural Engine f32性能（Float16比50-60%性能の確認）
- メモリ使用量削減効果
- 開発・デバッグ効率改善

## アーキテクチャ設計

### 1. F32Tensor専用構造体

```rust
/// f32専用テンソル（変換コスト最小化）
pub struct F32Tensor {
    /// CPU側データ
    data: Array<f32, IxDyn>,

    /// GPU共有バッファ（Metal用）
    metal_buffer: Option<Arc<MetalBuffer>>,

    /// Neural Engine共有バッファ（CoreML用）
    coreml_buffer: Option<Arc<MLMultiArray>>,

    /// デバイス最適化状態
    device_state: DeviceState,

    /// 勾配追跡
    requires_grad: bool,
}

#[derive(Debug, Clone)]
enum DeviceState {
    CPU,
    Metal { device_id: usize },
    CoreML { device_id: usize },
    Synchronized, // 全デバイス同期済み
}
```

### 2. ゼロコピー転送機構

```rust
/// f32統一バッファ管理
pub struct F32UnifiedBuffer {
    /// 共有メモリプール
    memory_pool: Arc<Mutex<F32MemoryPool>>,

    /// デバイス間同期機構
    sync_manager: F32SyncManager,
}

impl F32UnifiedBuffer {
    /// Metal-Neural Engine間の直接転送
    fn transfer_metal_to_coreml(&self, tensor: &mut F32Tensor) -> Result<()> {
        // Metal GPU → Neural Engine直接転送（変換なし）
    }

    /// 全デバイス同期
    fn synchronize_all(&self, tensor: &mut F32Tensor) -> Result<()> {
        // CPU ↔ Metal ↔ Neural Engine同期
    }
}
```

### 3. 統一実行エンジン

```rust
/// f32統一ハイブリッド実行エンジン
pub struct F32HybridExecutor {
    metal_executor: MetalKernelExecutor,
    coreml_executor: CoreMLGraph,
    device_selector: F32DeviceSelector,
}

impl F32HybridExecutor {
    /// 最適デバイス選択（f32専用）
    fn select_optimal_device(&self, op: &F32Operation) -> DeviceType {
        match op {
            F32Operation::MatMul { size } if *size > 10000 => DeviceType::Metal(0),
            F32Operation::Conv2D { .. } => DeviceType::CoreML(0), // f32で実行
            F32Operation::Activation { .. } => DeviceType::CoreML(0),
            _ => DeviceType::CPU,
        }
    }

    /// 統一実行（変換コストなし）
    fn execute(&self, op: F32Operation) -> Result<F32Tensor> {
        let device = self.select_optimal_device(&op);
        match device {
            DeviceType::Metal(_) => self.execute_metal_f32(op),
            DeviceType::CoreML(_) => self.execute_coreml_f32(op),
            DeviceType::CPU => self.execute_cpu_f32(op),
        }
    }
}
```

## 実装フェーズ

### Phase 1: 基盤実装（Week 1-2）

**1.1 F32Tensor構造体**
- ファイル: `src/tensor/f32_tensor.rs`
- 基本データ構造とAPI
- CPU実行の実装

**1.2 変換レスMetal実行**
- ファイル: `src/gpu/f32_metal.rs`
- Metal GPU直接実行パス
- バッファ管理の統一

**1.3 基本テスト**
- ファイル: `tests/f32_hybrid_tests.rs`
- 正確性検証
- 基本性能測定

### Phase 2: Neural Engine統合（Week 3-4）

**2.1 CoreML f32実行パス**
- ファイル: `src/gpu/coreml/f32_backend.rs`
- Neural Engine f32専用実行
- MLMultiArray最適化

**2.2 デバイス間転送**
- ファイル: `src/gpu/f32_transfer.rs`
- ゼロコピー転送機構
- 同期メカニズム

**2.3 統合テスト**
- Metal ↔ Neural Engine転送テスト
- マルチデバイス実行検証

### Phase 3: 最適化とベンチマーク（Week 5-6）

**3.1 性能最適化**
- メモリプール最適化
- 同期オーバーヘッド削減
- キャッシュ機構

**3.2 包括的ベンチマーク**
- 現行システムとの比較
- ワークロード別性能測定
- メモリ使用量分析

**3.3 レポート作成**
- 性能改善効果の定量化
- 実装コスト評価

## 実装優先順位

### High Priority
1. **F32Tensor基本構造** - 全体の基盤
2. **Metal直接実行** - 最大の効果期待
3. **性能測定基盤** - 効果検証必須

### Medium Priority
1. **Neural Engine f32実行** - 性能劣化の確認
2. **ゼロコピー転送** - 効率改善
3. **デバイス選択最適化** - 総合性能向上

### Low Priority
1. **自動勾配対応** - 後続実装可能
2. **複雑なメモリ最適化** - 基本効果確認後
3. **完全な型統合** - 概念実証後検討

## 成功指標

### 性能指標
- **変換コスト削減**: 15-25%削減目標
- **メモリ効率**: 20-30%改善目標
- **Neural Engine性能**: 現状維持（Float16比60%以上）

### 実装品質指標
- **テストカバレッジ**: 90%以上
- **ベンチマーク網羅性**: 主要ワークロード100%
- **コード品質**: clippy警告0件

## リスク管理

### 技術リスク
- **Neural Engine性能劣化**: Float16→f32で予想以上の性能低下
- **メモリ同期複雑性**: デバイス間同期の実装困難
- **互換性問題**: 既存システムとの統合課題

### 対策
- **段階的実装**: 各フェーズでの効果検証
- **fallback機構**: 現行システムへの自動フォールバック
- **詳細測定**: 各段階での性能詳細分析

## 期待される成果

### 短期成果（6週間）
- f32統一ハイブリッドのプロトタイプ
- 変換コスト削減効果の定量化
- Neural Engine f32性能の実測値

### 中期成果（3-6ヶ月）
- 実用レベルの実装
- 包括的性能比較レポート
- 本流への統合判断材料

### 長期成果（6-12ヶ月）
- 次世代ハイブリッドシステムの設計指針
- Apple Silicon最適化のベストプラクティス
- RusTorch v0.7での統合検討材料

## 次のステップ

1. **Phase 1開始**: F32Tensor基本構造の実装
2. **Metal統合**: 変換レス実行パスの構築
3. **初期ベンチマーク**: 基本効果の確認

この実験により、f32統一ハイブリッドの実用性と、RusTorchの次世代アーキテクチャ方針を明確化します。