# バックエンド統合実装計画

**日付**: 2025-10-08
**優先順位**: Metal → mac-hybrid → CUDA → hybrid-f32

## 概要

RusTorchのバックエンド対応を段階的に実装・検証します。現在、トークナイザーの修正が完了し、正しいトークンIDが生成されるようになったため、次は各種バックエンドの統合に集中します。

---

## Phase 1: Metal対応 🥇 【最優先】

**目標**: Apple Metal GPUで推論を実行できるようにする
**環境**: M4 Pro搭載Mac
**期待される成果**: GPU加速による高速推論

### タスク詳細

#### 1.1 現状確認とビルドテスト
```bash
# Metalフィーチャーでビルド
cargo build --release --features metal

# エラーがあれば記録
# 成功すれば次へ
```

**成功基準**:
- ✅ コンパイルエラーなし
- ✅ Metalライブラリがリンクされる
- ✅ バイナリが生成される

#### 1.2 ビルドエラーの調査と修正
**予想される問題**:
- Metal APIの変更による非互換性
- 依存関係の不足
- Rust Metal bindingのバージョン問題

**修正アプローチ**:
1. エラーメッセージを詳細に分析
2. Metal crateのドキュメント確認
3. 必要に応じてCargo.toml更新
4. コードの修正（API変更対応）

#### 1.3 GPU推論の動作確認
```bash
# Q4_K_Mモデルでテスト
rustorch-cli \
  -m ~/.rustorch/models/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -b metal
```

**検証項目**:
- ✅ Metal GPUが認識される
- ✅ モデルがGPUにロードされる
- ✅ 推論が実行される
- ✅ 出力が正しい（llama.cppと一致）
- ✅ CPU版より高速

#### 1.4 全量子化フォーマットでテスト
| Format | テスト | 結果 | 備考 |
|--------|--------|------|------|
| Q4_K_M | ⏳ | - | 基本テスト |
| Q5_K_M | ⏳ | - | 中間精度 |
| Q6_K | ⏳ | - | 修正済み |
| Q8_0 | ⏳ | - | 未実装（別途対応） |

**期待される結果**:
- 各フォーマットで正しいトークン生成
- GPU使用率の確認
- パフォーマンスベンチマーク

### 成果物
- [ ] Metal対応の動作確認レポート
- [ ] パフォーマンスベンチマーク結果
- [ ] 発見した問題と修正内容のドキュメント

---

## Phase 2: mac-hybrid対応 🥈

**目標**: Metal + CoreMLの自動選択システムを実装
**前提条件**: Phase 1（Metal）完了

### タスク詳細

#### 2.1 mac-hybridフィーチャーの確認
```toml
# Cargo.toml
mac-hybrid = ["metal", "coreml", "hybrid-f32"]
```

**確認事項**:
- Metal統合の状態
- CoreML統合の実装
- 自動選択ロジックの有無

#### 2.2 Metal + CoreML統合
**実装内容**:
1. CoreMLバックエンドの初期化
2. Metalバックエンドとの切り替え
3. レイヤーごとの最適バックエンド選択

**設計方針**:
```rust
// 疑似コード
match operation_type {
    MatMul => Metal,           // GPU得意
    Convolution => CoreML,     // Neural Engine得意
    RMSNorm => Metal,          // GPU得意
    _ => Metal,                // デフォルト
}
```

#### 2.3 自動バックエンド選択の実装
**選択基準**:
- モデルサイズ
- 演算タイプ
- 利用可能なリソース
- パフォーマンス測定結果

**実装**:
```rust
pub fn select_backend(config: &ModelConfig) -> Backend {
    if is_coreml_available() && config.supports_coreml() {
        Backend::CoreML
    } else if is_metal_available() {
        Backend::Metal
    } else {
        Backend::CPU
    }
}
```

#### 2.4 動作確認
```bash
# mac-hybridでビルド
cargo build --release --features mac-hybrid

# テスト実行
rustorch-cli -m model.gguf -b mac-hybrid
```

**検証項目**:
- ✅ 自動選択が動作
- ✅ CoreML/Metal切り替えが適切
- ✅ パフォーマンスが向上

### 成果物
- [ ] mac-hybrid実装レポート
- [ ] バックエンド選択ロジックのドキュメント
- [ ] パフォーマンス比較（Metal単独 vs mac-hybrid）

---

## Phase 3: CUDA対応（ビルドのみ） 🥉

**目標**: CUDAフィーチャーでコンパイルが成功する
**注意**: 実際のGPUがないため、ビルド成功のみを目標とする

### タスク詳細

#### 3.1 CUDA依存関係の確認
```toml
# Cargo.toml
cuda = ["dep:cudarc"]
```

**確認項目**:
- cudarc crateのバージョン
- CUDA Toolkitの要件
- 条件付きコンパイルの設定

#### 3.2 ビルド環境の準備
**オプション**:
1. **Mock実装**: CUDA APIをスタブで置き換え
2. **条件付きコンパイル**: CUDA未検出時はダミー実装
3. **CI環境**: GitHub Actionsでビルドのみテスト

**推奨アプローチ**:
```rust
#[cfg(feature = "cuda")]
mod cuda_backend {
    // 実際のCUDA実装
}

#[cfg(not(feature = "cuda"))]
mod cuda_backend {
    // ダミー実装（コンパイルのみ通す）
}
```

#### 3.3 コンパイル成功の確認
```bash
# CUDAフィーチャーでビルド（エラー許容）
cargo build --release --features cuda 2>&1 | tee cuda_build.log

# 成功基準: コンパイルエラーがない（リンクエラーは許容）
```

**期待される結果**:
- ✅ Rustコードがコンパイル成功
- ⚠️ CUDA libraries not foundの警告（許容）
- ❌ コンパイルエラー（要修正）

### 成果物
- [ ] CUDAビルドログ
- [ ] 発見した問題のリスト
- [ ] 修正が必要なコードの特定

---

## Phase 4: hybrid-f32の修正 🔧

**目標**: hybrid-f32フィーチャーのコンパイルエラーを修正
**前提条件**: Phase 1（Metal）完了

### タスク詳細

#### 4.1 現在のエラーの詳細分析
```bash
# エラー詳細を収集
cargo build --release --features hybrid-f32 2>&1 > hybrid_f32_errors.log
```

**分析項目**:
- エラーの種類（型エラー、trait境界、ライフタイム等）
- 影響範囲（どのファイル・モジュールか）
- 根本原因の特定

#### 4.2 エラー原因の調査
**予想される問題**:
1. **f32/f64型の不一致**
   - `hybrid-f32`はf32を前提
   - 既存コードがf64を使用している箇所

2. **Metal APIとの統合問題**
   - Metalはf32のみサポート
   - 型変換が不足

3. **trait境界の問題**
   - ジェネリクスの制約が不足
   - Floatトレイトの実装不足

#### 4.3 修正の実装
**修正戦略**:

**戦略A: 条件付きコンパイル**
```rust
#[cfg(feature = "hybrid-f32")]
type Float = f32;

#[cfg(not(feature = "hybrid-f32"))]
type Float = f64;
```

**戦略B: ジェネリクス**
```rust
pub struct Tensor<T: Float> {
    data: Vec<T>,
    // ...
}

impl<T: Float> Tensor<T> {
    // f32/f64両対応
}
```

**戦略C: 型変換レイヤー**
```rust
pub fn convert_to_f32(data: &[f64]) -> Vec<f32> {
    data.iter().map(|&x| x as f32).collect()
}
```

#### 4.4 段階的修正
1. **エラー1件ずつ修正**: 一度に全て修正しない
2. **テスト駆動**: 修正後すぐにコンパイル
3. **ドキュメント化**: 各修正の理由を記録

#### 4.5 動作テスト
```bash
# ビルド成功後
cargo build --release --features hybrid-f32

# 動作確認
rustorch-cli -m model.gguf -b hybrid-f32
```

**検証項目**:
- ✅ コンパイル成功
- ✅ Metal GPU使用
- ✅ f32精度で推論実行
- ✅ 出力が正しい
- ✅ パフォーマンスが適切

### 成果物
- [ ] hybrid-f32修正レポート
- [ ] 型システムの設計ドキュメント
- [ ] f32/f64切り替えのベストプラクティス

---

## 全体のマイルストーン

### Week 1: Metal完全対応
- [x] トークナイザー修正完了
- [ ] Metal単独ビルド成功
- [ ] Metal GPU推論動作確認
- [ ] 全量子化フォーマットテスト

### Week 2: ハイブリッドシステム
- [ ] mac-hybrid実装
- [ ] 自動バックエンド選択
- [ ] パフォーマンスベンチマーク

### Week 3: ビルド対応
- [ ] CUDAビルド成功
- [ ] hybrid-f32エラー修正
- [ ] 全バックエンドドキュメント完成

---

## 技術的課題と解決策

### 課題1: Metal API変更への対応
**解決策**:
- Metal crateの最新ドキュメント参照
- サンプルコードから学習
- 必要に応じてcrateバージョン更新

### 課題2: f32/f64型の統一
**解決策**:
- ジェネリクスによる柔軟な設計
- 条件付きコンパイルの活用
- 型変換レイヤーの実装

### 課題3: パフォーマンス測定
**解決策**:
- ベンチマークフレームワークの構築
- 各バックエンドでの比較テスト
- プロファイリングツールの活用

---

## 次のステップ

**今すぐ開始**: Phase 1.1 - Metal現状確認とビルドテスト

```bash
# 実行コマンド
cargo build --release --features metal
```

このドキュメントは進捗に応じて更新します。
