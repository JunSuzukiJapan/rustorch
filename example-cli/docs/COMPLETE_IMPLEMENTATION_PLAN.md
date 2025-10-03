# RusTorch CLI 完全実装計画

## 📋 目標

**現状**: 70% 実装完了（Phase 1-6）
**目標**: 100% 実装完了（REQUIREMENTS.md の全要件達成）

このドキュメントは、未実装機能すべてを体系的に実装するためのロードマップです。

---

## 🎯 Phase 7-12 実装ロードマップ

### Phase 7: 設定管理システム 🔴 高優先度

**目的**: TOML設定ファイルによる永続的な設定管理

**実装内容**:
- `~/.rustorch/config.toml` 自動読み込み機能
- 設定ファイルのパース・バリデーション
- コマンドライン引数との優先順位制御（CLI > 環境変数 > 設定ファイル > デフォルト）
- 設定の保存機能（`:config save` コマンド）

**新規ファイル**:
- `src/utils/config.rs` - 設定管理モジュール
- `src/utils/config/loader.rs` - TOML読み込み
- `src/utils/config/schema.rs` - 設定スキーマ定義

**設定ファイル例**:
```toml
[model]
default = "models/llama-7b.gguf"
cache_dir = "~/.rustorch/models"

[generation]
max_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 40

[backend]
default = "metal"
fallback = "cpu"

[session]
auto_save = true
history_file = "~/.rustorch/history"
max_history = 1000

[ui]
color = true
stream = true
show_metrics = false
```

**依存関係**:
- `toml = "0.8"` - TOML パース
- `dirs = "5.0"` - ホームディレクトリ取得

**推定工数**: 2-3時間
**優先度**: 🔴 高（ユーザビリティ大幅向上）

**完了条件**:
- ✅ TOML設定ファイルの自動読み込み
- ✅ 全設定項目のパース・バリデーション
- ✅ CLI引数との優先順位制御
- ✅ `:config save` コマンド実装
- ✅ 設定ファイルのテストカバレッジ 80%+

---

### Phase 8: コマンドライン引数完全実装 🟡 中優先度

**目的**: `--save-history` / `--load-history` フラグ追加

**実装内容**:
- `--save-history <path>` - セッション終了時にヒストリー保存
- `--load-history <path>` - 起動時にヒストリー読み込み
- 自動ヒストリー保存機能（Ctrl+C時も対応）

**変更ファイル**:
- `src/cli/args.rs` - フラグ追加
- `src/cli/repl.rs` - ヒストリー保存・読み込み処理
- `src/session/mod.rs` - ヒストリー管理機能

**推定工数**: 1-2時間
**優先度**: 🟡 中（REPLコマンドで代替可能だが、CLI完全性のため実装推奨）

**完了条件**:
- ✅ `--save-history` フラグ動作
- ✅ `--load-history` フラグ動作
- ✅ 自動保存機能（Ctrl+C ハンドリング）
- ✅ ヒストリーファイル形式のドキュメント化

---

### Phase 9: モデルフォーマット拡張 🟡 中優先度

**目的**: MLX および PyTorch フォーマット対応

**実装内容**:

#### MLX フォーマット対応
- MLX モデルファイル（.mlx）のパース
- メタデータ抽出
- RusTorch Tensor への変換

**新規ファイル**:
- `src/model/formats/mlx.rs` - MLX ローダー

**依存関係**:
- MLX バイナリフォーマット仕様の調査
- 可能であれば `mlx-rs` クレート使用

#### PyTorch フォーマット対応
- `.pt` / `.pth` ファイルのパース
- `pickle` 形式のデシリアライズ
- state_dict 抽出

**新規ファイル**:
- `src/model/formats/pytorch.rs` - PyTorch ローダー

**依存関係**:
- `serde-pickle = "1.1"` - Pickle デシリアライズ
- `pyo3 = "0.20"` (オプション) - Python interop

**推定工数**: 3-4時間
**優先度**: 🟡 中（エコシステム互換性向上）

**完了条件**:
- ✅ MLX モデル読み込み成功
- ✅ PyTorch モデル読み込み成功
- ✅ 自動フォーマット検出機能更新
- ✅ 各フォーマットのテストケース追加

---

### Phase 10: 実モデル推論実装 🔴 高優先度

**目的**: RusTorch API を使用した実際のモデル推論機能

**実装内容**:

#### Transformer モデル実装
- Multi-Head Attention 層
- Feed-Forward 層
- Layer Normalization
- Positional Encoding
- GPT/LLaMA アーキテクチャ

**新規ファイル**:
- `src/model/architectures/mod.rs` - アーキテクチャ定義
- `src/model/architectures/transformer.rs` - Transformer 実装
- `src/model/architectures/gpt.rs` - GPT モデル
- `src/model/architectures/llama.rs` - LLaMA モデル

#### トークナイザー統合
- BPE トークナイザー
- SentencePiece トークナイザー
- Hugging Face tokenizers 統合

**依存関係**:
- `tokenizers = "0.15"` - Hugging Face tokenizers
- RusTorch の `nn::Module`, `Tensor` API

**推論エンジン改修**:
- `src/model/inference.rs` の完全実装
- ダミー実装からRusTorch APIへの移行
- KV-Cache 実装（推論高速化）
- Top-k / Top-p / Temperature サンプリング

**推定工数**: 8-10時間
**優先度**: 🔴 高（コア機能）

**完了条件**:
- ✅ GPT-2 モデルで実推論成功
- ✅ LLaMA モデルで実推論成功
- ✅ トークナイザー正常動作
- ✅ 生成品質の検証（coherent text generation）
- ✅ 推論速度計測（tokens/sec）

---

### Phase 11: バックエンド最適化 🟡 中優先度

**目的**: Metal/CUDA/OpenCL バックエンドの完全統合

**実装内容**:

#### Metal バックエンド（macOS）
- RusTorch Metal API 統合
- GPU メモリ管理
- Metal Performance Shaders (MPS) 活用

**変更ファイル**:
- `src/backend/metal.rs` - Metal 実装完全化
- `src/backend/device.rs` - デバイス管理強化

#### CUDA バックエンド（NVIDIA GPU）
- CUDA カーネル統合
- cuBLAS / cuDNN 活用
- Multi-GPU 対応

**新規ファイル**:
- `src/backend/cuda.rs` - CUDA バックエンド

**依存関係**:
- RusTorch の CUDA feature フラグ

#### OpenCL バックエンド（汎用GPU）
- OpenCL カーネル統合
- クロスプラットフォーム対応

**新規ファイル**:
- `src/backend/opencl.rs` - OpenCL バックエンド

#### バックエンド自動選択
- ハードウェア検出
- 最適バックエンド自動選択
- フォールバック機能

**変更ファイル**:
- `src/backend/mod.rs` - 自動選択ロジック

**推定工数**: 6-8時間
**優先度**: 🟡 中（パフォーマンス向上）

**完了条件**:
- ✅ Metal バックエンドで推論成功（macOS）
- ✅ CUDA バックエンドで推論成功（Linux/Windows）
- ✅ OpenCL バックエンドで推論成功（汎用）
- ✅ 自動バックエンド選択機能動作
- ✅ バックエンド間のパフォーマンス比較ベンチマーク

---

### Phase 12: パフォーマンス計測・最適化 🟢 低優先度

**目的**: 要件定義のパフォーマンス基準達成

**実装内容**:

#### レスポンス時間計測
- 初回トークン生成時間（Time to First Token: TTFT）
- トークン生成速度（tokens/sec）
- エンドツーエンドレスポンス時間

**新規ファイル**:
- `src/metrics/mod.rs` - メトリクス収集
- `src/metrics/timing.rs` - 時間計測
- `src/metrics/reporter.rs` - レポート生成

#### メモリ使用量モニタリング
- ヒープメモリ使用量
- GPU メモリ使用量
- モデルサイズ vs 実際のメモリ使用量

**依存関係**:
- `sysinfo = "0.30"` - システムメトリクス

#### パフォーマンスベンチマーク
- 標準ベンチマークスイート作成
- 各バックエンドでの性能比較
- モデルサイズ別のベンチマーク

**新規ファイル**:
- `benches/inference_benchmark.rs` - 推論ベンチマーク
- `benches/backend_comparison.rs` - バックエンド比較

#### 最適化実施
- プロファイリング（`perf`, `instruments`）
- ボトルネック特定
- 最適化実装
  - KV-Cache 最適化
  - メモリアロケーション最適化
  - バッチ処理最適化

**推定工数**: 4-6時間
**優先度**: 🟢 低（機能実装後の最適化フェーズ）

**完了条件**:
- ✅ TTFT < 200ms（7B モデル、Metal/CUDA）
- ✅ 生成速度 > 20 tokens/sec（7B モデル）
- ✅ メモリ使用量 < モデルサイズ × 1.5
- ✅ ベンチマークレポート作成
- ✅ 最適化前後のパフォーマンス比較ドキュメント

---

## 📊 実装スケジュール概要

| Phase | 内容 | 工数 | 優先度 | 依存関係 |
|-------|------|------|--------|----------|
| Phase 7 | 設定管理システム | 2-3h | 🔴 高 | なし |
| Phase 8 | CLI引数完全実装 | 1-2h | 🟡 中 | なし |
| Phase 9 | モデルフォーマット拡張 | 3-4h | 🟡 中 | なし |
| Phase 10 | 実モデル推論実装 | 8-10h | 🔴 高 | Phase 9 |
| Phase 11 | バックエンド最適化 | 6-8h | 🟡 中 | Phase 10 |
| Phase 12 | パフォーマンス計測 | 4-6h | 🟢 低 | Phase 10, 11 |

**総推定工数**: 24-33時間

---

## 🎯 推奨実装順序

### ステージ1: 基盤強化（優先実装）
1. **Phase 7** - 設定管理システム（2-3h）
2. **Phase 8** - CLI引数完全実装（1-2h）

**理由**: ユーザビリティ向上、他Phase への影響なし、早期に完了可能

### ステージ2: コア機能完成（重点実装）
3. **Phase 9** - モデルフォーマット拡張（3-4h）
4. **Phase 10** - 実モデル推論実装（8-10h）

**理由**: コア機能、Phase 11/12 の前提条件

### ステージ3: 最適化（後続実装）
5. **Phase 11** - バックエンド最適化（6-8h）
6. **Phase 12** - パフォーマンス計測（4-6h）

**理由**: 機能完成後の最適化フェーズ、計測可能性の確保

---

## 🔄 各Phase の成果物

### Phase 7
- ✅ `src/utils/config.rs` - 設定管理モジュール
- ✅ TOML設定ファイル読み込み機能
- ✅ ドキュメント更新（設定ファイル仕様）

### Phase 8
- ✅ `--save-history` / `--load-history` 実装
- ✅ 自動ヒストリー保存機能
- ✅ CLI引数完全性達成

### Phase 9
- ✅ `src/model/formats/mlx.rs`
- ✅ `src/model/formats/pytorch.rs`
- ✅ モデルフォーマット網羅性達成

### Phase 10
- ✅ `src/model/architectures/` - Transformer 実装
- ✅ 実推論機能（GPT-2, LLaMA）
- ✅ トークナイザー統合

### Phase 11
- ✅ Metal/CUDA/OpenCL 完全統合
- ✅ バックエンド自動選択機能
- ✅ パフォーマンスベンチマーク

### Phase 12
- ✅ メトリクス収集システム
- ✅ パフォーマンスレポート
- ✅ 最適化実施・検証

---

## 📈 実装完了後の達成状態

### 機能面
- ✅ すべての要件定義項目 100% 実装
- ✅ GGUF/Safetensors/ONNX/MLX/PyTorch 全フォーマット対応
- ✅ CPU/Metal/CUDA/OpenCL 全バックエンド対応
- ✅ 実モデル推論（GPT-2, LLaMA）
- ✅ TOML設定ファイル完全対応

### 品質面
- ✅ テストカバレッジ 80%+
- ✅ Clippy warnings 0件
- ✅ ドキュメント完全性
- ✅ パフォーマンス要件達成

### ユーザビリティ面
- ✅ 設定ファイルによる永続設定
- ✅ 自動ヒストリー管理
- ✅ バックエンド自動選択
- ✅ 包括的なエラーメッセージ

---

## 🚀 次のアクション

1. **Phase 7 開始準備**
   - `toml` および `dirs` クレート追加
   - 設定スキーマ設計
   - テストケース設計

2. **実装開始判断**
   - 優先度の確認
   - リソース割り当て
   - マイルストーン設定

**推奨**: Phase 7（設定管理システム）から着手し、段階的に実装を進める。

---

**作成日**: 2025年10月3日
**バージョン**: 1.0
**ステータス**: 実装計画確定
