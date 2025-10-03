# RusTorch CLI - 要件定義書

## 

## 1. プロジェクト概要

### 1.1 目的
RusTorchライブラリを活用したローカルLLM対話型REPLアプリケーションの開発

### 1.2 ターゲットユーザー
- ローカル環境でLLMを実行したい開発者
- プライバシーを重視するユーザー
- オフライン環境でLLMを使用したいユーザー

### 1.3 主要機能
- ローカルLLMモデルとの対話型チャット
- 複数のバックエンド（CPU/GPU/Neural Engine）対応
- セッション履歴の管理と保存
- インタラクティブなREPLインターフェース

## 2. 機能要件

### 2.1 コマンドライン引数

#### 必須オプション
```bash
rustorch-cli --model <MODEL_PATH> [OPTIONS]
```

#### バックエンド指定
- `--backend cpu`: CPU演算
- `--backend cuda`: NVIDIA CUDA GPU
- `--backend metal`: Apple Metal GPU
- `--backend opencl`: OpenCL GPU
- `--backend hybrid`: ハイブリッドモード（自動選択）
- `--backend hybrid-f32`: F32精度ハイブリッドモード

#### 追加オプション
- `--config <CONFIG_FILE>`: 設定ファイル指定（デフォルト: `~/.rustorch/config.toml`）
- `--log-level <LEVEL>`: ログレベル（trace/debug/info/warn/error）
- `--save-history <FILE>`: 会話履歴保存先
- `--load-history <FILE>`: 会話履歴読み込み
- `--max-tokens <N>`: 最大生成トークン数（デフォルト: 2048）
- `--temperature <F>`: サンプリング温度（デフォルト: 0.7）
- `--top-p <F>`: Top-pサンプリング（デフォルト: 0.9）
- `--help`: ヘルプ表示
- `--version`: バージョン表示

### 2.2 対話インターフェース

#### 入力方式
- **単一行入力**: 通常のプロンプト入力
- **マルチライン入力**: Ctrl+Dで入力終了
- **プロンプト表示**: `You> ` / `Assistant> `
- **進捗表示**: 推論中のプログレスバー

#### 特殊コマンド
- `/exit` または `/quit`: アプリケーション終了
- `/clear`: 会話履歴のクリア
- `/help`: ヘルプ表示
- `/save <FILE>`: 会話履歴を保存
- `/load <FILE>`: 会話履歴を読み込み
- `/model <PATH>`: モデルの再読み込み
- `/backend <TYPE>`: バックエンド切り替え
- `/stats`: 統計情報表示（推論時間、トークン数など）
- `/system <MESSAGE>`: システムプロンプト設定

### 2.3 モデルサポート

#### サポート形式
- **GGUF**: llama.cpp互換形式（優先サポート）
- **Safetensors**: Hugging Face標準形式
- **ONNX**: ONNXランタイム形式
- **MLX**: Apple Silicon専用形式
- **PyTorch**: .pt/.pth形式（変換機能）

#### モデルサイズ
- 1B～70B+ パラメータまで対応
- メモリ使用量に応じた自動量子化サポート

### 2.4 セッション管理

#### 会話履歴
- メモリ内保持（セッション中）
- JSON形式での保存/読み込み
- タイムスタンプ付き履歴
- コンテキストウィンドウ管理

#### 設定ファイル（TOML形式）
```toml
[model]
path = "~/.rustorch/models/llama-7b.gguf"
format = "gguf"

[backend]
default = "metal"
fallback = ["cpu"]

[generation]
max_tokens = 2048
temperature = 0.7
top_p = 0.9
top_k = 40

[interface]
prompt_user = "You> "
prompt_assistant = "Assistant> "
show_progress = true

[logging]
level = "info"
file = "~/.rustorch/logs/cli.log"

[history]
auto_save = true
file = "~/.rustorch/history/latest.json"
max_entries = 1000
```

### 2.5 推論エンジン

#### Transformer実装
- **アーキテクチャ**: Decoder-only Transformer
- **最適化**: Flash Attention対応（条件付き）
- **量子化**: INT8/INT4対応（バックエンド依存）
- **KVキャッシュ**: 効率的なキャッシュ管理

#### トークナイザー
- **推奨**: tokenizers (Hugging Face)
- **代替**: tiktoken, SentencePiece
- **機能**: BPE, WordPiece, Unigram対応

### 2.6 パフォーマンス要件

#### レスポンス時間
- 初回トークン生成: <2秒（7Bモデル、Metal GPU）
- 継続トークン生成: <100ms/token
- モデルロード: <30秒（7Bモデル）

#### メモリ使用量
- ベースライン: モデルサイズ × 1.5倍以内
- ピーク: モデルサイズ × 2倍以内

#### 並列処理
- バッチ推論対応（将来拡張）
- マルチスレッド最適化

## 3. 非機能要件

### 3.1 互換性
- **OS**: macOS (Apple Silicon/Intel), Linux, Windows
- **Rust**: 1.70以上
- **GPU**: CUDA 11.x+, Metal 3.0+, OpenCL 2.0+

### 3.2 エラーハンドリング
- ユーザーフレンドリーなエラーメッセージ
- デバッグモード（`--log-level debug`）
- 自動リカバリー（可能な場合）

### 3.3 セキュリティ
- ローカルモデル実行（外部通信なし）
- 会話履歴の暗号化（オプション）
- モデルファイルの整合性チェック

### 3.4 拡張性
- プラグインシステム（将来拡張）
- カスタムモデルアダプター
- API サーバーモード（将来拡張）

## 4. 制約事項

### 4.1 技術的制約
- RusTorchライブラリのAPI制限
- ハードウェアメモリ制限
- モデル形式の互換性

### 4.2 スコープ外
- クラウドベースのLLM連携
- GUIインターフェース
- マルチユーザー対応
- リアルタイムストリーミング（Phase 1）

## 5. 開発フェーズ

### Phase 1: MVP（最小限の機能）
- 基本的なREPLインターフェース
- CPU/Metalバックエンド対応
- GGUF形式サポート
- 基本的な会話履歴管理

### Phase 2: 拡張機能
- 全バックエンド対応
- 複数モデル形式サポート
- 高度な設定管理
- パフォーマンス最適化

### Phase 3: エンタープライズ機能
- APIサーバーモード
- プラグインシステム
- 詳細な統計/監視機能
- バッチ推論対応

## 6. 成功基準

### 6.1 機能的成功基準
- ✅ ユーザーがローカルLLMと対話できる
- ✅ 複数のバックエンドで動作する
- ✅ 会話履歴の保存/読み込みが可能
- ✅ 特殊コマンドがすべて動作する

### 6.2 非機能的成功基準
- ✅ 初回トークン生成が2秒以内
- ✅ メモリ使用量がモデルサイズの2倍以内
- ✅ エラーが発生してもクラッシュしない
- ✅ ドキュメントが完備されている

## 7. 参考資料

### 7.1 類似プロジェクト
- llama.cpp CLI
- ollama
- GPT4All
- Claude Code CLI（インターフェース参考）

### 7.2 技術資料
- RusTorch documentation
- Hugging Face Transformers
- GGUF specification
- MLX documentation

## 8. 用語集

- **REPL**: Read-Eval-Print Loop（対話型インターフェース）
- **GGUF**: GPT-Generated Unified Format（llama.cpp互換形式）
- **KVキャッシュ**: Key-Value Cache（推論高速化）
- **量子化**: Quantization（モデル圧縮技術）
- **トークナイザー**: テキストをトークンに分割するツール

---

**作成日**: 2025-10-03
**バージョン**: 1.0
**ステータス**: Draft