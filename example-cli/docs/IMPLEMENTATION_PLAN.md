# RusTorch CLI - 実装計画書

## 1. アーキテクチャ設計

### 1.1 モジュール構成

```
example-cli/
├── Cargo.toml                 # プロジェクト設定
├── README.md                  # ユーザードキュメント
├── REQUIREMENTS.md            # 要件定義書
├── IMPLEMENTATION_PLAN.md     # 本ドキュメント
├── src/
│   ├── main.rs               # エントリーポイント
│   ├── cli/
│   │   ├── mod.rs            # CLIモジュール
│   │   ├── args.rs           # コマンドライン引数解析
│   │   ├── repl.rs           # REPLインターフェース
│   │   └── commands.rs       # 特殊コマンド処理
│   ├── model/
│   │   ├── mod.rs            # モデルモジュール
│   │   ├── loader.rs         # モデルローダー
│   │   ├── inference.rs      # 推論エンジン
│   │   ├── transformer.rs    # Transformer実装
│   │   └── formats/
│   │       ├── mod.rs        # フォーマット管理
│   │       ├── gguf.rs       # GGUF形式
│   │       ├── safetensors.rs # Safetensors形式
│   │       ├── onnx.rs       # ONNX形式
│   │       └── mlx.rs        # MLX形式
│   ├── tokenizer/
│   │   ├── mod.rs            # トークナイザーモジュール
│   │   ├── hf.rs             # Hugging Face tokenizers
│   │   └── bpe.rs            # BPE実装
│   ├── session/
│   │   ├── mod.rs            # セッション管理
│   │   ├── history.rs        # 会話履歴
│   │   └── config.rs         # 設定管理
│   ├── backend/
│   │   ├── mod.rs            # バックエンド管理
│   │   ├── cpu.rs            # CPU実装
│   │   ├── cuda.rs           # CUDA実装
│   │   ├── metal.rs          # Metal実装
│   │   ├── opencl.rs         # OpenCL実装
│   │   └── hybrid.rs         # ハイブリッド実装
│   ├── utils/
│   │   ├── mod.rs            # ユーティリティ
│   │   ├── progress.rs       # プログレスバー
│   │   ├── logger.rs         # ロギング
│   │   └── error.rs          # エラーハンドリング
│   └── lib.rs                # ライブラリエクスポート
├── tests/
│   ├── integration_tests.rs  # 統合テスト
│   ├── model_tests.rs        # モデルテスト
│   └── cli_tests.rs          # CLIテスト
└── examples/
    └── basic_usage.rs        # 基本使用例
```

### 1.2 依存関係

#### 必須依存
```toml
[dependencies]
# RusTorchコア
rustorch = { path = "..", default-features = false }

# CLI
clap = { version = "4.5", features = ["derive", "cargo", "env"] }
rustyline = "14.0"  # REPL入力

# トークナイザー
tokenizers = "0.19"

# モデル形式
safetensors = "0.4"
gguf-rs = "0.1"  # または自前実装
candle-core = "0.7"  # ONNXサポート
mlx-rs = { version = "0.1", optional = true }  # Apple Silicon

# シリアライゼーション
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"

# ロギング・エラー
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1.0"
thiserror = "1.0"

# プログレスバー
indicatif = "0.17"

# 非同期処理
tokio = { version = "1.40", features = ["rt", "macros"], optional = true }
```

#### フィーチャーフラグ
```toml
[features]
default = ["cpu"]
cpu = ["rustorch/cpu"]
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"]
opencl = ["rustorch/opencl"]
hybrid = ["rustorch/hybrid"]
hybrid-f32 = ["rustorch/hybrid-f32"]
mlx = ["dep:mlx-rs"]
async = ["dep:tokio"]
```

## 2. 実装フェーズ

### Phase 1: 基盤構築（Week 1）

#### 1.1 プロジェクトセットアップ
- [ ] Cargo.tomlの作成
- [ ] ディレクトリ構造の構築
- [ ] 基本的なCI/CD設定

#### 1.2 CLIインターフェース
- [ ] clap による引数解析実装
- [ ] 基本的なREPL実装（rustyline）
- [ ] エラーハンドリング基盤

#### 1.3 設定管理
- [ ] TOML設定ファイル読み込み
- [ ] デフォルト設定の定義
- [ ] 環境変数サポート

### Phase 2: モデルサポート（Week 2-3）

#### 2.1 トークナイザー統合
- [ ] Hugging Face tokenizers統合
- [ ] トークン化/逆トークン化API
- [ ] 特殊トークン処理

#### 2.2 モデルローダー（GGUF優先）
- [ ] GGUF形式パーサー実装
- [ ] メタデータ読み込み
- [ ] 重み読み込みとテンソル変換
- [ ] 量子化サポート（INT8/INT4）

#### 2.3 Transformer推論エンジン
- [ ] Decoder-only Transformer実装
- [ ] アテンションメカニズム
- [ ] KVキャッシュ管理
- [ ] 生成ループ実装

### Phase 3: バックエンド統合（Week 3-4）

#### 3.1 CPU実装
- [ ] RusTorchテンソル演算統合
- [ ] SIMD最適化
- [ ] マルチスレッド対応

#### 3.2 Metal実装（macOS）
- [ ] Metal GPU統合
- [ ] カーネル最適化
- [ ] メモリ管理

#### 3.3 その他バックエンド
- [ ] CUDA実装（Linux/Windows）
- [ ] OpenCL実装（汎用）
- [ ] ハイブリッドモード実装

### Phase 4: REPL機能拡張（Week 4-5）

#### 4.1 特殊コマンド実装
- [ ] `/exit`, `/quit`: 終了
- [ ] `/clear`: 履歴クリア
- [ ] `/help`: ヘルプ表示
- [ ] `/save`, `/load`: 履歴管理
- [ ] `/model`: モデル切り替え
- [ ] `/backend`: バックエンド切り替え
- [ ] `/stats`: 統計表示
- [ ] `/system`: システムプロンプト設定

#### 4.2 セッション管理
- [ ] 会話履歴の保持
- [ ] JSON形式での保存/読み込み
- [ ] コンテキストウィンドウ管理
- [ ] 自動保存機能

#### 4.3 UI/UX改善
- [ ] プログレスバー表示
- [ ] トークンストリーミング表示
- [ ] カラー出力対応
- [ ] マルチライン入力サポート

### Phase 5: 追加モデル形式（Week 5-6）

#### 5.1 Safetensors対応
- [ ] Safetensors読み込み
- [ ] Hugging Face互換性

#### 5.2 ONNX対応
- [ ] ONNX Runtime統合
- [ ] モデル変換ツール

#### 5.3 MLX対応（Apple Silicon）
- [ ] MLX Rust bindings統合
- [ ] Neural Engine最適化

### Phase 6: テスト・最適化（Week 6-7）

#### 6.1 テストカバレッジ
- [ ] 単体テスト（80%以上）
- [ ] 統合テスト
- [ ] エンドツーエンドテスト
- [ ] パフォーマンステスト

#### 6.2 最適化
- [ ] プロファイリング
- [ ] メモリ使用量最適化
- [ ] 推論速度最適化
- [ ] バイナリサイズ削減

#### 6.3 ドキュメント
- [ ] README作成
- [ ] APIドキュメント
- [ ] 使用例の充実
- [ ] トラブルシューティングガイド

## 3. 技術詳細

### 3.1 GGUF形式パーサー実装

```rust
// example-cli/src/model/formats/gguf.rs

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use rustorch::Tensor;

pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GGUFMetadata {
    pub key: String,
    pub value: GGUFValue,
}

pub enum GGUFValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum GGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    // ... その他の量子化形式
}

pub struct GGUFLoader {
    reader: BufReader<File>,
    header: GGUFHeader,
    metadata: Vec<GGUFMetadata>,
    tensors: Vec<GGUFTensorInfo>,
}

impl GGUFLoader {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)
            .context("Failed to open GGUF file")?;
        let mut reader = BufReader::new(file);

        let header = Self::read_header(&mut reader)?;
        let metadata = Self::read_metadata(&mut reader, header.metadata_kv_count)?;
        let tensors = Self::read_tensor_info(&mut reader, header.tensor_count)?;

        Ok(Self { reader, header, metadata, tensors })
    }

    fn read_header(reader: &mut BufReader<File>) -> Result<GGUFHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if &magic != b"GGUF" {
            anyhow::bail!("Invalid GGUF magic number");
        }

        let version = Self::read_u32(reader)?;
        let tensor_count = Self::read_u64(reader)?;
        let metadata_kv_count = Self::read_u64(reader)?;

        Ok(GGUFHeader {
            magic: u32::from_le_bytes(magic),
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    pub fn load_tensor(&mut self, name: &str) -> Result<Tensor> {
        // テンソルデータの読み込みと変換
        todo!("Implement tensor loading")
    }

    // ヘルパーメソッド
    fn read_u32(reader: &mut BufReader<File>) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(reader: &mut BufReader<File>) -> Result<u64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
}
```

### 3.2 Transformer推論エンジン

```rust
// example-cli/src/model/transformer.rs

use rustorch::{Tensor, nn};
use anyhow::Result;

pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
}

pub struct DecoderLayer {
    self_attn: MultiHeadAttention,
    mlp: MLP,
    ln1: LayerNorm,
    ln2: LayerNorm,
}

pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
}

pub struct KVCache {
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_cache: vec![],
            v_cache: vec![],
            seq_len: 0,
        }
    }

    pub fn update(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        // KVキャッシュの更新
        todo!()
    }

    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.k_cache.len() {
            Some((&self.k_cache[layer_idx], &self.v_cache[layer_idx]))
        } else {
            None
        }
    }
}

pub struct TransformerModel {
    config: TransformerConfig,
    embed_tokens: nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: nn::Linear,
}

impl TransformerModel {
    pub fn new(config: TransformerConfig) -> Self {
        // モデル初期化
        todo!()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        // フォワードパス
        let hidden_states = self.embed_tokens.forward(input_ids)?;

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                kv_cache.as_mut().map(|c| (idx, c)),
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    pub fn generate(
        &self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();
        let mut kv_cache = KVCache::new(self.config.num_layers);

        for _ in 0..max_tokens {
            let input_tensor = Tensor::from_vec(
                generated.clone(),
                &[1, generated.len()],
            )?;

            let logits = self.forward(&input_tensor, Some(&mut kv_cache))?;
            let next_token = self.sample_token(&logits, temperature, top_p)?;

            generated.push(next_token);

            // EOS チェック
            if next_token == self.config.eos_token_id {
                break;
            }
        }

        Ok(generated)
    }

    fn sample_token(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        // Top-p サンプリング実装
        todo!()
    }
}
```

### 3.3 REPLインターフェース

```rust
// example-cli/src/cli/repl.rs

use rustyline::{Editor, Result as RLResult};
use rustyline::error::ReadlineError;
use crate::model::TransformerModel;
use crate::session::SessionManager;
use indicatif::{ProgressBar, ProgressStyle};

pub struct REPL {
    model: TransformerModel,
    session: SessionManager,
    editor: Editor<()>,
}

impl REPL {
    pub fn new(model: TransformerModel, session: SessionManager) -> Self {
        let editor = Editor::<()>::new();
        Self { model, session, editor }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        println!("RusTorch CLI - Local LLM Chat");
        println!("Type '/help' for available commands, '/exit' to quit.\n");

        loop {
            match self.editor.readline("You> ") {
                Ok(line) => {
                    self.editor.add_history_entry(&line);

                    if line.starts_with('/') {
                        if !self.handle_command(&line)? {
                            break; // /exit または /quit
                        }
                    } else {
                        self.handle_message(&line)?;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {}", err);
                    break;
                }
            }
        }

        Ok(())
    }

    fn handle_command(&mut self, command: &str) -> anyhow::Result<bool> {
        let parts: Vec<&str> = command.split_whitespace().collect();

        match parts[0] {
            "/exit" | "/quit" => {
                println!("Goodbye!");
                return Ok(false);
            }
            "/help" => self.show_help(),
            "/clear" => self.session.clear_history(),
            "/save" => {
                let path = parts.get(1).unwrap_or(&"history.json");
                self.session.save_history(path)?;
            }
            "/load" => {
                let path = parts.get(1).unwrap_or(&"history.json");
                self.session.load_history(path)?;
            }
            "/stats" => self.show_stats(),
            _ => println!("Unknown command: {}", parts[0]),
        }

        Ok(true)
    }

    fn handle_message(&mut self, message: &str) -> anyhow::Result<()> {
        self.session.add_user_message(message);

        // プ��グレスバー表示
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message("Thinking...");

        // 推論実行
        let response = self.generate_response(message)?;

        pb.finish_and_clear();

        println!("Assistant> {}", response);
        self.session.add_assistant_message(&response);

        Ok(())
    }

    fn generate_response(&self, message: &str) -> anyhow::Result<String> {
        // トークン化
        let input_ids = self.session.tokenize(message)?;

        // 生成
        let output_ids = self.model.generate(
            input_ids,
            self.session.config.max_tokens,
            self.session.config.temperature,
            self.session.config.top_p,
        )?;

        // デコード
        let response = self.session.decode(&output_ids)?;

        Ok(response)
    }

    fn show_help(&self) {
        println!("Available commands:");
        println!("  /exit, /quit       - Exit the application");
        println!("  /clear             - Clear conversation history");
        println!("  /help              - Show this help message");
        println!("  /save [FILE]       - Save conversation history");
        println!("  /load [FILE]       - Load conversation history");
        println!("  /stats             - Show statistics");
    }

    fn show_stats(&self) {
        println!("Statistics:");
        println!("  Messages: {}", self.session.message_count());
        println!("  Tokens: {}", self.session.total_tokens());
        println!("  Backend: {}", self.session.backend_name());
    }
}
```

### 3.4 セッション管理

```rust
// example-cli/src/session/mod.rs

use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::fs;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionHistory {
    pub messages: Vec<Message>,
}

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

pub struct SessionManager {
    pub config: GenerationConfig,
    history: SessionHistory,
    tokenizer: Tokenizer,
}

impl SessionManager {
    pub fn new(tokenizer: Tokenizer, config: GenerationConfig) -> Self {
        Self {
            config,
            history: SessionHistory { messages: vec![] },
            tokenizer,
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.history.messages.push(Message {
            role: "user".to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        });
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        self.history.messages.push(Message {
            role: "assistant".to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        });
    }

    pub fn clear_history(&mut self) {
        self.history.messages.clear();
        println!("History cleared.");
    }

    pub fn save_history(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.history)?;
        fs::write(path, json)?;
        println!("History saved to {}", path);
        Ok(())
    }

    pub fn load_history(&mut self, path: &str) -> Result<()> {
        let json = fs::read_to_string(path)?;
        self.history = serde_json::from_str(&json)?;
        println!("History loaded from {}", path);
        Ok(())
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(ids, false)?)
    }

    pub fn message_count(&self) -> usize {
        self.history.messages.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.history.messages.iter()
            .map(|m| self.tokenize(&m.content).unwrap_or_default().len())
            .sum()
    }

    pub fn backend_name(&self) -> &str {
        "CPU" // TODO: 実際のバックエンド名を返す
    }
}
```

## 4. マイルストーン

### Milestone 1: MVP (Week 3)
- ✅ 基本的なREPL動作
- ✅ CPU推論エンジン
- ✅ GGUF形式サポート
- ✅ 基本的な会話機能

### Milestone 2: 機能拡張 (Week 5)
- ✅ 全バックエンド対応
- ✅ 複数モデル形式
- ✅ 特殊コマンド完備
- ✅ セッション管理

### Milestone 3: 本番リリース (Week 7)
- ✅ テストカバレッジ80%+
- ✅ パフォーマンス最適化
- ✅ 完全なドキュメント
- ✅ バイナリリリース

## 5. リスク管理

### 技術的リスク
- **GGUF形式の複雑性**: 早期プロトタイピングで検証
- **パフォーマンス問題**: プロファイリングツール常時使用
- **メモリ制約**: 段階的なメモリ最適化

### スケジュールリスク
- **依存ライブラリの遅延**: 代替実装の準備
- **予期せぬバグ**: バッファ期間の確保

## 6. 次のステップ

1. ✅ 要件定義完了
2. ✅ 実装計画完了
3. → プロジェクトセットアップ開始
4. → 基本的なCLI実装
5. → モデルローダー実装

---

**作成日**: 2025-10-03
**バージョン**: 1.0
**ステータス**: Ready for Implementation