# example-cli Refactoring Plan
生成日時: 2025-10-08

## 🎯 リファクタリングの目的

Metal GPU統合の前に、example-cliのコード品質を改善し、保守性を向上させる。

## 📊 現状分析

### コード統計
- **総ファイル数**: 57 Rustファイル
- **総行数**: 12,057行
- **公開型**: 77個（struct/enum）
- **公開関数**: 268個
- **TODO/FIXME**: 3個のみ（良好）

### 最大ファイル
1. **inference.rs** (932行) - 推論エンジン
2. **tensor_loader.rs** (653行) - テンソルローダー
3. **repl.rs** (522行) - REPLインターフェース
4. **gguf.rs** (511行) - GGUFフォーマット
5. **main.rs** (505行) - エントリーポイント

## 🔴 主要な問題点

### 1. main.rs の肥大化（505行）

**問題箇所**: 32-400行（`start_cli()`関数）

#### 1.1 設定マージロジックの冗長性（46-76行）
```rust
// 悪い例: 手動でフィールドごとにマージ
let max_tokens = if args.max_tokens != 512 {
    args.max_tokens
} else {
    file_config.generation.max_tokens
};

let temperature = if (args.temperature - 0.7).abs() > f32::EPSILON {
    args.temperature
} else {
    file_config.generation.temperature
};
// ... 繰り返し
```

**改善案**: ConfigBuilderパターンまたはmerge()メソッド
```rust
// 良い例
let gen_config = GenerationConfig::builder()
    .with_cli_args(&args)
    .with_file_config(&file_config)
    .build();
```

#### 1.2 バックエンドロード  ロジックの重複（112-300行）

**問題**:
- `hybrid-f32`バックエンド: 112-198行（87行）
- `mac-hybrid`バックエンド: 202-270行（69行）
- `metal`バックエンド: 271-340行（70行）
- **ほぼ同じコードが3回繰り返されている**

**重複コード例**:
```rust
// hybrid-f32 (120-128行)
let model_name_lower = model_path.file_name()
    .and_then(|n| n.to_str())
    .unwrap_or("")
    .to_lowercase();

let is_llama = model_name_lower.contains("llama") ||
              model_name_lower.contains("mistral") ||
              model_name_lower.contains("mixtral");

// mac-hybrid (210-218行) - 完全に同じ
let model_name_lower = model_path.file_name()
    .and_then(|n| n.to_str())
    .unwrap_or("")
    .to_lowercase();

let is_llama = model_name_lower.contains("llama") ||
              model_name_lower.contains("mistral") ||
              model_name_lower.contains("mixtral");

// metal (279-287行) - 完全に同じ
```

**改善案**: BackendLoaderファクトリーパターン
```rust
// 新しい構造
pub struct BackendLoader;

impl BackendLoader {
    pub fn load_model(
        backend: Backend,
        model_path: &Path,
        engine: &mut InferenceEngine
    ) -> Result<()> {
        let architecture = Self::detect_architecture(model_path)?;

        match (backend, architecture) {
            (Backend::HybridF32, Architecture::Llama) => {
                Self::load_f32_llama(model_path, DeviceType::Metal, engine)
            }
            (Backend::MacHybrid, Architecture::Llama) => {
                Self::load_f32_llama(model_path, DeviceType::Hybrid, engine)
            }
            // ...
        }
    }

    fn detect_architecture(model_path: &Path) -> Result<Architecture> {
        let model_name = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if model_name.contains("llama") ||
           model_name.contains("mistral") ||
           model_name.contains("mixtral") {
            Ok(Architecture::Llama)
        } else {
            Ok(Architecture::GPT)
        }
    }

    fn load_f32_llama(
        model_path: &Path,
        device_type: DeviceType,
        engine: &mut InferenceEngine
    ) -> Result<()> {
        // 共通実装
    }
}
```

### 2. inference.rs の複雑性（932行）

**現状**: 単一のimplブロックに28個のメソッド

**改善案**:
- 推論ロジックを分離: `InferenceEngine` + `InferenceBackend` trait
- サンプリング戦略をモジュール化

### 3. モジュール間の責任分離不足

**現状の問題**:
- `main.rs`がバックエンドロード詳細を知りすぎている
- `InferenceEngine`が複数のモデル型を直接保持（f64/f32/Llama）

## 🎯 リファクタリング計画

### Phase 1: 設定マージの改善（優先度: 高）

**対象ファイル**: `src/utils/config.rs`

**実装**:
```rust
impl GenerationConfig {
    /// Merge CLI args with file config
    /// CLI args take precedence
    pub fn merge(file_config: &GenerationConfig, args: &CliArgs) -> Self {
        Self {
            max_tokens: Self::override_value(
                args.max_tokens,
                512, // default
                file_config.max_tokens
            ),
            temperature: Self::override_f32(
                args.temperature,
                0.7, // default
                file_config.temperature
            ),
            top_p: Self::override_f32(
                args.top_p,
                0.9, // default
                file_config.top_p
            ),
            top_k: Self::override_value(
                args.top_k,
                40, // default
                file_config.top_k as u32
            ) as usize,
        }
    }

    fn override_value<T: PartialEq>(cli: T, default: T, file: T) -> T {
        if cli != default { cli } else { file }
    }

    fn override_f32(cli: f32, default: f32, file: f32) -> f32 {
        if (cli - default).abs() > f32::EPSILON { cli } else { file }
    }
}
```

**削減行数**: main.rs から約30行削減

### Phase 2: バックエンドローダーの分離（優先度: 高）

**新規ファイル**: `src/model/backend_loader.rs`

**構造**:
```rust
pub enum Architecture {
    GPT,
    Llama,
}

pub struct BackendLoader;

impl BackendLoader {
    /// Load model with specified backend
    pub fn load_model(
        backend: Backend,
        model_path: &Path,
        engine: &mut InferenceEngine,
    ) -> Result<()> {
        let arch = Self::detect_architecture(model_path)?;

        #[cfg(feature = "hybrid-f32")]
        if backend == Backend::HybridF32 {
            return Self::load_hybrid_f32(model_path, arch, engine);
        }

        #[cfg(feature = "mac-hybrid")]
        if backend == Backend::Hybrid {
            return Self::load_mac_hybrid(model_path, arch, engine);
        }

        #[cfg(feature = "metal")]
        if backend == Backend::Metal {
            return Self::load_metal(model_path, arch, engine);
        }

        // CPU fallback
        Self::load_cpu(model_path, arch, engine)
    }

    fn detect_architecture(path: &Path) -> Result<Architecture>;
    fn load_hybrid_f32(path: &Path, arch: Architecture, engine: &mut InferenceEngine) -> Result<()>;
    fn load_mac_hybrid(path: &Path, arch: Architecture, engine: &mut InferenceEngine) -> Result<()>;
    fn load_metal(path: &Path, arch: Architecture, engine: &mut InferenceEngine) -> Result<()>;
    fn load_cpu(path: &Path, arch: Architecture, engine: &mut InferenceEngine) -> Result<()>;
}
```

**削減行数**: main.rsから約200行削減

### Phase 3: InferenceEngineの簡素化（優先度: 中）

**現状の問題**:
```rust
pub struct InferenceEngine {
    model: Option<TransformerModel>,
    gpt_model: Option<GPTModel>,
    #[cfg(feature = "hybrid-f32")]
    f32_gpt_model: Option<F32GPTModel>,
    #[cfg(feature = "hybrid-f32")]
    f32_llama_model: Option<F32LlamaModel>,
    // ...
}
```

**改善案**: Enum Dispatchパターン
```rust
pub enum ModelBackend {
    Transformer(TransformerModel),
    GPT(GPTModel),
    #[cfg(feature = "hybrid-f32")]
    F32GPT(F32GPTModel),
    #[cfg(feature = "hybrid-f32")]
    F32Llama(F32LlamaModel),
}

pub struct InferenceEngine {
    backend: Option<ModelBackend>,
    generation_config: GenerationConfig,
    sampling_config: SamplingConfig,
    loader: ModelLoader,
}

impl InferenceEngine {
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        match &mut self.backend {
            Some(ModelBackend::Transformer(m)) => self.generate_transformer(m, prompt),
            Some(ModelBackend::GPT(m)) => self.generate_gpt(m, prompt),
            #[cfg(feature = "hybrid-f32")]
            Some(ModelBackend::F32GPT(m)) => self.generate_f32_gpt(m, prompt),
            #[cfg(feature = "hybrid-f32")]
            Some(ModelBackend::F32Llama(m)) => self.generate_f32_llama(m, prompt),
            None => Err(anyhow::anyhow!("No model loaded")),
        }
    }
}
```

### Phase 4: モジュール構造の整理（優先度: 低）

**提案構造**:
```
src/
├── main.rs (簡素化: 150行以下)
├── lib.rs
├── backend/
│   ├── mod.rs
│   ├── cpu.rs
│   ├── metal.rs
│   ├── cuda.rs
│   └── hybrid.rs
├── model/
│   ├── mod.rs
│   ├── loader.rs
│   ├── backend_loader.rs (新規)
│   ├── inference.rs (簡素化)
│   └── architectures/
├── config/  (新規: utils/config.rs から移動)
│   ├── mod.rs
│   ├── generation.rs
│   └── merge.rs
└── ...
```

## 📋 実装順序

### Step 1: 設定マージの改善 ✅
- [ ] `GenerationConfig::merge()`実装
- [ ] main.rsで使用
- [ ] テスト追加

### Step 2: バックエンドローダー分離 ✅
- [ ] `src/model/backend_loader.rs`作成
- [ ] `Architecture`enum定義
- [ ] `BackendLoader`実装
- [ ] main.rsから移行
- [ ] テスト追加

### Step 3: main.rsの簡素化確認 ✅
- [ ] 行数確認（目標: 200行以下）
- [ ] 複雑度確認
- [ ] ドキュメント追加

### Step 4: InferenceEngine簡素化（オプション）
- [ ] `ModelBackend` enum実装
- [ ] `InferenceEngine`リファクタリング
- [ ] 既存機能の動作確認

## 🎓 期待される効果

### コード品質
- ✅ main.rs: 505行 → 約200行（60%削減）
- ✅ 重複コード削減: 約200行
- ✅ モジュール責任の明確化

### 保守性
- ✅ 新しいバックエンド追加が容易
- ✅ 設定マージロジックの一元管理
- ✅ テスト容易性の向上

### Metal統合の準備
- ✅ バックエンドロジックが分離されているため、Metalバックエンド統合が容易
- ✅ InferenceEngineの複雑性が低減され、GPU統合のテストが簡単

## 🚀 次のアクション

1. **即座に開始**: Step 1（設定マージ改善）
2. **Phase 1完了後**: Step 2（バックエンドローダー分離）
3. **リファクタリング完了後**: Metal GPU統合に進む

## 📊 成功基準

- [ ] main.rsが200行以下
- [ ] 重複コードゼロ（DRY原則遵守）
- [ ] 既存機能すべて動作（regression無し）
- [ ] 新しいテストカバレッジ >80%
- [ ] Metal統合の準備完了
