# GGUF Tokenizer Extraction Implementation (October 7, 2025)

## 概要

GGUFファイルから埋め込まれたトークナイザー情報を直接抽出する機能を実装しました。これにより、別途tokenizer.jsonファイルを用意することなく、GGUFファイル単体で推論が可能になります。

## 実装内容

### 1. GGUF形式にトークナイザー抽出機能を追加

**ファイル**: `src/formats/gguf.rs`

```rust
/// Extract tokenizer vocabulary from GGUF metadata
pub fn extract_tokenizer_vocab(&self) -> RusTorchResult<Vec<String>> {
    // tokenizer.ggml.tokens配列を読み取り
}

/// Get tokenizer model type (e.g., "llama", "gpt2")
pub fn get_tokenizer_model(&self) -> Option<String> {
    // tokenizer.ggml.modelメタデータを読み取り
}
```

### 2. GGUF専用トークナイザー実装

**ファイル**: `example-cli/src/tokenizer/gguf_tokenizer.rs`

```rust
pub struct GGUFTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
}
```

**特徴**:
- GGUFから抽出した語彙を使用
- Tokenizerトレイトを実装
- BOS/EOS/PAD/UNKトークンIDをサポート

### 3. ModelLoaderの自動抽出機能

**ファイル**: `example-cli/src/model/loader.rs`

GGUF形式の場合、以下の優先順位でトークナイザーを取得：
1. ✅ **GGUFファイルから抽出**（NEW！）
2. 外部tokenizer.jsonファイル
3. ダミートークナイザー

```rust
if matches!(format, ModelFormat::GGUF) {
    match rustorch::formats::gguf::GGUFLoader::from_file(path) {
        Ok(gguf) => {
            match gguf.extract_tokenizer_vocab() {
                Ok(vocab) => {
                    tracing::info!("✅ Extracted {} tokens from GGUF file", vocab.len());
                    return Ok(Box::new(GGUFTokenizer::new(vocab)));
                }
                ...
            }
        }
    }
}
```

## テスト結果

### Mistral-7B-Instruct-v0.2 Q4_K_M

```
✅ Extracted 32000 tokens from GGUF file
📝 Tokenizer model type: llama
```

**成功した点**:
- ✅ 32000トークンの語彙を正常に抽出
- ✅ トークナイザーモデルタイプ（llama）を認識
- ✅ トークンIDからトークン文字列へのマッピング完成
- ✅ エラーなく推論実行

**現在の制限**:
- ⚠️ 簡易実装のため出力はまだ意味不明
- BPEアルゴリズムが未実装（単純な単語分割のみ）
- マージルールが未実装

## GGUFメタデータ構造

GGUFファイルには以下のトークナイザー情報が含まれています：

```
tokenizer.ggml.model: "llama"
tokenizer.ggml.tokens: [32000個の文字列配列]
tokenizer.ggml.scores: [各トークンのスコア]
tokenizer.ggml.token_type: [各トークンのタイプ]
tokenizer.ggml.merges: [BPEマージルール] ← 未実装
```

## 今後の改善点

### 優先度: 高

1. **BPEアルゴリズムの実装**
   - `tokenizer.ggml.merges`の読み取り
   - Byte-Pair Encodingアルゴリズムの実装
   - テキスト→トークンID変換の正確化

2. **SentencePieceサポート**
   - ▁（underscore）プレフィックスの正しい処理
   - 特殊トークンの適切な扱い

### 優先度: 中

3. **トークンタイプとスコアの活用**
   - `tokenizer.ggml.scores`の読み取り
   - `tokenizer.ggml.token_type`の読み取り
   - サンプリング時のスコア活用

4. **正規化とプリプロセス**
   - NFCノーマライゼーション
   - 大文字小文字変換
   - 特殊文字の処理

### 優先度: 低

5. **パフォーマンス最適化**
   - トークナイザーのキャッシング
   - マルチスレッド対応

## 参照実装

- **llama.cpp**: BPEアルゴリズムの参照実装
- **Hugging Face tokenizers**: SentencePiece実装
- **GGML仕様**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## メリット

### ✅ 実装した機能

1. **認証不要**: HuggingFace認証が不要
2. **ファイル一元化**: GGUFファイル単体で完結
3. **自動抽出**: 手動でtokenizer.jsonをダウンロード不要
4. **一貫性保証**: モデルとトークナイザーの不一致を防止

### 🚧 今後の改善で達成

5. **正確なトークン化**: BPE実装で意味のある出力生成
6. **完全な互換性**: llama.cppと同等の出力品質

## 結論

GGUFからのトークナイザー抽出機能は**基盤部分が完成**しました。語彙の抽出とマッピングは正常に動作しています。

**次のステップ**: BPEアルゴリズムの実装により、意味のある出力を生成可能になります。

---

実装者: Claude Code  
日付: 2025年10月7日
