// Common trait for all model format loaders
use anyhow::Result;
use std::path::Path;

use super::ModelMetadata;

/// Common interface for all model format loaders
/// 全モデル形式ローダーの共通インターフェース
pub trait FormatLoader: Sized {
    /// Load model metadata from file
    /// ファイルからモデルメタデータを読み込み
    fn load_metadata(path: &Path) -> Result<ModelMetadata>;

    /// Get default tokenizer path for this format
    /// この形式のデフォルトトークナイザーパスを取得
    fn default_tokenizer_path(model_path: &Path) -> Option<std::path::PathBuf> {
        // Most formats use .tokenizer.json in the same directory
        let tokenizer_path = model_path
            .with_extension("")
            .with_extension("tokenizer.json");

        if tokenizer_path.exists() {
            Some(tokenizer_path)
        } else {
            None
        }
    }

    /// Check if this loader can handle the given path
    /// このローダーが指定されたパスを処理できるか確認
    fn can_load(path: &Path) -> bool;
}

/// Helper functions for metadata extraction
/// メタデータ抽出のヘルパー関数
pub mod metadata_utils {
    use super::*;

    /// Extract layer number from tensor name (e.g., "layer.0.weight" -> Some(0))
    /// テンソル名からレイヤー番号を抽出
    pub fn extract_layer_number(name: &str) -> Option<usize> {
        name.split(&['.', '_'][..])
            .find_map(|part| part.parse::<usize>().ok())
    }

    /// Count maximum layer number from tensor names
    /// テンソル名から最大レイヤー番号をカウント
    pub fn count_layers(tensor_names: &[impl AsRef<str>]) -> usize {
        tensor_names
            .iter()
            .filter_map(|name| extract_layer_number(name.as_ref()))
            .max()
            .map(|n| n + 1)
            .unwrap_or(6) // Default to 6 layers
    }

    /// Extract model name from path
    /// パスからモデル名を抽出
    pub fn extract_model_name(path: &Path) -> String {
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            metadata_utils::extract_layer_number("layer.0.weight"),
            Some(0)
        );
        assert_eq!(
            metadata_utils::extract_layer_number("block_12.attn"),
            Some(12)
        );
        assert_eq!(
            metadata_utils::extract_layer_number("layers.23.mlp.weight"),
            Some(23)
        );
        assert_eq!(metadata_utils::extract_layer_number("embed.weight"), None);
    }

    #[test]
    fn test_count_layers() {
        let names = vec![
            "layer.0.weight",
            "layer.5.weight",
            "layer.11.weight",
            "embed.weight",
        ];
        assert_eq!(metadata_utils::count_layers(&names), 12); // 0-11 = 12 layers
    }
}
