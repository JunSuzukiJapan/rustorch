/// Unified model download manager
use super::{
    ModelIdentifier, ModelSource, DownloadOptions,
    HuggingFaceClient, OllamaClient,
    progress::{ProgressBar, Spinner},
};
use anyhow::Result;
use std::path::PathBuf;

pub struct ModelDownloadManager {
    hf_client: Option<HuggingFaceClient>,
    ollama_client: Option<OllamaClient>,
}

impl ModelDownloadManager {
    pub fn new() -> Result<Self> {
        // Initialize HuggingFace client with token from environment
        let hf_token = DownloadOptions::get_hf_token();
        let hf_client = HuggingFaceClient::new(hf_token).ok();

        // Initialize Ollama client if server is running
        let ollama_client = OllamaClient::new().ok();

        Ok(Self {
            hf_client,
            ollama_client,
        })
    }

    /// Download a model from specified source
    pub fn download(
        &self,
        identifier: &ModelIdentifier,
        options: &DownloadOptions,
    ) -> Result<PathBuf> {
        match identifier.source {
            ModelSource::HuggingFace => self.download_from_huggingface(identifier, options),
            ModelSource::Ollama => self.download_from_ollama(identifier, options),
            ModelSource::ModelScope => {
                anyhow::bail!("ModelScope support not yet implemented")
            }
        }
    }

    fn download_from_huggingface(
        &self,
        identifier: &ModelIdentifier,
        options: &DownloadOptions,
    ) -> Result<PathBuf> {
        let client = self
            .hf_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HuggingFace client not initialized"))?;

        println!("🔍 Searching for models in {}...", identifier.repo_id);

        // Determine which file to download
        let file = if let Some(ref filename) = identifier.filename {
            // User specified exact filename
            use super::huggingface::ModelFile;
            ModelFile {
                filename: filename.clone(),
                size: 0, // Will be determined during download
                url: String::new(),
            }
        } else if let Some(ref quant) = options.quantization {
            // User specified quantization level - find matching GGUF
            println!("🎯 Looking for {} quantization...", quant);
            client.recommend_gguf(&identifier.repo_id, Some(quant))?
        } else {
            // Auto-detect best GGUF file
            println!("🎯 Auto-detecting best model file...");
            client.recommend_gguf(&identifier.repo_id, None)?
        };

        println!(
            "📥 Downloading: {} ({} MB)",
            file.filename,
            file.size / 1_048_576
        );

        // Prepare output path
        let output_path = options
            .output_dir
            .join(&identifier.repo_id.replace('/', "_"))
            .join(&file.filename);

        // Check if already downloaded
        if output_path.exists() && !options.force {
            println!("✓ Model already exists at: {:?}", output_path);
            println!("  Use --force to re-download");
            return Ok(output_path);
        }

        // Download with progress callback
        use std::sync::{Arc, Mutex};
        let last_percentage = Arc::new(Mutex::new(0u32));

        let result = client.download_file(
            &identifier.repo_id,
            &file.filename,
            &output_path,
            identifier.revision.as_deref(),
            Some(Box::new(move |downloaded, total| {
                if total > 0 {
                    let percentage = (downloaded as f64 / total as f64 * 100.0) as u32;
                    let mut last = last_percentage.lock().unwrap();

                    // Only print every 10%
                    if percentage / 10 > *last / 10 {
                        println!("  Progress: {}%", percentage);
                        *last = percentage;
                    }
                }
            })),
        )?;

        println!();
        println!("✓ Model downloaded successfully!");
        println!("📂 Location: {:?}", result);

        Ok(result)
    }

    fn download_from_ollama(
        &self,
        identifier: &ModelIdentifier,
        _options: &DownloadOptions,
    ) -> Result<PathBuf> {
        let client = self
            .ollama_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized. Is Ollama running?"))?;

        // Check if Ollama server is running
        if !client.is_running() {
            anyhow::bail!(
                "Ollama server is not running. Please start Ollama first:\n\
                 - macOS: Run 'ollama serve' or start Ollama.app\n\
                 - Linux: Run 'systemctl start ollama' or 'ollama serve'\n\
                 - Windows: Start Ollama application"
            );
        }

        let model_name = &identifier.repo_id;

        // Check if already pulled
        if client.is_model_available(model_name)? {
            println!("✓ Model {} is already available in Ollama", model_name);

            if let Some(path) = client.get_model_path(model_name) {
                return Ok(path);
            } else {
                // Model exists but path cannot be determined
                // Return a virtual path indicator
                return Ok(PathBuf::from(format!("ollama://{}", model_name)));
            }
        }

        println!("📥 Pulling model from Ollama: {}", model_name);

        use std::sync::{Arc, Mutex};
        let last_status = Arc::new(Mutex::new(String::new()));

        client.pull_model(
            model_name,
            Some(Box::new(move |progress| {
                let mut last = last_status.lock().unwrap();
                let status_msg = if let (Some(total), Some(completed)) = (progress.total, progress.completed) {
                    let percentage = (completed as f64 / total as f64 * 100.0) as u32;
                    format!("  Progress: {}%", percentage)
                } else {
                    format!("  Status: {}", progress.status)
                };

                // Only print if status changed
                if *last != status_msg {
                    println!("{}", status_msg);
                    *last = status_msg;
                }
            })),
        )?;

        println!("✓ Model {} pulled successfully", model_name);

        // Return model path (virtual for Ollama)
        Ok(PathBuf::from(format!("ollama://{}", model_name)))
    }

    /// List available models from a source
    pub fn list_models(&self, source: ModelSource) -> Result<Vec<String>> {
        match source {
            ModelSource::Ollama => {
                let client = self
                    .ollama_client
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Ollama client not available"))?;

                let models = client.list_models()?;
                Ok(models.into_iter().map(|m| m.name).collect())
            }
            ModelSource::HuggingFace => {
                anyhow::bail!("Listing HuggingFace models requires search API (not yet implemented)")
            }
            ModelSource::ModelScope => {
                anyhow::bail!("ModelScope not yet implemented")
            }
        }
    }
}

impl Default for ModelDownloadManager {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelDownloadManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = ModelDownloadManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_download_options_default() {
        let opts = DownloadOptions::default();
        assert!(opts.output_dir.to_string_lossy().contains(".rustorch"));
    }
}
