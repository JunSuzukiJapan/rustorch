/// HuggingFace Hub API client for model downloads
use anyhow::Result;
use std::path::{Path, PathBuf};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

const HF_ENDPOINT: &str = "https://huggingface.co";

/// HuggingFace model file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub filename: String,
    pub size: u64,
    pub url: String,
}

/// HuggingFace API client
pub struct HuggingFaceClient {
    client: Client,
    token: Option<String>,
}

impl HuggingFaceClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .user_agent("rustorch-cli/0.1.0")
            .build()?;

        Ok(Self { client, token })
    }

    /// List available files in a repository
    pub fn list_files(&self, repo_id: &str, revision: Option<&str>) -> Result<Vec<ModelFile>> {
        let revision = revision.unwrap_or("main");
        let url = format!("{}/api/models/{}/tree/{}", HF_ENDPOINT, repo_id, revision);

        tracing::debug!("Fetching file list from: {}", url);

        let mut request = self.client.get(&url);

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send()?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to list files for {}: HTTP {}",
                repo_id,
                response.status()
            );
        }

        // Parse response - HuggingFace returns array of file objects
        let files: Vec<serde_json::Value> = response.json()?;

        let model_files: Vec<ModelFile> = files
            .into_iter()
            .filter_map(|file| {
                let filename = file.get("path")?.as_str()?.to_string();
                let size = file.get("size")?.as_u64().unwrap_or(0);

                // Construct download URL
                let url = format!(
                    "{}/{}/resolve/{}/{}",
                    HF_ENDPOINT, repo_id, revision, filename
                );

                Some(ModelFile {
                    filename,
                    size,
                    url,
                })
            })
            .collect();

        Ok(model_files)
    }

    /// Download a specific file from repository
    pub fn download_file(
        &self,
        repo_id: &str,
        filename: &str,
        output_path: &Path,
        revision: Option<&str>,
        progress_callback: Option<Box<dyn Fn(u64, u64)>>,
    ) -> Result<PathBuf> {
        let revision = revision.unwrap_or("main");
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_ENDPOINT, repo_id, revision, filename
        );

        tracing::info!("Downloading {} from {}", filename, repo_id);
        tracing::debug!("Download URL: {}", url);

        let mut request = self.client.get(&url);

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let mut response = request.send()?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download {}: HTTP {}",
                filename,
                response.status()
            );
        }

        // Get total size
        let total_size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Download to temporary file first
        let temp_path = output_path.with_extension("tmp");
        let mut file = std::fs::File::create(&temp_path)?;

        let mut downloaded = 0u64;
        let mut buffer = vec![0; 8192];

        loop {
            use std::io::Read;
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }

            use std::io::Write;
            file.write_all(&buffer[..n])?;

            downloaded += n as u64;

            if let Some(ref callback) = progress_callback {
                callback(downloaded, total_size);
            }
        }

        // Rename temporary file to final path
        std::fs::rename(&temp_path, output_path)?;

        tracing::info!("Downloaded {} to {:?}", filename, output_path);

        Ok(output_path.to_path_buf())
    }

    /// Find GGUF files in repository (common for quantized models)
    pub fn find_gguf_files(&self, repo_id: &str) -> Result<Vec<ModelFile>> {
        let files = self.list_files(repo_id, None)?;

        let gguf_files: Vec<ModelFile> = files
            .into_iter()
            .filter(|f| f.filename.ends_with(".gguf"))
            .collect();

        if gguf_files.is_empty() {
            anyhow::bail!("No GGUF files found in repository: {}", repo_id);
        }

        Ok(gguf_files)
    }

    /// Recommend best GGUF file based on quantization preference
    pub fn recommend_gguf(
        &self,
        repo_id: &str,
        quantization_preference: Option<&str>,
    ) -> Result<ModelFile> {
        let gguf_files = self.find_gguf_files(repo_id)?;

        // If preference specified, try to match
        if let Some(quant) = quantization_preference {
            if let Some(file) = gguf_files
                .iter()
                .find(|f| f.filename.to_lowercase().contains(&quant.to_lowercase()))
            {
                return Ok(file.clone());
            }
        }

        // Default: prefer Q4_K_M (good balance of quality/size)
        let preferred_order = ["q4_k_m", "q4_0", "q5_k_m", "q8_0", "f16"];

        for pref in &preferred_order {
            if let Some(file) = gguf_files
                .iter()
                .find(|f| f.filename.to_lowercase().contains(pref))
            {
                return Ok(file.clone());
            }
        }

        // Fallback: return first file
        Ok(gguf_files[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_client_creation() {
        let client = HuggingFaceClient::new(None);
        assert!(client.is_ok());

        let client_with_token = HuggingFaceClient::new(Some("test-token".to_string()));
        assert!(client_with_token.is_ok());
    }

    #[test]
    fn test_model_file_structure() {
        let file = ModelFile {
            filename: "model.gguf".to_string(),
            size: 1024,
            url: "https://example.com/model.gguf".to_string(),
        };

        assert_eq!(file.filename, "model.gguf");
        assert_eq!(file.size, 1024);
    }
}
