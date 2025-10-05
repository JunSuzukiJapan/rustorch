/// Ollama API client for model downloads
use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const OLLAMA_ENDPOINT: &str = "http://localhost:11434";

/// Ollama model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// Ollama pull progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

/// Ollama API client
pub struct OllamaClient {
    client: Client,
    endpoint: String,
}

impl OllamaClient {
    pub fn new() -> Result<Self> {
        let endpoint = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| OLLAMA_ENDPOINT.to_string());

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 min timeout for large downloads
            .build()?;

        Ok(Self { client, endpoint })
    }

    /// Check if Ollama server is running
    pub fn is_running(&self) -> bool {
        let url = format!("{}/api/tags", self.endpoint);

        self.client
            .get(&url)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// List locally available models
    pub fn list_models(&self) -> Result<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.endpoint);

        let response = self.client.get(&url).send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list Ollama models: HTTP {}", response.status());
        }

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<OllamaModel>,
        }

        let tags: TagsResponse = response.json()?;
        Ok(tags.models)
    }

    /// Pull a model from Ollama library
    pub fn pull_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(&PullProgress)>>,
    ) -> Result<()> {
        let url = format!("{}/api/pull", self.endpoint);

        tracing::info!("Pulling Ollama model: {}", model_name);

        #[derive(Serialize)]
        struct PullRequest {
            name: String,
            stream: bool,
        }

        let request_body = PullRequest {
            name: model_name.to_string(),
            stream: true,
        };

        let response = self.client.post(&url).json(&request_body).send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to pull model: HTTP {}", response.status());
        }

        // Stream progress updates
        use std::io::{BufRead, BufReader};

        let reader = BufReader::new(response);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<PullProgress>(&line) {
                Ok(progress) => {
                    if let Some(ref callback) = progress_callback {
                        callback(&progress);
                    }

                    // Check for completion
                    if progress.status == "success" {
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse progress: {}", e);
                }
            }
        }

        tracing::info!("Model {} pulled successfully", model_name);

        Ok(())
    }

    /// Delete a model from local storage
    pub fn delete_model(&self, model_name: &str) -> Result<()> {
        let url = format!("{}/api/delete", self.endpoint);

        #[derive(Serialize)]
        struct DeleteRequest {
            name: String,
        }

        let response = self
            .client
            .delete(&url)
            .json(&DeleteRequest {
                name: model_name.to_string(),
            })
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to delete model: HTTP {}", response.status());
        }

        tracing::info!("Model {} deleted", model_name);

        Ok(())
    }

    /// Get Ollama model storage path (if accessible)
    pub fn get_model_path(&self, model_name: &str) -> Option<PathBuf> {
        // Ollama stores models in platform-specific locations:
        // - macOS: ~/.ollama/models
        // - Linux: /usr/share/ollama/.ollama/models or ~/.ollama/models
        // - Windows: %USERPROFILE%\.ollama\models

        let base_dir = if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
            dirs::home_dir()?.join(".ollama").join("models")
        } else if cfg!(target_os = "windows") {
            dirs::home_dir()?.join(".ollama").join("models")
        } else {
            return None;
        };

        // Model files are stored with blob digests
        // This is a simplified path - actual implementation may need to query Ollama API
        Some(base_dir.join("blobs").join(model_name))
    }

    /// Check if a model is already pulled
    pub fn is_model_available(&self, model_name: &str) -> Result<bool> {
        let models = self.list_models()?;
        Ok(models.iter().any(|m| m.name == model_name))
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new().expect("Failed to create Ollama client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_pull_progress_parsing() {
        let json = r#"{"status":"downloading","total":1000,"completed":500}"#;
        let progress: PullProgress = serde_json::from_str(json).unwrap();

        assert_eq!(progress.status, "downloading");
        assert_eq!(progress.total, Some(1000));
        assert_eq!(progress.completed, Some(500));
    }

    #[test]
    fn test_ollama_model_parsing() {
        let json = r#"{"name":"llama2:7b","size":3825819519,"digest":"abc123","modified_at":"2024-01-01T00:00:00Z"}"#;
        let model: OllamaModel = serde_json::from_str(json).unwrap();

        assert_eq!(model.name, "llama2:7b");
        assert_eq!(model.size, 3825819519);
    }
}
