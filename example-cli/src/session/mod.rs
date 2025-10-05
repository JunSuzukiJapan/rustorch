pub mod config;
pub mod history;

use anyhow::Result;
use std::fs;
use std::path::Path;

pub use config::GenerationConfig;
pub use history::{Message, SessionHistory};

pub struct SessionManager {
    pub config: GenerationConfig,
    history: SessionHistory,
    backend_name: String,
    model_name: String,
    system_prompt: Option<String>,
    auto_save_path: Option<std::path::PathBuf>,
}

impl SessionManager {
    pub fn new(
        config: GenerationConfig,
        backend_name: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            config,
            history: SessionHistory::new(),
            backend_name: backend_name.into(),
            model_name: model_name.into(),
            system_prompt: None,
            auto_save_path: None,
        }
    }

    /// Create a dummy session for testing
    pub fn new_dummy(
        config: GenerationConfig,
        backend_name: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self::new(config, backend_name, model_name)
    }

    pub fn with_auto_save(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.auto_save_path = Some(path.into());
        self
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.history.add_user_message(content);
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        self.history.add_assistant_message(content);
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn save_history(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.history)?;

        // Create parent directories if they don't exist
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_history(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let json = fs::read_to_string(path)?;
        self.history = serde_json::from_str(&json)?;
        Ok(())
    }

    pub fn auto_save(&self) -> Result<()> {
        if let Some(path) = &self.auto_save_path {
            self.save_history(path)?;
        }
        Ok(())
    }

    pub fn message_count(&self) -> usize {
        self.history.message_count()
    }

    pub fn total_tokens(&self) -> usize {
        // Improved token estimation using character-based calculation
        // GPT-style tokenizers typically use ~4 characters per token on average
        self.history
            .messages()
            .iter()
            .map(|m| {
                // Estimate tokens: character count / 4 + word count * 0.5
                // This provides better approximation than word count alone
                let char_count = m.content.chars().count();
                let word_count = m.content.split_whitespace().count();
                (char_count / 4).max(word_count)
            })
            .sum()
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn set_backend_name(&mut self, backend: impl Into<String>) {
        self.backend_name = backend.into();
    }

    pub fn set_model_name(&mut self, model: impl Into<String>) {
        self.model_name = model.into();
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        &self.config
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = Some(prompt.into());
    }

    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    pub fn print_config(&self) {
        println!("  Model:           {}", self.model_name);
        println!("  Backend:         {}", self.backend_name);
        println!("  Max tokens:      {}", self.config.max_tokens);
        println!("  Temperature:     {:.2}", self.config.temperature);
        println!("  Top-p:           {:.2}", self.config.top_p);
        println!("  Top-k:           {}", self.config.top_k);

        if let Some(prompt) = &self.system_prompt {
            println!("  System prompt:   {}", prompt);
        }
    }

    pub fn get_messages(&self) -> &[Message] {
        self.history.messages()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_session() -> SessionManager {
        let config = GenerationConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        };
        SessionManager::new(config, "cpu", "test-model")
    }

    #[test]
    fn test_session_creation() {
        let session = create_test_session();
        assert_eq!(session.backend_name(), "cpu");
        assert_eq!(session.model_name(), "test-model");
        assert_eq!(session.message_count(), 0);
    }

    #[test]
    fn test_add_messages() {
        let mut session = create_test_session();
        session.add_user_message("Hello");
        session.add_assistant_message("Hi there!");

        assert_eq!(session.message_count(), 2);
    }

    #[test]
    fn test_clear_history() {
        let mut session = create_test_session();
        session.add_user_message("Test");
        assert_eq!(session.message_count(), 1);

        session.clear_history();
        assert_eq!(session.message_count(), 0);
    }

    #[test]
    fn test_system_prompt() {
        let mut session = create_test_session();
        assert!(session.system_prompt().is_none());

        session.set_system_prompt("You are a helpful assistant");
        assert_eq!(session.system_prompt(), Some("You are a helpful assistant"));
    }

    #[test]
    fn test_save_and_load_history() {
        use tempfile::NamedTempFile;

        let mut session = create_test_session();
        session.add_user_message("Hello");
        session.add_assistant_message("Hi!");

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        session.save_history(path).unwrap();

        let mut new_session = create_test_session();
        new_session.load_history(path).unwrap();

        assert_eq!(new_session.message_count(), 2);
    }
}
