use anyhow::Result;
use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;

use super::commands::Command;
use crate::model::InferenceEngine;
use crate::session::SessionManager;
use crate::utils::ProgressIndicator;

pub struct REPL {
    editor: DefaultEditor,
    session: SessionManager,
    engine: InferenceEngine,
    show_progress: bool,
    use_chat_template: bool,  // Toggle for chat template usage
}

impl REPL {
    pub fn new(
        session: SessionManager,
        engine: InferenceEngine,
        show_progress: bool,
    ) -> Result<Self> {
        let editor = DefaultEditor::new()?;
        Ok(Self {
            editor,
            session,
            engine,
            show_progress,
            use_chat_template: true,  // Enable by default
        })
    }

    /// Set chat template usage (for auto-detection)
    pub fn set_use_chat_template(&mut self, enabled: bool) {
        self.use_chat_template = enabled;
    }

    /// Get current chat template state
    pub fn use_chat_template(&self) -> bool {
        self.use_chat_template
    }

    pub fn run(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            // Read input (potentially multiline)
            let input = match self.read_input() {
                Ok(Some(text)) => text,
                Ok(None) => break, // EOF or exit
                Err(e) => {
                    eprintln!("{}", format!("Error: {}", e).red());
                    continue;
                }
            };

            if input.is_empty() {
                continue;
            }

            // Add to history
            self.editor.add_history_entry(&input)?;

            // Process command or message
            if input.starts_with('/') {
                match Command::parse(&input) {
                    Ok(cmd) => {
                        if !self.handle_command(cmd)? {
                            break; // Exit command
                        }
                    }
                    Err(e) => {
                        eprintln!("{}", format!("Error: {}", e).red());
                    }
                }
            } else if let Err(e) = self.handle_message(&input) {
                eprintln!("{}", format!("Error: {}", e).red());
            }
        }

        Ok(())
    }

    /// Read input from user, supporting multiline input
    fn read_input(&mut self) -> Result<Option<String>> {
        let mut lines = Vec::new();

        // Create prompt with template status indicator
        let template_indicator = if self.use_chat_template { "📋" } else { "  " };
        let mut prompt = format!("{}You> ", template_indicator);

        loop {
            match self.editor.readline(&prompt) {
                Ok(line) => {
                    let trimmed = line.trim_end();

                    // Check if line ends with backslash (continuation)
                    if let Some(stripped) = trimmed.strip_suffix('\\') {
                        // Remove backslash and add line
                        lines.push(stripped.to_string());
                        prompt = "...> ".to_string(); // Continuation prompt
                        continue;
                    } else {
                        // Add final line and break
                        lines.push(line);
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    if lines.is_empty() {
                        println!("{}", "Use /exit to quit, or press Ctrl+D".bright_black());
                        return Ok(Some(String::new()));
                    } else {
                        // Cancel multiline input
                        println!("{}", "Multiline input cancelled.".yellow());
                        return Ok(Some(String::new()));
                    }
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    return Ok(None);
                }
                Err(err) => {
                    return Err(err.into());
                }
            }
        }

        let combined = lines.join("\n");
        Ok(Some(combined.trim().to_string()))
    }

    fn print_welcome(&self) {
        println!(
            "{}",
            "╔════════════════════════════════════════════════════════════╗".bright_blue()
        );
        println!(
            "{}",
            "║           RusTorch CLI - Local LLM Chat                   ║".bright_blue()
        );
        println!(
            "{}",
            "╚════════════════════════════════════════════════════════════╝".bright_blue()
        );
        println!();
        println!(
            "{} {}",
            "Backend:".bright_green(),
            self.session.backend_name().cyan()
        );
        println!(
            "{} {}",
            "Model:".bright_green(),
            self.session.model_name().cyan()
        );

        // Template status indicator
        let template_status = if self.use_chat_template {
            "ON".bright_green()
        } else {
            "OFF".bright_red()
        };
        println!(
            "{} {} {}",
            "Chat Template:".bright_green(),
            template_status,
            "(Press TAB to toggle)".bright_black()
        );

        println!();
        println!(
            "{}",
            "Type '/help' for available commands, '/exit' to quit.".bright_black()
        );
        println!();
    }

    fn handle_command(&mut self, command: Command) -> Result<bool> {
        match command {
            Command::Exit => {
                self.handle_exit()?;
                return Ok(false);
            }
            Command::Help => self.handle_help(),
            Command::Clear => self.handle_clear()?,
            Command::Save(path) => self.handle_save(path)?,
            Command::Load(path) => self.handle_load(path)?,
            Command::Model(path) => self.handle_model(path)?,
            Command::SwitchBackend(backend) => self.handle_switch_backend(&backend)?,
            Command::Stats => self.handle_stats(),
            Command::System(prompt) => self.handle_system(&prompt)?,
            Command::Config => self.handle_config(),
            Command::ConfigSave(path) => self.handle_config_save(path)?,
            Command::TemplateToggle => self.handle_template_toggle(),
            Command::Unknown(cmd) => {
                println!("Unknown command: {}", cmd);
                println!("Type '/help' for available commands.");
            }
        }

        Ok(true)
    }

    fn handle_template_toggle(&mut self) {
        self.use_chat_template = !self.use_chat_template;
        let status = if self.use_chat_template {
            "ON".bright_green()
        } else {
            "OFF".bright_red()
        };
        println!(
            "{} Chat template is now {}",
            "✓".bright_green(),
            status
        );
        println!(
            "{}",
            "  → Raw tokens will be used without template wrapper".bright_black()
        );
    }

    fn handle_exit(&mut self) -> Result<()> {
        println!("{}", "Saving session...".yellow());
        if let Err(e) = self.session.auto_save() {
            eprintln!(
                "{} {}",
                "Warning:".yellow(),
                format!("Failed to auto-save session: {}", e).red()
            );
        }
        println!("{}", "Goodbye!".bright_cyan());
        Ok(())
    }

    fn handle_help(&self) {
        println!("{}", Command::help_text());
    }

    fn handle_clear(&mut self) -> Result<()> {
        self.session.clear_history();
        println!("{}", "Conversation history cleared.".green());
        Ok(())
    }

    fn handle_save(&mut self, path: Option<PathBuf>) -> Result<()> {
        let path = path.unwrap_or_else(crate::cli::CliArgs::get_default_history_path);
        self.session.save_history(&path)?;
        println!(
            "{} {}",
            "Conversation saved to:".green(),
            path.display().to_string().cyan()
        );
        Ok(())
    }

    fn handle_load(&mut self, path: Option<PathBuf>) -> Result<()> {
        let path = path.unwrap_or_else(crate::cli::CliArgs::get_default_history_path);
        self.session.load_history(&path)?;
        println!(
            "{} {}",
            "Conversation loaded from:".green(),
            path.display().to_string().cyan()
        );
        Ok(())
    }

    fn handle_model(&mut self, path: Option<PathBuf>) -> Result<()> {
        match path {
            Some(model_path) => {
                // Validate file exists
                if !model_path.exists() {
                    eprintln!(
                        "{} Model file not found: {}",
                        "Error:".red(),
                        model_path.display()
                    );
                    return Ok(());
                }

                println!(
                    "{} {}",
                    "Loading model:".bright_green(),
                    model_path.display().to_string().cyan()
                );

                // Update session model name
                self.session
                    .set_model_name(model_path.display().to_string());
                println!("{}", "Model path updated.".green());
                println!(
                    "{}",
                    "Note: Actual model loading will be implemented with full inference support."
                        .bright_black()
                );
            }
            None => {
                // Show current model info
                println!(
                    "{} {}",
                    "Current model:".bright_green(),
                    self.session.model_name().cyan()
                );
                println!("{}", "Usage: /model <path>".bright_black());
            }
        }
        Ok(())
    }

    fn handle_switch_backend(&mut self, backend: &str) -> Result<()> {
        use crate::cli::Backend;

        // Parse backend from string
        let new_backend = match backend.to_lowercase().as_str() {
            "cpu" => Backend::Cpu,
            "cuda" | "gpu" => Backend::Cuda,
            "metal" => Backend::Metal,
            _ => {
                eprintln!("{} Unknown backend: {}", "Error:".red(), backend);
                println!("{}", "Available backends: cpu, cuda, metal".bright_black());
                return Ok(());
            }
        };

        // Check if backend is available
        if !new_backend.is_available() {
            eprintln!(
                "{} Backend '{}' may not be available on this system",
                "Warning:".yellow(),
                backend
            );
            println!("{}", "Attempting to switch anyway...".bright_black());
        }

        // Update session backend
        self.session.set_backend_name(new_backend.as_str());
        println!(
            "{} {}",
            "Backend switched to:".green(),
            new_backend.as_str().cyan()
        );

        // Note: In a full implementation, we would recreate the inference engine
        // with the new backend. For now, we just update the name.
        println!(
            "{}",
            "Note: Backend switch will take effect for new models.".bright_black()
        );

        Ok(())
    }

    fn handle_stats(&self) {
        println!(
            "{}",
            "╔════════════════════════════════════════════════════════════╗".bright_blue()
        );
        println!(
            "{}",
            "║                     Statistics                             ║".bright_blue()
        );
        println!(
            "{}",
            "╚════════════════════════════════════════════════════════════╝".bright_blue()
        );
        println!();
        println!(
            "  {:<16} {}",
            "Messages:".bright_green(),
            self.session.message_count().to_string().cyan()
        );
        println!(
            "  {:<16} {}",
            "Total tokens:".bright_green(),
            self.session.total_tokens().to_string().cyan()
        );
        println!(
            "  {:<16} {}",
            "Backend:".bright_green(),
            self.session.backend_name().cyan()
        );
        println!(
            "  {:<16} {}",
            "Model:".bright_green(),
            self.session.model_name().cyan()
        );
        println!();
    }

    fn handle_system(&mut self, prompt: &str) -> Result<()> {
        self.session.set_system_prompt(prompt);
        println!("{}", "System prompt updated.".green());
        Ok(())
    }

    fn handle_config(&self) {
        println!(
            "{}",
            "╔════════════════════════════════════════════════════════════╗".bright_blue()
        );
        println!(
            "{}",
            "║                   Configuration                            ║".bright_blue()
        );
        println!(
            "{}",
            "╚════════════════════════════════════════════════════════════╝".bright_blue()
        );
        println!();
        self.session.print_config();
        println!();
    }

    fn handle_config_save(&self, path: Option<PathBuf>) -> Result<()> {
        use crate::utils::Config;

        // Create config from current session settings
        let mut config = Config::default();

        // Set generation config from session
        let gen_config = self.session.generation_config();
        config.generation.max_tokens = gen_config.max_tokens;
        config.generation.temperature = gen_config.temperature;
        config.generation.top_p = gen_config.top_p;
        config.generation.top_k = gen_config.top_k as usize;

        // Set backend
        config.backend.default = self.session.backend_name().to_string();

        // Save to file
        if let Some(path) = path {
            config.save_to_file(&path)?;
            println!(
                "{} {}",
                "Configuration saved to:".green(),
                path.display().to_string().cyan()
            );
        } else {
            config.save_default()?;
            let config_path = Config::default_config_path()?;
            println!(
                "{} {}",
                "Configuration saved to:".green(),
                config_path.display().to_string().cyan()
            );
        }

        Ok(())
    }

    fn handle_message(&mut self, message: &str) -> Result<()> {
        self.session.add_user_message(message);

        if self.show_progress {
            // Show brief progress indicator
            let progress = ProgressIndicator::new("Thinking");
            std::thread::sleep(std::time::Duration::from_millis(300));
            progress.finish();
        }

        // Display assistant label
        print!("{} ", "Assistant>".bright_magenta().bold());
        std::io::Write::flush(&mut std::io::stdout())?;

        // Generate response (streaming disabled for KV cache compatibility)
        let full_response = self.generate_response(message)?;
        println!("{}", full_response.white());

        self.session.add_assistant_message(full_response.trim());

        Ok(())
    }

    fn generate_response(&mut self, message: &str) -> Result<String> {
        self.engine.generate(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{InferenceEngine, ModelLoader};
    use crate::session::GenerationConfig;

    fn create_test_session() -> SessionManager {
        let config = GenerationConfig {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        };
        SessionManager::new_dummy(config, "cpu", "test-model")
    }

    fn create_test_engine() -> InferenceEngine {
        let loader = ModelLoader::dummy();
        let config = GenerationConfig::default();
        InferenceEngine::new(loader, config)
    }

    #[test]
    fn test_repl_creation() {
        let session = create_test_session();
        let engine = create_test_engine();
        let repl = REPL::new(session, engine, true);
        assert!(repl.is_ok());
    }

    #[test]
    fn test_generate_response() {
        let session = create_test_session();
        let engine = create_test_engine();
        let repl = REPL::new(session, engine, false).unwrap();
        let response = repl.generate_response("Hello").unwrap();
        assert!(!response.is_empty());
    }
}
