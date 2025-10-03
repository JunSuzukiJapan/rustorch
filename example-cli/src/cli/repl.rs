use anyhow::Result;
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
}

impl REPL {
    pub fn new(session: SessionManager, engine: InferenceEngine, show_progress: bool) -> Result<Self> {
        let editor = DefaultEditor::new()?;
        Ok(Self {
            editor,
            session,
            engine,
            show_progress,
        })
    }

    pub fn run(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            match self.editor.readline("You> ") {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    self.editor.add_history_entry(line)?;

                    if line.starts_with('/') {
                        match Command::parse(line) {
                            Ok(cmd) => {
                                if !self.handle_command(cmd)? {
                                    break; // Exit command
                                }
                            }
                            Err(e) => {
                                eprintln!("Error: {}", e);
                            }
                        }
                    } else {
                        if let Err(e) = self.handle_message(line) {
                            eprintln!("Error: {}", e);
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    println!("Use /exit or /quit to exit, or press Ctrl+D");
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

    fn print_welcome(&self) {
        println!("╔════════════════════════════════════════════════════════════╗");
        println!("║           RusTorch CLI - Local LLM Chat                   ║");
        println!("╚════════════════════════════════════════════════════════════╝");
        println!();
        println!("Backend: {}", self.session.backend_name());
        println!("Model: {}", self.session.model_name());
        println!();
        println!("Type '/help' for available commands, '/exit' to quit.");
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
            Command::Unknown(cmd) => {
                println!("Unknown command: {}", cmd);
                println!("Type '/help' for available commands.");
            }
        }

        Ok(true)
    }

    fn handle_exit(&mut self) -> Result<()> {
        println!("Saving session...");
        if let Err(e) = self.session.auto_save() {
            eprintln!("Warning: Failed to auto-save session: {}", e);
        }
        println!("Goodbye!");
        Ok(())
    }

    fn handle_help(&self) {
        println!("{}", Command::help_text());
    }

    fn handle_clear(&mut self) -> Result<()> {
        self.session.clear_history();
        println!("Conversation history cleared.");
        Ok(())
    }

    fn handle_save(&mut self, path: Option<PathBuf>) -> Result<()> {
        let path = path.unwrap_or_else(|| {
            crate::cli::CliArgs::get_default_history_path()
        });
        self.session.save_history(&path)?;
        println!("Conversation saved to: {}", path.display());
        Ok(())
    }

    fn handle_load(&mut self, path: Option<PathBuf>) -> Result<()> {
        let path = path.unwrap_or_else(|| {
            crate::cli::CliArgs::get_default_history_path()
        });
        self.session.load_history(&path)?;
        println!("Conversation loaded from: {}", path.display());
        Ok(())
    }

    fn handle_model(&mut self, _path: Option<PathBuf>) -> Result<()> {
        println!("Model switching is not yet implemented.");
        Ok(())
    }

    fn handle_switch_backend(&mut self, backend: &str) -> Result<()> {
        println!("Backend switching is not yet implemented.");
        println!("Requested backend: {}", backend);
        Ok(())
    }

    fn handle_stats(&self) {
        println!("╔════════════════════════════════════════════════════════════╗");
        println!("║                     Statistics                             ║");
        println!("╚════════════════════════════════════════════════════════════╝");
        println!();
        println!("  Messages:        {}", self.session.message_count());
        println!("  Total tokens:    {}", self.session.total_tokens());
        println!("  Backend:         {}", self.session.backend_name());
        println!("  Model:           {}", self.session.model_name());
        println!();
    }

    fn handle_system(&mut self, prompt: &str) -> Result<()> {
        self.session.set_system_prompt(prompt);
        println!("System prompt updated.");
        Ok(())
    }

    fn handle_config(&self) {
        println!("╔════════════════════════════════════════════════════════════╗");
        println!("║                   Configuration                            ║");
        println!("╚════════════════════════════════════════════════════════════╝");
        println!();
        self.session.print_config();
        println!();
    }

    fn handle_message(&mut self, message: &str) -> Result<()> {
        self.session.add_user_message(message);

        let response = if self.show_progress {
            let progress = ProgressIndicator::new("Thinking...");
            let result = self.generate_response(message);
            progress.finish();
            result?
        } else {
            self.generate_response(message)?
        };

        println!("Assistant> {}", response);
        self.session.add_assistant_message(&response);

        Ok(())
    }

    fn generate_response(&self, message: &str) -> Result<String> {
        self.engine.generate(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::GenerationConfig;
    use crate::model::{ModelLoader, InferenceEngine};

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
