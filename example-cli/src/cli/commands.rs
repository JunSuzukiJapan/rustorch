use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum Command {
    /// Exit the application
    Exit,
    /// Display help message
    Help,
    /// Clear conversation history
    Clear,
    /// Save conversation history to file
    Save(Option<PathBuf>),
    /// Load conversation history from file
    Load(Option<PathBuf>),
    /// Reload or switch model
    Model(Option<PathBuf>),
    /// Switch backend
    SwitchBackend(String),
    /// Display statistics
    Stats,
    /// Set system prompt
    System(String),
    /// Display current configuration
    Config,
    /// Unknown command
    Unknown(String),
}

impl Command {
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();

        if !input.starts_with('/') {
            anyhow::bail!("Commands must start with '/'");
        }

        let parts: Vec<&str> = input[1..].split_whitespace().collect();

        if parts.is_empty() {
            return Ok(Command::Unknown(input.to_string()));
        }

        let command = parts[0].to_lowercase();
        let args = &parts[1..];

        match command.as_str() {
            "exit" | "quit" | "q" => Ok(Command::Exit),
            "help" | "h" | "?" => Ok(Command::Help),
            "clear" | "cls" => Ok(Command::Clear),
            "save" => {
                let path = args.first().map(PathBuf::from);
                Ok(Command::Save(path))
            }
            "load" => {
                let path = args.first().map(PathBuf::from);
                Ok(Command::Load(path))
            }
            "model" => {
                let path = args.first().map(PathBuf::from);
                Ok(Command::Model(path))
            }
            "backend" => {
                if args.is_empty() {
                    anyhow::bail!("Usage: /backend <cpu|cuda|metal|opencl|hybrid|hybrid-f32>");
                }
                Ok(Command::SwitchBackend(args[0].to_string()))
            }
            "stats" | "status" => Ok(Command::Stats),
            "system" | "sys" => {
                if args.is_empty() {
                    anyhow::bail!("Usage: /system <prompt>");
                }
                let prompt = args.join(" ");
                Ok(Command::System(prompt))
            }
            "config" | "cfg" => Ok(Command::Config),
            _ => Ok(Command::Unknown(input.to_string())),
        }
    }

    pub fn help_text() -> &'static str {
        r#"Available Commands:
  /exit, /quit, /q         Exit the application
  /help, /h, /?            Show this help message
  /clear, /cls             Clear conversation history
  /save [FILE]             Save conversation history (default: latest.json)
  /load [FILE]             Load conversation history
  /model [FILE]            Reload or switch to a different model
  /backend <TYPE>          Switch computation backend
                           Types: cpu, cuda, metal, opencl, hybrid, hybrid-f32
  /stats, /status          Display statistics (tokens, time, etc.)
  /system <PROMPT>         Set system prompt
  /config, /cfg            Display current configuration

Tips:
  - Press Ctrl+C or Ctrl+D to exit
  - Use Up/Down arrows to navigate history
  - Multi-line input: type your message and press Enter
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exit_command() {
        assert!(matches!(Command::parse("/exit").unwrap(), Command::Exit));
        assert!(matches!(Command::parse("/quit").unwrap(), Command::Exit));
        assert!(matches!(Command::parse("/q").unwrap(), Command::Exit));
    }

    #[test]
    fn test_parse_help_command() {
        assert!(matches!(Command::parse("/help").unwrap(), Command::Help));
        assert!(matches!(Command::parse("/h").unwrap(), Command::Help));
        assert!(matches!(Command::parse("/?").unwrap(), Command::Help));
    }

    #[test]
    fn test_parse_clear_command() {
        assert!(matches!(Command::parse("/clear").unwrap(), Command::Clear));
        assert!(matches!(Command::parse("/cls").unwrap(), Command::Clear));
    }

    #[test]
    fn test_parse_save_command() {
        let cmd = Command::parse("/save test.json").unwrap();
        match cmd {
            Command::Save(Some(path)) => assert_eq!(path, PathBuf::from("test.json")),
            _ => panic!("Expected Save command with path"),
        }

        let cmd = Command::parse("/save").unwrap();
        assert!(matches!(cmd, Command::Save(None)));
    }

    #[test]
    fn test_parse_backend_command() {
        let cmd = Command::parse("/backend metal").unwrap();
        match cmd {
            Command::SwitchBackend(backend) => assert_eq!(backend, "metal"),
            _ => panic!("Expected SwitchBackend command"),
        }

        assert!(Command::parse("/backend").is_err());
    }

    #[test]
    fn test_parse_system_command() {
        let cmd = Command::parse("/system You are a helpful assistant").unwrap();
        match cmd {
            Command::System(prompt) => assert_eq!(prompt, "You are a helpful assistant"),
            _ => panic!("Expected System command"),
        }

        assert!(Command::parse("/system").is_err());
    }

    #[test]
    fn test_parse_unknown_command() {
        let cmd = Command::parse("/unknown").unwrap();
        assert!(matches!(cmd, Command::Unknown(_)));
    }

    #[test]
    fn test_parse_without_slash() {
        assert!(Command::parse("help").is_err());
    }
}
