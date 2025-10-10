use anyhow::Result;
use clap::Parser;
use colored::Colorize;

use rustorch_cli::{
    init_logger, BackendLoader, CliArgs, Commands, Config, DownloadOptions, GenerationConfig,
    InferenceEngine, ModelDownloadManager, ModelIdentifier, ModelLoader, SessionManager, REPL,
    TuiApp,
};

fn main() -> Result<()> {
    // Parse command line arguments
    let args = CliArgs::parse();

    // Handle subcommands first
    if let Some(command) = &args.command {
        return handle_command(command);
    }

    // Initialize logger
    init_logger(args.log_level);

    // Validate arguments
    args.validate()?;

    // Start CLI
    start_cli(args)?;

    Ok(())
}

/// Start the CLI REPL interface
fn start_cli(args: CliArgs) -> Result<()> {
    // Log startup information
    tracing::info!("RusTorch CLI starting...");
    tracing::info!("Backend: {}", args.backend.as_str());
    tracing::debug!("Configuration: {:?}", args);

    // Load config file (if exists)
    let file_config = Config::load_default().unwrap_or_else(|e| {
        tracing::debug!("Could not load config file: {}", e);
        Config::default()
    });

    // Merge configs: CLI args override config file
    let gen_config = GenerationConfig::merge_from_cli(&file_config.generation, &args);

    // Validate generation config
    gen_config.validate()?;

    // Load model
    let model_loader = if let Some(path) = &args.model {
        tracing::info!("Loading model from: {}", path.display());
        ModelLoader::from_file(path)?
    } else if let Some(default_model) = file_config.model.default.as_ref() {
        // Try to load default model from config file
        tracing::info!("Loading default model from config: {}", default_model);
        ModelLoader::from_file(default_model)?
    } else {
        anyhow::bail!(
            "No model specified. Please provide a model path using:\n\
             - Command line: --model <path>\n\
             - Config file: set model.default in config file\n\
             - Environment: Use --config to specify config file path"
        );
    };

    let model_name = model_loader.metadata().name.clone();
    let model_path = model_loader.path().to_path_buf();

    // Create inference engine
    let mut engine = InferenceEngine::new(model_loader, gen_config.clone());

    // Load model with specified backend using BackendLoader
    tracing::info!("Loading model from: {}", model_path.display());
    BackendLoader::load(&args.backend, &model_path, &mut engine)?;
    // If --tokens is provided, run single generation and exit
    if let Some(tokens_str) = &args.tokens {
        tracing::info!("Direct token input mode");

        // Parse comma-separated token IDs
        let token_ids: Result<Vec<u32>> = tokens_str
            .split(',')
            .map(|s| s.trim().parse::<u32>()
                .map_err(|e| anyhow::anyhow!("Invalid token ID '{}': {}", s, e)))
            .collect();

        let token_ids = token_ids?;

        println!("🔍 Input tokens: {:?}", token_ids);

        // Generate from tokens directly
        let output = engine.generate_from_tokens(token_ids)?;

        println!("\n📝 Output:\n{}", output);

        return Ok(());
    }

    // Create session manager
    let mut session = SessionManager::new(gen_config, args.backend.as_str(), &model_name);

    // Set system prompt if provided
    if let Some(prompt) = &args.system_prompt {
        session.set_system_prompt(prompt);
    }

    // Enable auto-save if specified
    if let Some(path) = &args.save_history {
        session = session.with_auto_save(path);
    }

    // Load history if specified
    if let Some(path) = &args.load_history {
        if path.exists() {
            session.load_history(path)?;
            tracing::info!("Loaded conversation history from: {}", path.display());
        } else {
            tracing::warn!("History file not found: {}", path.display());
        }
    }

    // Auto-detect chat template requirement from model name
    let use_template = should_use_chat_template(&model_name);

    // Choose between TUI mode and REPL mode
    if args.tui {
        // Run TUI mode
        let mut tui = TuiApp::new(session, engine, use_template);
        tui.run()?;
    } else {
        // Run traditional REPL mode
        let mut repl = REPL::new(session, engine, !args.no_progress)?;
        repl.set_use_chat_template(use_template);

        // Notify user about auto-detection
        if use_template {
            println!(
                "{} {}",
                "ℹ️  Chat/Instruct model detected:".bright_cyan(),
                "template enabled".bright_green()
            );
            println!(
                "   {}",
                "Use /template to toggle if needed".bright_black()
            );
        } else {
            println!(
                "{} {}",
                "ℹ️  Base/Completion model detected:".bright_cyan(),
                "template disabled".bright_yellow()
            );
            println!(
                "   {}",
                "Use /template to enable if needed".bright_black()
            );
        }
        println!();

        repl.run()?;
    }

    tracing::info!("RusTorch CLI exiting...");

    Ok(())
}

/// Detect if model requires chat template based on name
fn should_use_chat_template(model_name: &str) -> bool {
    let name_lower = model_name.to_lowercase();

    // Keywords indicating chat/instruct models
    let chat_keywords = [
        "chat",
        "instruct",
        "assistant",
        "conversation",
        "dialogue",
    ];

    chat_keywords.iter().any(|&keyword| name_lower.contains(keyword))
}

/// Handle subcommands
fn handle_command(command: &Commands) -> Result<()> {
    match command {
        Commands::Download {
            model_id,
            output_dir,
            format,
            quantization,
            force,
            token,
        } => {
            println!("🚀 RusTorch Model Downloader\n");

            // Parse model identifier
            let identifier = ModelIdentifier::parse(model_id)?;

            // Prepare download options
            let mut options = DownloadOptions::default();

            if let Some(dir) = output_dir {
                options.output_dir = dir.clone();
            }

            options.format = format.clone();
            options.quantization = quantization.clone();
            options.force = *force;
            options.token = token.clone();

            // Create download manager
            let manager = ModelDownloadManager::new()?;

            // Download model
            let path = manager.download(&identifier, &options)?;

            println!("\n✅ Model downloaded successfully!");
            println!("📂 Path: {}", path.display());
            println!("\n🚀 Starting CLI with downloaded model...\n");

            // Create new args with the downloaded model
            let mut cli_args = CliArgs::parse();
            cli_args.model = Some(path);
            cli_args.command = None; // Clear the download command

            // Initialize logger
            init_logger(cli_args.log_level);

            // Validate arguments
            cli_args.validate()?;

            // Continue with normal CLI startup
            start_cli(cli_args)?;

            Ok(())
        }

        Commands::List { source } => {
            use rustorch_cli::download::ModelSource;

            println!("📋 Listing models from {}...\n", source);

            let source = ModelSource::from_prefix(source)?;
            let manager = ModelDownloadManager::new()?;

            let models = manager.list_models(source)?;

            if models.is_empty() {
                println!("No models found.");
            } else {
                println!("Available models:");
                for (i, model) in models.iter().enumerate() {
                    println!("  {}. {}", i + 1, model);
                }
                println!("\nTotal: {} models", models.len());
            }

            Ok(())
        }
    }
}
