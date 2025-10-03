use anyhow::Result;
use clap::Parser;

use rustorch_cli::{CliArgs, GenerationConfig, SessionManager, ModelLoader, InferenceEngine, REPL, init_logger};

fn main() -> Result<()> {
    // Parse command line arguments
    let args = CliArgs::parse();

    // Initialize logger
    init_logger(args.log_level);

    // Validate arguments
    args.validate()?;

    // Log startup information
    tracing::info!("RusTorch CLI starting...");
    tracing::info!("Backend: {}", args.backend.as_str());
    tracing::debug!("Configuration: {:?}", args);

    // Create generation config from CLI args
    let gen_config = GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    };

    // Validate generation config
    gen_config.validate()?;

    // Load model
    let model_loader = if let Some(path) = &args.model {
        tracing::info!("Loading model from: {}", path.display());
        ModelLoader::from_file(path)?
    } else {
        // TODO: Try to load from config file
        tracing::warn!("No model specified, using dummy model");
        ModelLoader::dummy()
    };

    let model_name = model_loader.metadata().name.clone();

    // Create inference engine
    let engine = InferenceEngine::new(model_loader, gen_config.clone());

    // Create session manager
    let mut session = SessionManager::new(
        gen_config,
        args.backend.as_str(),
        &model_name,
    );

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

    // Create and run REPL
    let mut repl = REPL::new(session, engine, !args.no_progress)?;
    repl.run()?;

    tracing::info!("RusTorch CLI exiting...");

    Ok(())
}
