use anyhow::Result;
use clap::Parser;

use rustorch_cli::{
    init_logger, CliArgs, Config, GenerationConfig, InferenceEngine, ModelLoader, SessionManager,
    REPL,
};

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

    // Load config file (if exists)
    let file_config = Config::load_default().unwrap_or_else(|e| {
        tracing::debug!("Could not load config file: {}", e);
        Config::default()
    });

    // Merge configs: CLI args override config file
    // Use CLI args if provided, otherwise use config file values
    let max_tokens = if args.max_tokens != 512 {
        args.max_tokens
    } else {
        file_config.generation.max_tokens
    };

    let temperature = if (args.temperature - 0.7).abs() > f32::EPSILON {
        args.temperature
    } else {
        file_config.generation.temperature
    };

    let top_p = if (args.top_p - 0.9).abs() > f32::EPSILON {
        args.top_p
    } else {
        file_config.generation.top_p
    };

    let top_k = if args.top_k != 40 {
        args.top_k
    } else {
        file_config.generation.top_k as u32
    };

    // Create generation config from merged values
    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
        top_k,
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

    // Create and run REPL
    let mut repl = REPL::new(session, engine, !args.no_progress)?;
    repl.run()?;

    tracing::info!("RusTorch CLI exiting...");

    Ok(())
}
