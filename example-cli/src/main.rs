use anyhow::Result;
use clap::Parser;

use rustorch_cli::{
    init_logger, CliArgs, Commands, Config, DownloadOptions, GenerationConfig, InferenceEngine,
    ModelDownloadManager, ModelIdentifier, ModelLoader, SessionManager, REPL,
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

    // Load RusTorch GPT model with specified backend
    tracing::info!("Loading RusTorch GPT model from: {}", model_path.display());

    // Convert args.backend to rustorch::backends::DeviceType
    use rustorch_cli::Backend as CliBackend;
    let device_type = match args.backend {
        CliBackend::Cpu => rustorch::backends::DeviceType::Cpu,
        CliBackend::Cuda => rustorch::backends::DeviceType::Cuda,
        CliBackend::Metal => rustorch::backends::DeviceType::Metal,
        CliBackend::Opencl => rustorch::backends::DeviceType::OpenCL,
        CliBackend::Hybrid | CliBackend::HybridF32 => {
            // For hybrid, prefer Metal on macOS, otherwise CPU
            #[cfg(target_os = "macos")]
            {
                rustorch::backends::DeviceType::Metal
            }
            #[cfg(not(target_os = "macos"))]
            {
                rustorch::backends::DeviceType::Cpu
            }
        }
    };

    match rustorch::models::GPTModel::from_gguf_with_backend(&model_path, device_type) {
        Ok(gpt_model) => {
            tracing::info!("✅ RusTorch GPT model loaded successfully on {:?} backend", gpt_model.device_type());
            engine.set_gpt_model(gpt_model);
        }
        Err(e) => {
            tracing::warn!("Failed to load RusTorch GPT model: {}", e);
            tracing::warn!("Falling back to dummy inference");
        }
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

    // Create and run REPL
    let mut repl = REPL::new(session, engine, !args.no_progress)?;
    repl.run()?;

    tracing::info!("RusTorch CLI exiting...");

    Ok(())
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
