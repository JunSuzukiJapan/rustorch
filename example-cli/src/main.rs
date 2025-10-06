use anyhow::Result;
use clap::Parser;
use colored::Colorize;

use rustorch_cli::{
    init_logger, CliArgs, Commands, Config, DownloadOptions, GenerationConfig, InferenceEngine,
    ModelDownloadManager, ModelIdentifier, ModelLoader, SessionManager, REPL, TuiApp,
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

    // Convert args.backend to device type
    use rustorch_cli::Backend as CliBackend;

    // For hybrid-f32 backend, use F32GPTModel (experimental)
    #[cfg(feature = "hybrid-f32")]
    if matches!(args.backend, CliBackend::HybridF32) {
        use rustorch::hybrid_f32::models::{DeviceType, F32GPTModel, F32LlamaModel, LlamaConfig};
        use rustorch::formats::gguf::GGUFLoader;

        let device_type = DeviceType::Metal;
        tracing::info!("Loading model with hybrid-f32 backend (Metal GPU)");

        // Detect model architecture from filename
        let model_name_lower = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        let is_llama = model_name_lower.contains("llama") ||
                      model_name_lower.contains("mistral") ||
                      model_name_lower.contains("mixtral");

        tracing::info!("🔍 Model filename: {}", model_name_lower);
        tracing::info!("🔍 Is Llama architecture: {}", is_llama);

        if is_llama {
            tracing::info!("🦙 Loading Llama-architecture model with hybrid-f32");

            // Load GGUF to extract config
            match GGUFLoader::from_file(&model_path) {
                Ok(loader) => {
                    match loader.get_model_params() {
                        Ok(params) => {
                            // Calculate num_kv_heads from K weight shape
                            let head_dim = params.hidden_size as usize / params.num_heads as usize;
                            let num_kv_heads = loader.get_tensor("blk.0.attn_k.weight")
                                .and_then(|tensor_info| {
                                    // K weight shape: [hidden_size, num_kv_heads * head_dim]
                                    tensor_info.dims.get(1).map(|&kv_dim| kv_dim as usize / head_dim)
                                })
                                .unwrap_or(params.num_heads as usize); // Fallback to MHA if not found

                            let config = LlamaConfig {
                                vocab_size: params.vocab_size as usize,
                                hidden_size: params.hidden_size as usize,
                                num_layers: params.num_layers as usize,
                                num_heads: params.num_heads as usize,
                                num_kv_heads,
                                intermediate_size: (params.hidden_size * 4) as usize, // Standard 4x expansion
                                max_seq_len: params.context_length as usize,
                                rms_norm_eps: 1e-5,
                                rope_theta: 10000.0,
                            };

                            tracing::info!("📋 Llama Config: vocab={}, hidden={}, layers={}, heads={}, kv_heads={}",
                                config.vocab_size, config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads);

                            match F32LlamaModel::from_gguf_with_config(&model_path, config, device_type) {
                                Ok(llama_model) => {
                                    tracing::info!("✅ F32 Llama model loaded successfully on Metal backend");
                                    engine.set_f32_llama_model(llama_model);
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to load F32 Llama model: {}", e);
                                    tracing::warn!("Falling back to dummy inference");
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to extract model params: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load GGUF: {}", e);
                }
            }
        } else {
            tracing::info!("🤖 Loading GPT-architecture model with hybrid-f32");

            match F32GPTModel::from_gguf_with_device(&model_path, device_type) {
                Ok(f32_model) => {
                    tracing::info!("✅ F32 GPT model loaded successfully on Metal backend");
                    engine.set_f32_gpt_model(f32_model);
                }
                Err(e) => {
                    tracing::warn!("Failed to load F32 GPT model: {}", e);
                    tracing::warn!("Falling back to dummy inference");
                }
            }
        }
    }

    // For mac-hybrid backend, detect model architecture and load accordingly
    #[cfg(feature = "mac-hybrid")]
    if matches!(args.backend, CliBackend::Hybrid) {
        use rustorch::hybrid_f32::models::{DeviceType as F32DeviceType, F32GPTModel, F32LlamaModel, LlamaConfig};
        use rustorch::formats::gguf::GGUFLoader;

        let device_type = F32DeviceType::Hybrid;
        tracing::info!("🚀 Loading model with mac-hybrid backend (Metal/CoreML)");

        // Detect model architecture from filename or weights
        let model_name_lower = model_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        let is_llama = model_name_lower.contains("llama") ||
                      model_name_lower.contains("mistral") ||
                      model_name_lower.contains("mixtral");

        tracing::info!("🔍 Model filename: {}", model_name_lower);
        tracing::info!("🔍 Is Llama architecture: {}", is_llama);

        if is_llama {
            tracing::info!("🦙 Detected Llama-architecture model");

            // Load GGUF to extract config
            match GGUFLoader::from_file(&model_path) {
                Ok(loader) => {
                    match loader.get_model_params() {
                        Ok(params) => {
                            // Create Llama config from params
                            let llama_config = LlamaConfig {
                                vocab_size: params.vocab_size as usize,
                                hidden_size: params.hidden_size as usize,
                                num_layers: params.num_layers as usize,
                                num_heads: params.num_heads as usize,
                                num_kv_heads: params.num_heads as usize, // Default to MHA (same as num_heads)
                                intermediate_size: (params.hidden_size * 4) as usize, // Standard 4x hidden size
                                max_seq_len: params.context_length as usize,
                                rms_norm_eps: 1e-5,
                                rope_theta: 10000.0,
                            };

                            match F32LlamaModel::from_gguf_with_config(&model_path, llama_config, F32DeviceType::Hybrid) {
                                Ok(llama_model) => {
                                    tracing::info!("✅ F32 Llama model loaded successfully on mac-hybrid backend");
                                    engine.set_f32_llama_model(llama_model);
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to load F32 Llama model: {}", e);
                                    tracing::warn!("Falling back to GPT architecture");
                                    // Try GPT as fallback
                                    if let Ok(gpt_model) = F32GPTModel::from_gguf_with_device(&model_path, device_type) {
                                        engine.set_f32_gpt_model(gpt_model);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to get model params: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load GGUF: {}", e);
                }
            }
        } else {
            tracing::info!("📝 Detected GPT-architecture model");
            match F32GPTModel::from_gguf_with_device(&model_path, device_type) {
                Ok(f32_model) => {
                    tracing::info!("✅ F32 GPT model loaded successfully on mac-hybrid backend");
                    engine.set_f32_gpt_model(f32_model);
                }
                Err(e) => {
                    tracing::warn!("Failed to load F32 GPT model: {}", e);
                    tracing::warn!("Falling back to dummy inference");
                }
            }
        }
    }

    // For other backends, use standard GPTModel (f64, CPU-only)
    #[cfg(not(any(feature = "hybrid-f32", feature = "mac-hybrid")))]
    let use_standard_model = true;
    #[cfg(all(feature = "hybrid-f32", not(feature = "mac-hybrid")))]
    let use_standard_model = !matches!(args.backend, CliBackend::HybridF32);
    #[cfg(feature = "mac-hybrid")]
    let use_standard_model = !matches!(args.backend, CliBackend::Hybrid | CliBackend::HybridF32);

    if use_standard_model {
        let device_type = match args.backend {
            CliBackend::Cpu => rustorch::backends::DeviceType::Cpu,
            CliBackend::Cuda => rustorch::backends::DeviceType::Cuda,
            CliBackend::Metal => {
                tracing::warn!("⚠️  Metal backend uses f64 GPTModel (no GPU acceleration)");
                tracing::warn!("   Use --backend hybrid with --features mac-hybrid for f32 GPU acceleration");
                rustorch::backends::DeviceType::Metal
            }
            CliBackend::Opencl => rustorch::backends::DeviceType::OpenCL,
            CliBackend::Hybrid => {
                #[cfg(not(feature = "mac-hybrid"))]
                {
                    tracing::error!("❌ mac-hybrid feature not enabled!");
                    tracing::error!("   Build with: cargo build --features mac-hybrid");
                    tracing::error!("   Falling back to CPU (f64)");
                    rustorch::backends::DeviceType::Cpu
                }
                #[cfg(feature = "mac-hybrid")]
                {
                    // Unreachable when mac-hybrid is enabled (handled above)
                    rustorch::backends::DeviceType::Cpu
                }
            }
            CliBackend::HybridF32 => {
                #[cfg(not(feature = "hybrid-f32"))]
                {
                    tracing::error!("❌ hybrid-f32 feature not enabled!");
                    tracing::error!("   Build with: cargo build --features hybrid-f32");
                    tracing::error!("   Or use --backend hybrid with --features mac-hybrid for CoreML support");
                    rustorch::backends::DeviceType::Cpu
                }
                #[cfg(feature = "hybrid-f32")]
                {
                    // Unreachable when hybrid-f32 is enabled (handled above)
                    rustorch::backends::DeviceType::Cpu
                }
            }
        };

        match rustorch::models::GPTModel::from_gguf_with_backend(&model_path, device_type) {
            Ok(gpt_model) => {
                tracing::info!("✅ GPTModel (f64) loaded successfully on {:?} backend", gpt_model.device_type());
                tracing::info!("   Note: f64 precision, CPU-only (no GPU acceleration)");
                engine.set_gpt_model(gpt_model);
            }
            Err(e) => {
                tracing::warn!("Failed to load RusTorch GPT model: {}", e);
                tracing::warn!("Falling back to dummy inference");
            }
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
