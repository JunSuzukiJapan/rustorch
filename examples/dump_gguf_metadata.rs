/// Example utility to dump GGUF metadata for debugging
///
/// Usage:
///   cargo run --example dump_gguf_metadata -- <path-to-gguf-file> [prefix-filter]
///
/// Examples:
///   cargo run --example dump_gguf_metadata -- model.gguf
///   cargo run --example dump_gguf_metadata -- model.gguf general
///   cargo run --example dump_gguf_metadata -- model.gguf llama

use rustorch::formats::gguf::GGUFLoader;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file> [prefix-filter]", args[0]);
        eprintln!("\nExamples:");
        eprintln!("  {} model.gguf", args[0]);
        eprintln!("  {} model.gguf general", args[0]);
        eprintln!("  {} model.gguf llama", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let filter_prefix = args.get(2).map(|s| s.as_str());

    // Expand ~ to home directory
    let expanded_path = if model_path.starts_with("~/") {
        let home = env::var("HOME").expect("HOME environment variable not set");
        model_path.replacen("~", &home, 1)
    } else {
        model_path.to_string()
    };

    println!("Loading GGUF file: {}", expanded_path);

    match GGUFLoader::from_file(&expanded_path) {
        Ok(loader) => {
            // Print model info summary
            println!("\n{}", loader.get_model_info());

            // Dump metadata
            loader.dump_metadata(filter_prefix);

            // Print some useful key groups
            println!("\n=== Key Groups ===");
            let prefixes = ["general.", "llama.", "tokenizer.", "gpt2.", "mistral.", "phi.", "gemma.", "qwen."];
            for prefix in &prefixes {
                let keys = loader.get_keys_with_prefix(prefix);
                if !keys.is_empty() {
                    println!("  {}: {} keys", prefix, keys.len());
                }
            }

            println!("\n✅ Metadata dump complete");
        }
        Err(e) => {
            eprintln!("❌ Error loading GGUF file: {:?}", e);
            std::process::exit(1);
        }
    }
}
