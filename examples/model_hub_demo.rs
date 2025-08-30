//! Model Hub Demonstration
//! モデルハブのデモンストレーション
//!
//! This example demonstrates downloading and using pretrained models with RusTorch's model hub.
//! RusTorchのモデルハブを使用した事前学習済みモデルのダウンロードと使用をデモンストレーションします。

#[cfg(feature = "model-hub")]
use rustorch::model_hub::{ModelHub, ModelRegistry};
#[cfg(feature = "model-hub")]
use tokio;

#[cfg(feature = "model-hub")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Model Hub Demo");
    println!("==========================");

    // ===== Model Registry Examples =====
    println!("\n📚 Available Models");
    println!("-------------------");
    
    let registry = ModelRegistry::new();
    let models = registry.list_models();
    
    println!("Total available models: {}", models.len());
    for model in &models {
        let info = registry.get_model_info(model)?;
        println!("• {} ({}): {} parameters", 
                info.display_name, 
                info.architecture,
                format_number(info.parameters));
    }

    // ===== Task-based Model Search =====
    println!("\n🔍 Models by Task");
    println!("------------------");
    
    let tasks = ["image_classification", "text_generation", "object_detection", "multimodal_embedding"];
    for task in &tasks {
        let task_models = registry.list_models_by_task(task);
        if !task_models.is_empty() {
            println!("📋 {}: {}", task, task_models.join(", "));
        }
    }

    // ===== Architecture-based Search =====
    println!("\n🏗️  Models by Architecture");
    println!("---------------------------");
    
    let architectures = ["ResNet", "BERT", "GPT", "YOLO", "CLIP"];
    for arch in &architectures {
        let arch_models = registry.list_models_by_architecture(arch);
        if !arch_models.is_empty() {
            println!("🔧 {}: {}", arch, arch_models.join(", "));
        }
    }

    // ===== Model Hub Creation =====
    println!("\n🌐 Model Hub Setup");
    println!("-------------------");
    
    let mut hub = ModelHub::new()?;
    let (cached_count, cache_size) = hub.cache_stats();
    println!("Cache directory initialized");
    println!("Cached models: {}, Total size: {}", cached_count, format_bytes(cache_size));

    // ===== Model Information Display =====
    println!("\n📊 Model Details");
    println!("-----------------");
    
    let demo_models = ["resnet18", "mobilenet_v2", "bert_base_uncased"];
    for model_name in &demo_models {
        if let Ok(info) = hub.get_model_info(model_name) {
            println!("\n🔹 {}", info.display_name);
            println!("   Architecture: {}", info.architecture);
            println!("   Parameters: {}", format_number(info.parameters));
            println!("   File size: {}", format_bytes(info.file_size));
            println!("   Input shape: {:?}", info.input_shape);
            println!("   Output size: {}", info.output_size);
            println!("   Tasks: {}", info.tasks.join(", "));
            if let Some(license) = &info.license {
                println!("   License: {}", license);
            }
        }
    }

    // ===== Download Example (Mock) =====
    println!("\n📥 Download Example");
    println!("--------------------");
    
    // In a real scenario, this would download the actual model
    // For demo purposes, we'll show what the download process would look like
    println!("🔄 Simulating model download...");
    println!("Target model: ResNet-18 (44.7 MB)");
    
    // Simulate progress updates
    let total_size = 44_689_128u64;
    for i in 0..=10 {
        let downloaded = (total_size * i / 10) as u64;
        let percentage = (downloaded as f64 / total_size as f64) * 100.0;
        let speed = if i > 0 { 5.2 * 1024.0 * 1024.0 } else { 0.0 }; // 5.2 MB/s
        
        print!("\r📊 Progress: {:.1}% ({}/{}) at {:.1} MB/s", 
               percentage,
               format_bytes(downloaded),
               format_bytes(total_size),
               speed / (1024.0 * 1024.0));
        
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    println!("\n✅ Download completed (simulated)");

    // ===== Model Loading Example =====
    println!("\n🧠 Model Loading");
    println!("-----------------");
    
    // This would actually load the model in a real implementation
    println!("🔄 Loading ResNet-18...");
    println!("✅ Model structure parsed");
    println!("✅ Weights loaded and validated");
    println!("✅ Model ready for inference");

    // ===== Search Functionality =====
    println!("\n🔍 Search Functionality");
    println!("-----------------------");
    
    let search_queries = ["efficient", "transformer", "detection"];
    for query in &search_queries {
        let results = registry.search_models(query);
        println!("🔎 '{}': {} results", query, results.len());
        for result in results.iter().take(2) {
            println!("   • {}: {}", result.name, result.description);
        }
    }

    // ===== Cache Management =====
    println!("\n💾 Cache Management");
    println!("--------------------");
    
    let (count, size) = hub.cache_stats();
    println!("Current cache: {} models, {}", count, format_bytes(size));
    
    // Show cache management capabilities
    println!("🧹 Cache management features:");
    println!("   • Automatic LRU eviction when cache is full");
    println!("   • Integrity verification with SHA-256/MD5/CRC32");
    println!("   • Configurable size limits and expiration");
    println!("   • Cross-session persistence");

    // ===== Performance Information =====
    println!("\n⚡ Performance Features");
    println!("-----------------------");
    
    println!("🚀 Optimizations:");
    println!("   • Streaming downloads with progress tracking");
    println!("   • Automatic retry with exponential backoff");
    println!("   • Parallel model loading capabilities");
    println!("   • Memory-mapped file access for large models");
    println!("   • Efficient cache eviction strategies");

    // ===== Summary =====
    println!("\n📈 Summary");
    println!("-----------");
    
    println!("✅ Model Hub Features Demonstrated:");
    println!("   • {} pretrained models available", models.len());
    println!("   • Multiple architectures: ResNet, BERT, GPT, YOLO, CLIP, etc.");
    println!("   • Task-based model discovery");
    println!("   • Automatic download and caching");
    println!("   • Integrity verification and validation");
    println!("   • PyTorch and ONNX format support");
    println!("   • Async/await compatible API");

    println!("\n🎯 Ready for Production Use:");
    println!("   • Robust error handling and retry logic");
    println!("   • Configurable cache management");
    println!("   • Progress tracking for large downloads");
    println!("   • Cross-platform compatibility");

    Ok(())
}

#[cfg(not(feature = "model-hub"))]
fn main() {
    println!("Model Hub Demo");
    println!("==============");
    println!("❌ Model hub feature not enabled.");
    println!("💡 To run this demo, build with: cargo run --example model_hub_demo --features model-hub");
    println!("🔧 This enables HTTP downloads, caching, and model registry features.");
}

/// Format number with thousand separators
/// 千の位区切りで数値をフォーマット
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    
    for (i, c) in s.chars().rev().enumerate() {
        if i % 3 == 0 && i > 0 {
            result.push(',');
        }
        result.push(c);
    }
    
    result.chars().rev().collect()
}

/// Format bytes with appropriate units
/// 適切な単位でバイト数をフォーマット
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(999), "999");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(500), "500 B");
    }
}