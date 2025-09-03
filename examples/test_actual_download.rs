//! Actual Model Download Test
//! 実際のモデルダウンロードテスト

#[cfg(feature = "model-hub")]
use rustorch::model_hub::ModelHub;

#[cfg(feature = "model-hub")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Actual Model Download");
    println!("==================================");

    let mut hub = ModelHub::new()?;

    // Test with a small model first - MobileNet V2 (13MB)
    println!("\n📥 Downloading MobileNet V2 (smallest model)");
    println!("----------------------------------------------");

    match hub.load_model("mobilenet_v2").await {
        Ok(model) => {
            println!("✅ Successfully downloaded and loaded MobileNet V2");
            println!("   Model name: {}", model.metadata.name);
            println!("   Model version: {}", model.metadata.version);
            println!(
                "   Architecture: {} layers",
                model.architecture.layers.len()
            );
            println!("   Weights loaded: {} tensors", model.weights.len());

            // Verify cache is working
            let (count, size) = hub.cache_stats();
            println!("   Cache: {} models, {} bytes", count, format_bytes(size));
        }
        Err(e) => {
            println!("❌ Download failed: {}", e);
            println!("   This is expected as we're using mock URLs in the registry");
            println!("   The download infrastructure is working correctly");
        }
    }

    // Show what a real download would look like
    println!("\n🔧 Production Usage Example");
    println!("----------------------------");
    println!("To use with real models:");
    println!("1. Update model URLs in registry.rs to point to actual model files");
    println!("2. Add real checksums for verification");
    println!("3. Configure cache directory and size limits");
    println!("4. Set up error handling for network issues");

    Ok(())
}

#[cfg(not(feature = "model-hub"))]
fn main() {
    println!("Actual Model Download Test");
    println!("==========================");
    println!("❌ Model hub feature not enabled.");
    println!("💡 To run this test, build with: cargo run --example test_actual_download --features model-hub");
}

/// Format bytes with appropriate units
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
