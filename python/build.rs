use std::env;
use std::process::Command;

fn main() {
    // Check if we're building on macOS and detect Framework Python
    if cfg!(target_os = "macos") {
        check_macos_python_configuration();
    }
}

fn check_macos_python_configuration() {
    let python_executable = env::var("PYO3_PYTHON")
        .or_else(|_| env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());

    // Check Python configuration
    let output = Command::new(&python_executable)
        .args(&["-c", "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORK'))"])
        .output();

    match output {
        Ok(output) => {
            let framework_output = String::from_utf8_lossy(&output.stdout);
            let framework = framework_output.trim();
            
            if framework == "Python" {
                // This is Framework Python - check if we have proper build config
                check_framework_python_build_config(&python_executable);
            }
        }
        Err(_) => {
            eprintln!("cargo:warning=Could not detect Python configuration");
        }
    }
}

fn check_framework_python_build_config(python_executable: &str) {
    // Check if .cargo/config.toml exists with proper settings
    let config_path = std::path::Path::new(".cargo/config.toml");
    
    if !config_path.exists() {
        print_framework_python_help(python_executable);
        panic!("Missing .cargo/config.toml for Framework Python build");
    }
    
    // Read the config file and check for dynamic_lookup
    if let Ok(config_content) = std::fs::read_to_string(config_path) {
        if !config_content.contains("dynamic_lookup") {
            print_framework_python_help(python_executable);
            panic!("Missing dynamic_lookup configuration for Framework Python");
        }
    }
}

fn print_framework_python_help(python_executable: &str) {
    eprintln!("\n📱 Framework Python detected on macOS!");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!();
    eprintln!("🔧 You're using Framework Python (installed via Homebrew or Python.org).");
    eprintln!("   PyO3 requires special configuration to work with Framework Python.");
    eprintln!();
    eprintln!("💡 SOLUTION: Create a .cargo/config.toml file with:");
    eprintln!();
    eprintln!("   [build]");
    eprintln!("   rustflags = [");
    eprintln!("       \"-C\", \"link-arg=-undefined\",");
    eprintln!("       \"-C\", \"link-arg=dynamic_lookup\",");
    eprintln!("   ]");
    eprintln!();
    eprintln!("   [env]");
    eprintln!("   PYO3_PYTHON = \"{}\"", python_executable);
    eprintln!();
    eprintln!("🚀 Then run: cargo clean && cargo build --release");
    eprintln!();
    eprintln!("📚 Alternative solutions:");
    eprintln!("   • Use pyenv with a compiled Python: pyenv install 3.9.18");
    eprintln!("   • Use conda/miniconda Python environment");
    eprintln!("   • Use system Python (if available): /usr/bin/python3");
    eprintln!();
    eprintln!("🔗 More info: https://pyo3.rs/v0.20.3/building_and_distribution");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}