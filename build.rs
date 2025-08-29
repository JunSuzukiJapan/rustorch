//! Build script for `RusTorch`
//! `RusTorch`用ビルドスクリプト
//!
//! This build script ensures proper linking of LAPACK/BLAS libraries
//! when using the linalg-system feature.

use std::env;

#[allow(clippy::too_many_lines)] // Build script with platform-specific logic requires many lines
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check if we're building with any LAPACK/BLAS feature
    let has_linalg = env::var("CARGO_FEATURE_LINALG").is_ok();
    let has_linalg_system = env::var("CARGO_FEATURE_LINALG_SYSTEM").is_ok();
    let has_blas_optimized = env::var("CARGO_FEATURE_BLAS_OPTIMIZED").is_ok();

    if has_linalg || has_linalg_system || has_blas_optimized {
        // Windows requires special handling for LAPACK/BLAS
        if cfg!(target_os = "windows") {
            // On Windows, use system libraries or disable linalg features
            println!(
                "cargo:warning=Windows LAPACK/BLAS support limited - use --no-default-features"
            );
        } else {
            // Unix systems (Linux, macOS) - explicit linking

            // OS-specific library paths for Linux
            if cfg!(target_os = "linux") {
                println!("cargo:rustc-link-search=native=/usr/lib");
                println!("cargo:rustc-link-search=native=/usr/local/lib");

                // Multi-architecture support with Ubuntu-specific paths
                if cfg!(target_arch = "x86_64") {
                    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
                    println!(
                        "cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread"
                    );
                    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/blas");
                    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/lapack");
                    // Additional Ubuntu library paths for CI/CD environments
                    println!("cargo:rustc-link-search=native=/usr/lib64");
                    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
                } else if cfg!(target_arch = "aarch64") {
                    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
                    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/openblas-pthread");
                    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/blas");
                    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu/lapack");
                    println!("cargo:rustc-link-search=native=/usr/lib64");
                    println!("cargo:rustc-link-search=native=/lib/aarch64-linux-gnu");
                }
            }

            // Link LAPACK and BLAS libraries with specific library names for better compatibility
            if cfg!(target_os = "linux") {
                // Only link system libraries if NOT using linalg-system
                // Check for explicit BLAS/LAPACK library preferences
                let _blas_lib = env::var("BLAS_LIB")
                    .or_else(|_| env::var("RUSTORCH_BLAS_LIB"))
                    .unwrap_or_else(|_| "openblas".to_string());
                let _lapack_lib = env::var("LAPACK_LIB")
                    .or_else(|_| env::var("RUSTORCH_LAPACK_LIB"))
                    .unwrap_or_else(|_| "openblas".to_string());

                // Check for available BLAS/LAPACK libraries with multi-arch support
                let arch_paths = if cfg!(target_arch = "x86_64") {
                    vec!["/usr/lib/x86_64-linux-gnu", "/usr/lib64"]
                } else if cfg!(target_arch = "aarch64") {
                    vec!["/usr/lib/aarch64-linux-gnu", "/usr/lib64"]
                } else {
                    vec!["/usr/lib"]
                };

                let mut openblas_available = false;
                let mut separate_blas_available = false;
                let mut separate_lapack_available = false;

                for path in &arch_paths {
                    // Check OpenBLAS variants
                    if std::path::Path::new(&format!("{path}/libopenblas.so.0")).exists()
                        || std::path::Path::new(&format!("{path}/libopenblas.so")).exists()
                        || std::path::Path::new(&format!("{path}/libopenblas.a")).exists()
                    {
                        openblas_available = true;
                    }

                    // Check separate BLAS libraries
                    if std::path::Path::new(&format!("{path}/libblas.so")).exists()
                        || std::path::Path::new(&format!("{path}/libblas.a")).exists()
                    {
                        separate_blas_available = true;
                    }

                    // Check separate LAPACK libraries
                    if std::path::Path::new(&format!("{path}/liblapack.so")).exists()
                        || std::path::Path::new(&format!("{path}/liblapack.a")).exists()
                    {
                        separate_lapack_available = true;
                    }
                }

                // Also check common system paths
                for common_path in &["/usr/lib", "/usr/local/lib", "/opt/local/lib"] {
                    if !openblas_available
                        && (std::path::Path::new(&format!("{common_path}/libopenblas.so")).exists()
                            || std::path::Path::new(&format!("{common_path}/libopenblas.a"))
                                .exists())
                    {
                        openblas_available = true;
                    }
                    if !separate_blas_available
                        && (std::path::Path::new(&format!("{common_path}/libblas.so")).exists()
                            || std::path::Path::new(&format!("{common_path}/libblas.a")).exists())
                    {
                        separate_blas_available = true;
                    }
                    if !separate_lapack_available
                        && (std::path::Path::new(&format!("{common_path}/liblapack.so")).exists()
                            || std::path::Path::new(&format!("{common_path}/liblapack.a")).exists())
                    {
                        separate_lapack_available = true;
                    }
                }

                // Ubuntu LAPACK/BLAS linking - ensure both libraries are available
                // Ubuntu's OpenBLAS may not include complete LAPACK, so link both explicitly
                if openblas_available {
                    // Link both OpenBLAS and LAPACK to ensure all functions are available
                    println!("cargo:rustc-link-lib=openblas");
                    if separate_lapack_available {
                        println!("cargo:rustc-link-lib=lapack");
                    }
                } else if separate_blas_available && separate_lapack_available {
                    println!("cargo:rustc-link-lib=lapack");
                    println!("cargo:rustc-link-lib=blas");
                } else {
                    // Fallback: try standard library names
                    println!("cargo:rustc-link-lib=lapack");
                    println!("cargo:rustc-link-lib=blas");
                }
            } else if cfg!(target_os = "macos") {
                // macOS with intelligent BLAS/LAPACK detection
                let blas_lib = env::var("BLAS_LIB")
                    .or_else(|_| env::var("RUSTORCH_BLAS_LIB"))
                    .unwrap_or_else(|_| "framework".to_string());
                let _lapack_lib = env::var("LAPACK_LIB")
                    .or_else(|_| env::var("RUSTORCH_LAPACK_LIB"))
                    .unwrap_or_else(|_| "framework".to_string());

                // Check for Homebrew OpenBLAS (both ARM64 and x86_64)
                let mut openblas_found = false;
                if std::path::Path::new("/opt/homebrew/lib/libopenblas.dylib").exists() {
                    println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
                    openblas_found = true;
                }
                if std::path::Path::new("/usr/local/lib/libopenblas.dylib").exists() {
                    println!("cargo:rustc-link-search=native=/usr/local/lib");
                    openblas_found = true;
                }

                // macOS linking strategy: avoid duplicate libraries
                if blas_lib == "openblas" && openblas_found {
                    // Link OpenBLAS only if explicitly requested AND found
                    println!("cargo:rustc-link-lib=openblas");
                } else {
                    // Default: Use Accelerate framework (system BLAS/LAPACK on macOS)
                    println!("cargo:rustc-link-lib=framework=Accelerate");
                }
            } else {
                // Other Unix systems (FreeBSD, OpenBSD, etc.)
                // Try to detect available libraries, fallback to standard names
                let common_paths = ["/usr/lib", "/usr/local/lib", "/opt/local/lib"];
                let mut openblas_found = false;
                let mut separate_libs_found = false;

                for path in &common_paths {
                    if !openblas_found
                        && (std::path::Path::new(&format!("{path}/libopenblas.so")).exists()
                            || std::path::Path::new(&format!("{path}/libopenblas.a")).exists())
                    {
                        openblas_found = true;
                    }
                    if !separate_libs_found
                        && std::path::Path::new(&format!("{path}/liblapack.so")).exists()
                        && std::path::Path::new(&format!("{path}/libblas.so")).exists()
                    {
                        separate_libs_found = true;
                    }
                }

                if openblas_found {
                    println!("cargo:rustc-link-lib=openblas");
                } else if separate_libs_found {
                    println!("cargo:rustc-link-lib=lapack");
                    println!("cargo:rustc-link-lib=blas");
                } else {
                    // Final fallback - assume system has standard libraries
                    println!("cargo:rustc-link-lib=lapack");
                    println!("cargo:rustc-link-lib=blas");
                }
            }
        }

        // Additional custom library paths (Unix systems only)
        if cfg!(unix) {
            if let Ok(lib_dir) = env::var("RUSTORCH_LIB_DIR") {
                println!("cargo:rustc-link-search=native={lib_dir}");
            }
        }
    }

    // GPU backend build configuration
    #[cfg(feature = "cuda")]
    {
        // Try to detect CUDA installation paths
        let cuda_paths = [
            "/usr/local/cuda/lib64",
            "/opt/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib",
        ];

        let cuda_root = env::var("CUDA_ROOT")
            .or_else(|_| env::var("CUDA_PATH"))
            .or_else(|_| env::var("CUDA_HOME"));

        if let Ok(cuda_root) = cuda_root {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
            println!("cargo:rustc-link-search=native={}/lib", cuda_root);
        } else {
            // Try common CUDA installation paths
            for path in &cuda_paths {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                    break;
                }
            }
        }

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=curand");
        println!("cargo:rustc-link-lib=cusparse");
    }

    #[cfg(feature = "opencl")]
    {
        println!("cargo:rustc-link-lib=OpenCL");
    }

    // Metal framework is automatically linked on macOS
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }
}
