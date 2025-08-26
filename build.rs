//! Build script for RusTorch
//! RusTorch用ビルドスクリプト
//!
//! This build script ensures proper linking of LAPACK/BLAS libraries
//! when using the linalg-netlib feature.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Check if we're building with linalg-netlib feature
    if env::var("CARGO_FEATURE_LINALG_NETLIB").is_ok() {
        // Windows requires special handling for LAPACK/BLAS
        if cfg!(target_os = "windows") {
            // On Windows, we rely on netlib-src crate's build system
            // No explicit linking needed as netlib-src handles it
            println!("cargo:rustc-cfg=windows_netlib");
        } else {
            // Unix systems (Linux, macOS) - explicit linking

            // OS-specific library paths for Linux
            if cfg!(target_os = "linux") {
                println!("cargo:rustc-link-search=native=/usr/lib");
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
                println!(
                    "cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread"
                );
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/blas");
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/lapack");
            }

            // Common library path for Unix systems
            if cfg!(unix) {
                println!("cargo:rustc-link-search=native=/usr/local/lib");
            }

            // Link LAPACK and BLAS libraries with specific library names for better compatibility
            if cfg!(target_os = "linux") {
                // Check for explicit BLAS/LAPACK library preferences
                let blas_lib = env::var("BLAS_LIB").or_else(|_| env::var("RUSTORCH_BLAS_LIB")).unwrap_or_else(|_| "openblas".to_string());
                let lapack_lib = env::var("LAPACK_LIB").or_else(|_| env::var("RUSTORCH_LAPACK_LIB")).unwrap_or_else(|_| "openblas".to_string());
                
                // Link explicitly specified libraries
                println!("cargo:rustc-link-lib={}", blas_lib);
                if blas_lib != lapack_lib {
                    println!("cargo:rustc-link-lib={}", lapack_lib);
                }
                
                // Always link gfortran for Fortran runtime
                println!("cargo:rustc-link-lib=gfortran");
                
                // Add explicit dylib linking for better symbol resolution
                if blas_lib == "openblas" || lapack_lib == "openblas" {
                    println!("cargo:rustc-link-lib=dylib=openblas");
                }
            } else if cfg!(target_os = "macos") {
                // macOS with Homebrew OpenBLAS
                let blas_lib = env::var("BLAS_LIB").or_else(|_| env::var("RUSTORCH_BLAS_LIB")).unwrap_or_else(|_| "openblas".to_string());
                let lapack_lib = env::var("LAPACK_LIB").or_else(|_| env::var("RUSTORCH_LAPACK_LIB")).unwrap_or_else(|_| "openblas".to_string());
                
                // Check for Homebrew paths (both ARM64 and x86_64) and add them
                let mut homebrew_found = false;
                if std::path::Path::new("/opt/homebrew/lib/libopenblas.dylib").exists() {
                    println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
                    homebrew_found = true;
                }
                if std::path::Path::new("/usr/local/lib/libopenblas.dylib").exists() {
                    println!("cargo:rustc-link-search=native=/usr/local/lib");
                    homebrew_found = true;
                }
                
                // Add standard macOS library paths as fallback
                println!("cargo:rustc-link-search=native=/usr/lib");
                println!("cargo:rustc-link-search=native=/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A");
                
                // Link specified libraries or fallback to Accelerate framework
                if (blas_lib == "openblas" || lapack_lib == "openblas") && homebrew_found {
                    println!("cargo:rustc-link-lib=openblas");
                } else if blas_lib == "openblas" || lapack_lib == "openblas" {
                    // If OpenBLAS requested but not found, fallback to Accelerate
                    println!("cargo:rustc-link-lib=framework=Accelerate");
                } else {
                    println!("cargo:rustc-link-lib=lapack");
                    println!("cargo:rustc-link-lib=blas");
                }
            } else {
                // Other Unix systems
                println!("cargo:rustc-link-lib=lapack");
                println!("cargo:rustc-link-lib=blas");
            }
        }

        // For systems with different LAPACK/BLAS implementations
        if let Ok(lapack_lib) = env::var("RUSTORCH_LAPACK_LIB") {
            println!("cargo:rustc-link-lib={}", lapack_lib);
        }

        if let Ok(blas_lib) = env::var("RUSTORCH_BLAS_LIB") {
            println!("cargo:rustc-link-lib={}", blas_lib);
        }

        // Additional search paths
        if let Ok(lib_dir) = env::var("RUSTORCH_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }
    }

    // GPU backend build configuration
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
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
