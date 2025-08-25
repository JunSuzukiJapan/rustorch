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
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        
        // Link LAPACK and BLAS libraries
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=blas");
        println!("cargo:rustc-link-lib=gfortran");
        
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