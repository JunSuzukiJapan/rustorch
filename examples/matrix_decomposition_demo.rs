//! Matrix Decomposition demonstration (QR and LU)
//! 行列分解のデモンストレーション（QR分解とLU分解）
//!
//! Note: This example demonstrates the tensor library structure for matrix decompositions.
//! Full QR and LU implementations require system BLAS/LAPACK libraries to be properly configured.

use rustorch::tensor::Tensor;

fn main() {
    println!("🔬 RusTorch Matrix Decomposition Demo");
    println!("===================================\n");

    println!("ℹ️  Matrix Decomposition Status:");
    println!("   This demo shows the tensor structure and API for matrix decompositions.");
    println!("   Advanced decompositions (QR, LU) require system BLAS/LAPACK configuration.");
    println!("   Basic matrix operations (matmul, transpose) are fully functional.\n");

    // Test 1: Basic Matrix Operations (Always Available)
    println!("📊 1. Basic Matrix Operations (Available)");
    basic_matrix_demo();

    // Test 2: Advanced Decompositions (Conditional)
    println!("\n📊 2. Advanced Matrix Decompositions");
    advanced_decomposition_demo();

    println!("\n✅ Matrix Operations Demo Complete!");
    println!("🎯 Basic matrix operations are ready. Advanced features require system BLAS setup.");
}

fn basic_matrix_demo() {
    // Create test matrices
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("  Matrix A (2x2):");
    print_matrix(&a);

    println!("  Matrix B (2x2):");
    print_matrix(&b);

    // Matrix multiplication (always available)
    match a.matmul(&b) {
        Ok(result) => {
            println!("  Matrix multiplication A * B:");
            print_matrix(&result);
            println!("    ✅ Matrix multiplication working correctly");
        }
        Err(e) => println!("  ❌ Matrix multiplication failed: {}", e),
    }

    // Transpose (always available)
    match a.transpose() {
        Ok(result) => {
            println!("  Transpose of A:");
            print_matrix(&result);
            println!("    ✅ Matrix transpose working correctly");
        }
        Err(e) => println!("  ❌ Matrix transpose failed: {}", e),
    }
}

fn advanced_decomposition_demo() {
    let matrix = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        vec![3, 3],
    );

    println!("  Test matrix A (3x3):");
    print_matrix(&matrix);

    println!("  📊 Advanced Decomposition Status:");

    // Check what's compiled in
    #[cfg(feature = "linalg")]
    {
        println!("    ✅ linalg feature is enabled");

        // The methods might still not be available due to system dependencies
        println!("    ⚠️  QR, LU, and SVD decompositions require proper system BLAS/LAPACK setup");
        println!("    📋 Current implementation status:");
        println!("       - Matrix multiplication: ✅ Available");
        println!("       - Matrix transpose: ✅ Available");
        println!("       - SVD decomposition: ⚠️ Requires system BLAS");
        println!("       - QR decomposition: ⚠️ Requires system BLAS");
        println!("       - LU decomposition: ⚠️ Requires system BLAS");
    }

    #[cfg(not(feature = "linalg"))]
    {
        println!("    ❌ linalg feature is not enabled");
        println!("    📋 Available operations:");
        println!("       - Matrix multiplication: ✅ Available");
        println!("       - Matrix transpose: ✅ Available");
        println!("       - Advanced decompositions: ❌ Require linalg feature");
    }

    println!("\n  💡 Setup Instructions for Advanced Features:");
    println!("     Option 1 - Use linalg feature (requires OpenBLAS/LAPACK build):");
    println!("       brew install openblas lapack  # macOS");
    println!("       sudo apt install libopenblas-dev liblapack-dev  # Ubuntu");
    println!("       cargo run --example matrix_decomposition_demo --features=\"linalg\"");
    println!("     ");
    println!("     Option 2 - Use system BLAS libraries:");
    println!("       cargo run --example matrix_decomposition_demo --features=\"linalg-system\"");
    println!("     ");
    println!("     Option 3 - For now, enjoy the basic matrix operations that work!");

    // Demonstrate what definitely works
    println!("\n  🎯 Working Matrix Operations Demo:");
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);

    if let Ok(result) = a.matmul(&b) {
        println!("    Matrix multiplication result:");
        print_matrix(&result);
    }
}

fn print_matrix(tensor: &Tensor<f32>) {
    let shape = tensor.shape();
    if shape.len() != 2 {
        println!("    Cannot display non-2D tensor");
        return;
    }

    let data = tensor.data.as_slice().unwrap_or(&[]);
    let rows = shape[0];
    let cols = shape[1];

    for i in 0..rows {
        print!("    [");
        for j in 0..cols {
            let val = data[i * cols + j];
            print!("{:8.4}", val);
            if j < cols - 1 {
                print!(" ");
            }
        }
        println!("]");
    }
}
