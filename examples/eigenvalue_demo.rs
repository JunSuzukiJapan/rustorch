//! Eigenvalue Decomposition demonstration
//! 固有値分解のデモンストレーション

use rustorch::tensor::Tensor;

fn main() {
    println!("🔬 RusTorch Eigenvalue Decomposition Demo");
    println!("========================================\n");

    // Test 1: General eigenvalue decomposition (torch.eig)
    println!("📊 1. General Eigenvalue Decomposition (torch.eig)");
    general_eig_demo();
    
    // Test 2: Symmetric eigenvalue decomposition (torch.symeig)
    println!("\n📊 2. Symmetric Eigenvalue Decomposition (torch.symeig)");
    symmetric_eig_demo();
    
    // Test 3: Real-world application: PCA preparation
    println!("\n📊 3. PCA Preparation using Eigenvalue Decomposition");
    pca_demo();
    
    // Test 4: Eigenvalue properties verification
    println!("\n📊 4. Eigenvalue Properties Verification");
    verify_eigenvalue_properties();
    
    println!("\n✅ Eigenvalue Decomposition Demo Complete!");
    println!("🎯 Ready for PCA, dimensionality reduction, and matrix analysis!");
}

fn general_eig_demo() {
    // Create a general 3x3 matrix
    let matrix = Tensor::from_vec(
        vec![
            4.0f32, 1.0, 2.0,
            1.0, 3.0, 1.0,
            2.0, 1.0, 5.0
        ],
        vec![3, 3]
    );
    
    println!("  General matrix (3x3):");
    print_matrix(&matrix);
    
    // Compute eigenvalues only
    match matrix.eig(false) {
        Ok((eigenvals, _)) => {
            println!("\n  Eigenvalues (real + i*imag format):");
            print_eigenvalues(&eigenvals);
        }
        Err(e) => println!("  ❌ Eigenvalue computation failed: {}", e),
    }
    
    // Compute eigenvalues and eigenvectors
    match matrix.eig(true) {
        Ok((eigenvals, Some(eigenvecs))) => {
            println!("\n  With eigenvectors:");
            println!("    Eigenvalues shape: {:?}", eigenvals.shape());
            println!("    Eigenvectors shape: {:?}", eigenvecs.shape());
            
            println!("\n    Eigenvectors matrix:");
            print_matrix(&eigenvecs);
        }
        Ok((_, None)) => println!("  No eigenvectors returned"),
        Err(e) => println!("  ❌ Eigenvalue computation with vectors failed: {}", e),
    }
}

fn symmetric_eig_demo() {
    // Create a symmetric matrix
    let matrix = Tensor::from_vec(
        vec![
            5.0f32, 2.0, 1.0,
            2.0, 3.0, 1.0,
            1.0, 1.0, 4.0
        ],
        vec![3, 3]
    );
    
    println!("  Symmetric matrix (3x3):");
    print_matrix(&matrix);
    
    // Compute symmetric eigenvalues (ascending order, using lower triangle)
    match matrix.symeig(true, false) {
        Ok((eigenvals, Some(eigenvecs))) => {
            println!("\n  Symmetric eigenvalue decomposition:");
            println!("    Eigenvalues (real, sorted ascending): {:?}", 
                eigenvals.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
            );
            
            println!("\n    Orthonormal eigenvectors:");
            print_matrix(&eigenvecs);
            
            // Verify orthogonality
            if let Ok(vt) = eigenvecs.transpose() {
                if let Ok(vtv) = vt.matmul(&eigenvecs) {
                    println!("\n    Orthogonality check (V^T * V should be identity):");
                    print_matrix(&vtv);
                }
            }
        }
        Ok((eigenvals, None)) => {
            println!("\n  Eigenvalues only: {:?}", 
                eigenvals.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => println!("  ❌ Symmetric eigenvalue computation failed: {}", e),
    }
}

fn pca_demo() {
    // Create a data covariance matrix (typical PCA scenario)
    let covariance = Tensor::from_vec(
        vec![
            2.5f32, 1.2,
            1.2, 1.8
        ],
        vec![2, 2]
    );
    
    println!("  Covariance matrix for PCA:");
    print_matrix(&covariance);
    
    // Compute eigenvalues and eigenvectors for PCA
    match covariance.symeig(true, false) {
        Ok((eigenvals, Some(eigenvecs))) => {
            println!("\n  PCA Components:");
            let eigenvals_data = eigenvals.data.as_slice().unwrap_or(&[]);
            
            println!("    Principal component variances:");
            for (i, &val) in eigenvals_data.iter().rev().enumerate() {
                let explained_var = val / eigenvals_data.iter().sum::<f32>() * 100.0;
                println!("      PC{}: {:.4} ({:.1}% variance explained)", 
                    i + 1, val, explained_var);
            }
            
            println!("\n    Principal component directions (eigenvectors):");
            print_matrix(&eigenvecs);
        }
        Ok((_, None)) => println!("  No eigenvectors for PCA"),
        Err(e) => println!("  ❌ PCA eigenvalue computation failed: {}", e),
    }
}

fn verify_eigenvalue_properties() {
    // Test with a known matrix
    let matrix = Tensor::from_vec(
        vec![
            3.0f32, 0.0,
            0.0, 2.0
        ],
        vec![2, 2]
    );
    
    println!("  Test matrix (diagonal, known eigenvalues 3, 2):");
    print_matrix(&matrix);
    
    match matrix.symeig(false, false) {
        Ok((eigenvals, _)) => {
            println!("\n  Computed eigenvalues: {:?}", 
                eigenvals.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
            );
            
            // Verify trace (sum of eigenvalues should equal trace of matrix)
            let eigenval_sum: f32 = eigenvals.data.as_slice().unwrap_or(&[]).iter().sum();
            let matrix_trace = matrix.data.as_slice().unwrap_or(&[])[0] + 
                              matrix.data.as_slice().unwrap_or(&[])[3];
            
            println!("    Trace verification:");
            println!("      Sum of eigenvalues: {:.4}", eigenval_sum);
            println!("      Matrix trace: {:.4}", matrix_trace);
            println!("      Match: {}", (eigenval_sum - matrix_trace).abs() < 1e-4);
        }
        Err(e) => println!("  ❌ Eigenvalue verification failed: {}", e),
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
            if j < cols - 1 { print!(" "); }
        }
        println!("]");
    }
}

fn print_eigenvalues(eigenvals: &Tensor<f32>) {
    let shape = eigenvals.shape();
    if shape.len() != 2 || shape[1] != 2 {
        println!("    Invalid eigenvalue format");
        return;
    }
    
    let data = eigenvals.data.as_slice().unwrap_or(&[]);
    let n = shape[0];
    
    for i in 0..n {
        let real_part = data[i * 2];
        let imag_part = data[i * 2 + 1];
        
        if imag_part.abs() < 1e-6 {
            println!("    λ{}: {:.4}", i + 1, real_part);
        } else {
            println!("    λ{}: {:.4} + {:.4}i", i + 1, real_part, imag_part);
        }
    }
}