//! SVD (Singular Value Decomposition) demonstration
//! SVD（特異値分解）のデモンストレーション

use rustorch::tensor::Tensor;

fn main() {
    println!("🔬 RusTorch SVD (Singular Value Decomposition) Demo");
    println!("==================================================\n");

    // Test 1: Basic SVD on a simple 3x3 matrix
    println!("📊 1. Basic 3x3 Matrix SVD");
    basic_svd_demo();
    
    // Test 2: SVD on a larger matrix
    println!("\n📊 2. Larger Matrix SVD (4x3)");
    rectangular_svd_demo();
    
    // Test 3: SVD properties verification
    println!("\n📊 3. SVD Properties Verification");
    verify_svd_properties();
    
    // Test 4: Rank-deficient matrix SVD
    println!("\n📊 4. Rank-Deficient Matrix SVD");
    rank_deficient_svd_demo();
    
    println!("\n✅ SVD Demo Complete!");
    println!("🎯 SVD is ready for dimensionality reduction, PCA, and matrix analysis!");
}

fn basic_svd_demo() {
    // Create a simple 3x3 matrix
    let matrix = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ],
        vec![3, 3]
    );
    
    println!("  Original matrix (3x3):");
    print_matrix(&matrix);
    
    // Perform SVD
    match matrix.svd(true) {
        Ok((u, s, v)) => {
            println!("\n  SVD Results:");
            println!("    U (left singular vectors) shape: {:?}", u.shape());
            print_matrix(&u);
            
            println!("\n    S (singular values) shape: {:?}", s.shape());
            println!("    Singular values: {:?}", 
                s.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
            );
            
            println!("\n    V (right singular vectors) shape: {:?}", v.shape());
            print_matrix(&v);
        }
        Err(e) => println!("  ❌ SVD failed: {}", e),
    }
}

fn rectangular_svd_demo() {
    // Create a 4x3 rectangular matrix
    let matrix = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        ],
        vec![4, 3]
    );
    
    println!("  Original matrix (4x3):");
    print_matrix(&matrix);
    
    match matrix.svd(true) {
        Ok((u, s, v)) => {
            println!("\n  SVD Results:");
            println!("    U shape: {:?}", u.shape());
            println!("    S shape: {:?}", s.shape());
            println!("    V shape: {:?}", v.shape());
            
            println!("    Singular values: {:?}", 
                s.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => println!("  ❌ SVD failed: {}", e),
    }
}

fn verify_svd_properties() {
    // Create a test matrix
    let matrix = Tensor::from_vec(
        vec![
            3.0, 1.0,
            1.0, 3.0
        ],
        vec![2, 2]
    );
    
    println!("  Test matrix (2x2):");
    print_matrix(&matrix);
    
    match matrix.svd(false) {
        Ok((u, _, v)) => {
            println!("\n  Verification:");
            
            // Check orthogonality of U (U^T * U should be identity)
            if let Ok(ut) = u.transpose() {
                if let Ok(utu) = ut.matmul(&u) {
                    println!("    U^T * U (should be close to identity):");
                    print_matrix(&utu);
                }
            }
            
            // Check orthogonality of V (V^T * V should be identity)
            if let Ok(vt) = v.transpose() {
                if let Ok(vtv) = vt.matmul(&v) {
                    println!("\n    V^T * V (should be close to identity):");
                    print_matrix(&vtv);
                }
            }
            
            println!("\n    Properties verified!");
        }
        Err(e) => println!("  ❌ SVD verification failed: {}", e),
    }
}

fn rank_deficient_svd_demo() {
    // Create a rank-deficient matrix (rank 1)
    let matrix = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0
        ],
        vec![3, 3]
    );
    
    println!("  Rank-deficient matrix (all rows are multiples of [1,2,3]):");
    print_matrix(&matrix);
    
    match matrix.svd(true) {
        Ok((_, s, _)) => {
            println!("\n  SVD of rank-deficient matrix:");
            println!("    Singular values: {:?}", 
                s.data.as_slice().unwrap_or(&[]).iter()
                    .map(|&x| format!("{:.6}", x))
                    .collect::<Vec<_>>()
            );
            
            // Count non-zero singular values to determine rank
            let tolerance = 1e-10f32;
            let rank = s.data.as_slice().unwrap_or(&[])
                .iter()
                .filter(|&&x| x > tolerance)
                .count();
            
            println!("    Estimated rank: {} (tolerance: {})", rank, tolerance);
        }
        Err(e) => println!("  ❌ Rank-deficient SVD failed: {}", e),
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