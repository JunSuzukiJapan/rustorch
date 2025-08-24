//! Matrix Decomposition demonstration (QR and LU)
//! 行列分解のデモンストレーション（QR分解とLU分解）

use rustorch::tensor::Tensor;

fn main() {
    println!("🔬 RusTorch Matrix Decomposition Demo");
    println!("===================================\n");

    // Test 1: QR Decomposition
    println!("📊 1. QR Decomposition (A = Q * R)");
    qr_demo();
    
    // Test 2: LU Decomposition 
    println!("\n📊 2. LU Decomposition (PA = LU)");
    lu_demo();
    
    // Test 3: Applications - Linear System Solving
    println!("\n📊 3. Applications: Linear System Solving");
    linear_system_demo();
    
    // Test 4: Matrix Properties Verification
    println!("\n📊 4. Matrix Properties Verification");
    verify_decomposition_properties();
    
    println!("\n✅ Matrix Decomposition Demo Complete!");
    println!("🎯 Ready for linear system solving, least squares, and matrix analysis!");
}

fn qr_demo() {
    // Create a test matrix
    let matrix = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0  // Make it full rank
        ],
        vec![3, 3]
    );
    
    println!("  Original matrix A (3x3):");
    print_matrix(&matrix);
    
    match matrix.qr() {
        Ok((q, r)) => {
            println!("\n  QR Decomposition Results:");
            println!("    Q (orthogonal matrix) shape: {:?}", q.shape());
            print_matrix(&q);
            
            println!("\n    R (upper triangular matrix) shape: {:?}", r.shape());
            print_matrix(&r);
            
            // Verify orthogonality of Q: Q^T * Q should be identity
            if let (Ok(_), Ok(qtq)) = (q.transpose(), q.transpose().and_then(|qt| qt.matmul(&q))) {
                println!("\n    Orthogonality check (Q^T * Q):");
                print_matrix(&qtq);
            }
            
            // Verify reconstruction: Q * R should equal A
            if let Ok(qr_product) = q.matmul(&r) {
                println!("\n    Reconstruction check (Q * R):");
                print_matrix(&qr_product);
                
                // Check reconstruction error
                let error = calculate_matrix_error(&matrix, &qr_product);
                println!("    Reconstruction error (max absolute difference): {:.6}", error);
            }
        }
        Err(e) => println!("  ❌ QR decomposition failed: {}", e),
    }
}

fn lu_demo() {
    // Create a test matrix (ensure it's invertible)
    let matrix = Tensor::from_vec(
        vec![
            2.0f32, 1.0, 3.0,
            1.0, 3.0, 2.0,
            3.0, 2.0, 4.0
        ],
        vec![3, 3]
    );
    
    println!("  Original matrix A (3x3):");
    print_matrix(&matrix);
    
    match matrix.lu() {
        Ok((l, u, p)) => {
            println!("\n  LU Decomposition Results:");
            
            println!("    L (lower triangular with unit diagonal) shape: {:?}", l.shape());
            print_matrix(&l);
            
            println!("\n    U (upper triangular) shape: {:?}", u.shape());
            print_matrix(&u);
            
            println!("\n    P (permutation matrix) shape: {:?}", p.shape());
            print_matrix(&p);
            
            // Verify reconstruction: P * A should equal L * U
            if let (Ok(pa), Ok(lu_product)) = (p.matmul(&matrix), l.matmul(&u)) {
                println!("\n    Reconstruction check (PA vs LU):");
                println!("      PA:");
                print_matrix(&pa);
                println!("      LU:");
                print_matrix(&lu_product);
                
                let error = calculate_matrix_error(&pa, &lu_product);
                println!("    Reconstruction error (max absolute difference): {:.6}", error);
            }
        }
        Err(e) => println!("  ❌ LU decomposition failed: {}", e),
    }
}

fn linear_system_demo() {
    // Demonstrate solving Ax = b using QR and LU decompositions
    let a = Tensor::from_vec(
        vec![
            3.0f32, 1.0,
            1.0, 2.0
        ],
        vec![2, 2]
    );
    
    let b = Tensor::from_vec(vec![9.0f32, 8.0], vec![2]);
    
    println!("  Linear system Ax = b:");
    println!("    Matrix A:");
    print_matrix(&a);
    println!("    Vector b: [{:.4}, {:.4}]", b.data.as_slice().unwrap()[0], b.data.as_slice().unwrap()[1]);
    
    // Expected solution: x = [2, 3] (since 3*2 + 1*3 = 9, 1*2 + 2*3 = 8)
    println!("    Expected solution: x = [2.0000, 3.0000]");
    
    // Using QR decomposition
    if let Ok((_q, _r)) = a.qr() {
        println!("\n    Using QR decomposition:");
        println!("      This would require solving Rx = Q^T * b for triangular system");
        println!("      (Implementation of triangular solve would go here)");
    }
    
    // Using LU decomposition  
    if let Ok((_l, _u, _p)) = a.lu() {
        println!("\n    Using LU decomposition:");
        println!("      This would require solving Ly = Pb, then Ux = y");
        println!("      (Implementation of forward/backward substitution would go here)");
    }
}

fn verify_decomposition_properties() {
    // Test with identity matrix
    let identity = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 1.0],
        vec![2, 2]
    );
    
    println!("  Testing with identity matrix:");
    print_matrix(&identity);
    
    // QR of identity should be Q=I, R=I
    if let Ok((q, r)) = identity.qr() {
        println!("\n    QR of identity:");
        println!("      Q (should be identity):");
        print_matrix(&q);
        println!("      R (should be identity):");
        print_matrix(&r);
    }
    
    // LU of identity should be L=I, U=I, P=I
    if let Ok((l, u, p)) = identity.lu() {
        println!("\n    LU of identity:");
        println!("      L (should be identity):");
        print_matrix(&l);
        println!("      U (should be identity):");
        print_matrix(&u);
        println!("      P (should be identity):");
        print_matrix(&p);
    }
    
    // Test properties with rectangular matrix
    let rect_matrix = Tensor::from_vec(
        vec![1.0f32, 2.0, 4.0, 5.0, 7.0, 8.0],
        vec![3, 2]
    );
    
    println!("\n  Testing with rectangular matrix (3x2):");
    print_matrix(&rect_matrix);
    
    if let Ok((q, r)) = rect_matrix.qr() {
        println!("\n    QR shapes for 3x2 matrix:");
        println!("      Q shape: {:?} (should be 3x2)", q.shape());
        println!("      R shape: {:?} (should be 2x2)", r.shape());
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

fn calculate_matrix_error(a: &Tensor<f32>, b: &Tensor<f32>) -> f32 {
    let a_data = a.data.as_slice().unwrap_or(&[]);
    let b_data = b.data.as_slice().unwrap_or(&[]);
    
    if a_data.len() != b_data.len() {
        return f32::INFINITY;
    }
    
    let mut max_error = 0.0f32;
    for (a_val, b_val) in a_data.iter().zip(b_data.iter()) {
        let error = (a_val - b_val).abs();
        if error > max_error {
            max_error = error;
        }
    }
    
    max_error
}