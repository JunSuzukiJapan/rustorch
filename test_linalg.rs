// Quick test for OpenBLAS integration
extern crate rustorch;
use rustorch::tensor::Tensor;

fn main() {
    println!("🧪 Testing OpenBLAS integration with RusTorch");
    
    // Create a test matrix
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let matrix = Tensor::from_vec(data, vec![3, 3]);
    
    println!("\n📊 Original Matrix:");
    println!("{:?}", matrix);
    
    // Test SVD
    println!("\n🔬 Testing SVD (with OpenBLAS)...");
    match matrix.svd(false) {
        Ok((u, s, v)) => {
            println!("✅ SVD successful!");
            println!("  U shape: {:?}", u.shape());
            println!("  S shape: {:?}", s.shape());
            println!("  V shape: {:?}", v.shape());
        }
        Err(e) => println!("❌ SVD failed: {}", e),
    }
    
    // Test QR
    println!("\n🔬 Testing QR decomposition...");
    match matrix.qr() {
        Ok((q, r)) => {
            println!("✅ QR successful!");
            println!("  Q shape: {:?}", q.shape());
            println!("  R shape: {:?}", r.shape());
        }
        Err(e) => println!("❌ QR failed: {}", e),
    }
    
    // Test LU
    println!("\n🔬 Testing LU decomposition...");
    match matrix.lu() {
        Ok((l, u, p)) => {
            println!("✅ LU successful!");
            println!("  L shape: {:?}", l.shape());
            println!("  U shape: {:?}", u.shape());
            println!("  P shape: {:?}", p.shape());
        }
        Err(e) => println!("❌ LU failed: {}", e),
    }
    
    println!("\n🎉 OpenBLAS integration test complete!");
}