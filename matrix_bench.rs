// Comprehensive matrix decomposition benchmark
extern crate rustorch;
use rustorch::tensor::Tensor;
use std::time::Instant;

fn benchmark_operation<F>(name: &str, op: F, iterations: usize) -> f64
where
    F: Fn() -> Result<(), String>,
{
    println!("🔬 Benchmarking {} ({} iterations)...", name, iterations);
    
    // Warmup
    for _ in 0..5 {
        let _ = op();
    }
    
    let start = Instant::now();
    let mut success_count = 0;
    
    for _ in 0..iterations {
        if op().is_ok() {
            success_count += 1;
        }
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_nanos() as f64 / iterations as f64 / 1000.0; // μs
    
    println!("  ✅ {}/{} successful", success_count, iterations);
    println!("  📊 Average time: {:.2} μs", avg_time);
    
    avg_time
}

fn main() {
    println!("🚀 RusTorch Matrix Decomposition Benchmark");
    println!("=========================================");
    
    let sizes = [4, 8, 16, 32];
    let iterations = 100;
    
    for size in sizes {
        println!("\n📈 Testing {}x{} matrices:", size, size);
        
        // Create test matrix
        let data: Vec<f32> = (0..size*size)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let matrix = Tensor::from_vec(data.clone(), vec![size, size]);
        
        // Create symmetric matrix for symeig
        let sym_data: Vec<f32> = (0..size*size).map(|i| {
            let row = i / size;
            let col = i % size;
            if row <= col {
                (row + col + 1) as f32 * 0.1
            } else {
                ((col + row + 1) as f32) * 0.1
            }
        }).collect();
        let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);
        
        // Benchmark SVD
        let svd_time = benchmark_operation(
            "SVD", 
            || matrix.svd(false).map(|_| ()), 
            iterations
        );
        
        // Benchmark QR
        let qr_time = benchmark_operation(
            "QR", 
            || matrix.qr().map(|_| ()), 
            iterations
        );
        
        // Benchmark LU
        let lu_time = benchmark_operation(
            "LU", 
            || matrix.lu().map(|_| ()), 
            iterations
        );
        
        // Benchmark symmetric eigenvalue decomposition
        let symeig_time = benchmark_operation(
            "Symeig", 
            || sym_matrix.symeig(true, true).map(|_| ()), 
            iterations
        );
        
        // Benchmark general eigenvalue decomposition
        let eig_time = benchmark_operation(
            "Eig", 
            || matrix.eig(true).map(|_| ()), 
            iterations
        );
        
        // Summary
        println!("\n  📋 Summary for {}x{} matrices:", size, size);
        println!("     SVD:    {:.2} μs", svd_time);
        println!("     QR:     {:.2} μs", qr_time);
        println!("     LU:     {:.2} μs", lu_time);
        println!("     Symeig: {:.2} μs", symeig_time);
        println!("     Eig:    {:.2} μs", eig_time);
    }
    
    println!("\n🎯 Benchmark Complete!");
    println!("💡 All decompositions using pure Rust implementation");
}