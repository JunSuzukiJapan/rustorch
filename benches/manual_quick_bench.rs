use rustorch::tensor::Tensor;
use std::time::Instant;

fn main() {
    println!("🚀 RusTorch Matrix Decomposition - Quick Manual Benchmark");
    println!("========================================================");
    println!("⚡ Optimized for speed - no criterion overhead");
    
    // 超軽量ベンチマーク設定
    let iterations = 100; // 少ない反復回数
    let max_size = 12;    // 小さな最大サイズ
    
    benchmark_scaling_performance(iterations, max_size);
    benchmark_decomposition_comparison(iterations);
    benchmark_rectangular_matrices(iterations);
    benchmark_special_cases(iterations);
    
    println!("\n✅ Quick Benchmark Complete!");
    println!("💡 For detailed benchmarks, use: cargo bench --bench optimized_matrix_benchmark");
}

fn benchmark_scaling_performance(iterations: usize, max_size: usize) {
    println!("\n📊 1. Scaling Performance Analysis");
    println!("   Size │  SVD   │  QR    │  LU    │ Symeig │  Eig");
    println!("   ─────┼────────┼────────┼────────┼────────┼────────");
    
    for size in (4..=max_size).step_by(2) {
        // テスト行列作成
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 1.0) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data.clone(), vec![size, size]);
        
        // 対称行列作成
        let mut sym_data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    sym_data[i * size + j] = (i + 1) as f32;
                } else if i < j {
                    sym_data[i * size + j] = 0.3;
                    sym_data[j * size + i] = 0.3;
                }
            }
        }
        let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);
        
        // SVD benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.svd(false); // no eigenvectors for speed
        }
        let svd_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // QR benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.qr();
        }
        let qr_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // LU benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.lu();
        }
        let lu_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // Symeig benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sym_matrix.symeig(false, false); // no eigenvectors
        }
        let symeig_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // Eig benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.eig(false); // no eigenvectors
        }
        let eig_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        println!("   {:2}x{:<2} │ {:5.1}μs │ {:5.1}μs │ {:5.1}μs │ {:5.1}μs │ {:5.1}μs", 
                size, size, svd_time, qr_time, lu_time, symeig_time, eig_time);
    }
}

fn benchmark_decomposition_comparison(iterations: usize) {
    println!("\n📊 2. Decomposition Method Comparison (8x8 matrix)");
    
    let size = 8;
    let matrix_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 * 1.4 + 2.1) % 6.0 + 1.0)
        .collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);
    
    // 対称行列
    let mut sym_data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if i == j {
                sym_data[i * size + j] = (i + 1) as f32 * 1.5;
            } else if i < j {
                sym_data[i * size + j] = 0.4;
                sym_data[j * size + i] = 0.4;
            }
        }
    }
    let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);
    
    let methods: Vec<(&str, &Tensor, bool)> = vec![
        ("SVD", &matrix, false),
        ("QR", &matrix, false), 
        ("LU", &matrix, false),
        ("Symeig", &sym_matrix, true),
        ("Eig", &matrix, false),
    ];
    
    println!("   Method │   Time   │  Rel. Speed");
    println!("   ───────┼──────────┼────────────");
    
    let mut times = Vec::new();
    
    for (name, test_matrix, is_symeig) in &methods {
        let start = Instant::now();
        for _ in 0..iterations {
            match *name {
                "SVD" => { let _ = test_matrix.svd(false); },
                "QR" => { let _ = test_matrix.qr(); },
                "LU" => { let _ = test_matrix.lu(); },
                "Symeig" => { let _ = test_matrix.symeig(false, false); },
                "Eig" => { let _ = test_matrix.eig(false); },
                _ => {}
            }
        }
        let time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        times.push((name, time));
    }
    
    // 最速を基準に相対速度計算
    let fastest = times.iter().map(|(_, t)| *t).fold(f64::INFINITY, f64::min);
    
    for (name, time) in &times {
        let relative = time / fastest;
        println!("   {:6} │ {:6.1}μs │ {:6.1}x", name, time, relative);
    }
}

fn benchmark_rectangular_matrices(iterations: usize) {
    println!("\n📊 3. Rectangular Matrix Performance");
    
    let test_cases = vec![
        (6, 3, "6x3"),
        (8, 4, "8x4"),
        (10, 5, "10x5"),
        (12, 6, "12x6"),
    ];
    
    println!("   Size  │  SVD   │   QR");
    println!("   ──────┼────────┼────────");
    
    for (rows, cols, label) in test_cases {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 + 1.0) % 7.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);
        
        // SVD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.svd(false);
        }
        let svd_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // QR
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.qr();
        }
        let qr_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        println!("   {:5} │ {:5.1}μs │ {:5.1}μs", label, svd_time, qr_time);
    }
}

fn benchmark_special_cases(iterations: usize) {
    println!("\n📊 4. Special Matrix Cases (8x8)");
    
    let size = 8;
    
    // 単位行列
    let mut identity_data = vec![0.0f32; size * size];
    for i in 0..size {
        identity_data[i * size + i] = 1.0;
    }
    let identity = Tensor::from_vec(identity_data, vec![size, size]);
    
    // 対角行列
    let mut diagonal_data = vec![0.0f32; size * size];
    for i in 0..size {
        diagonal_data[i * size + i] = (i + 1) as f32;
    }
    let diagonal = Tensor::from_vec(diagonal_data, vec![size, size]);
    
    // ランダム様行列
    let random_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 * 17.0 + 31.0) % 13.0 + 1.0)
        .collect();
    let random = Tensor::from_vec(random_data, vec![size, size]);
    
    let test_cases = vec![
        ("Identity", &identity),
        ("Diagonal", &diagonal), 
        ("Random-like", &random),
    ];
    
    println!("   Case      │  SVD   │   QR   │   LU");
    println!("   ──────────┼────────┼────────┼────────");
    
    for (name, matrix) in test_cases {
        // SVD
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.svd(false);
        }
        let svd_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // QR
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.qr();
        }
        let qr_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        // LU
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix.lu();
        }
        let lu_time = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
        
        println!("   {:9} │ {:5.1}μs │ {:5.1}μs │ {:5.1}μs", name, svd_time, qr_time, lu_time);
    }
}