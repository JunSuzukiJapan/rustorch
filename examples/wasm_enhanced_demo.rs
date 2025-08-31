//! Enhanced WASM features demonstration
//! 強化WASM機能デモンストレーション

#[cfg(feature = "wasm")]
use rustorch::wasm::{
    autograd_simplified::*, distributions_enhanced::*, optimizer_enhanced::*, special_enhanced::*,
};

#[cfg(feature = "wasm")]
fn main() {
    println!("🚀 RusTorch Enhanced WASM Features Demo");
    println!("========================================");

    // Special Functions Demo / 特殊関数デモ
    println!("\n📊 Special Functions Performance:");

    let test_values = vec![0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0];

    println!("Gamma functions:");
    for &x in &test_values {
        let gamma_val = gamma_wasm(x);
        let lgamma_val = lgamma_wasm(x);
        let digamma_val = digamma_wasm(x);
        println!(
            "  x={:.1}: Γ(x)={:.6}, ln(Γ(x))={:.6}, ψ(x)={:.6}",
            x, gamma_val, lgamma_val, digamma_val
        );
    }

    println!("\nBessel functions (J_0, Y_0, I_0, K_0):");
    for &x in &[0.5, 1.0, 2.0, 5.0] {
        let j0 = bessel_j_wasm(0.0, x);
        let y0 = bessel_y_wasm(0.0, x);
        let i0 = bessel_i_wasm(0.0, x);
        let k0 = bessel_k_wasm(0.0, x);
        println!(
            "  x={:.1}: J_0={:.6}, Y_0={:.6}, I_0={:.6}, K_0={:.6}",
            x, j0, y0, i0, k0
        );
    }

    println!("\nError functions:");
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let erf_val = erf_wasm(x);
        let erfc_val = erfc_wasm(x);
        println!(
            "  x={:.1}: erf(x)={:.6}, erfc(x)={:.6}",
            x, erf_val, erfc_val
        );
    }

    // Enhanced Distributions Demo / 強化分布デモ
    println!("\n🎲 Enhanced Statistical Distributions:");

    // Normal distribution
    let normal = NormalDistributionWasm::new(0.0, 1.0);
    let normal_samples = normal.sample_array(5);
    println!("Normal(0,1) samples: {:?}", normal_samples);
    println!(
        "Normal(0,1) mean: {:.6}, variance: {:.6}",
        normal.mean(),
        normal.variance()
    );

    // Gamma distribution
    let gamma_dist = GammaDistributionWasm::new(2.0, 1.5);
    let gamma_samples = gamma_dist.sample_array(5);
    println!("Gamma(2,1.5) samples: {:?}", gamma_samples);
    println!(
        "Gamma(2,1.5) mean: {:.6}, variance: {:.6}",
        gamma_dist.mean(),
        gamma_dist.variance()
    );

    // Beta distribution
    let beta_dist = BetaDistributionWasm::new(2.0, 3.0);
    let beta_samples = beta_dist.sample_array(5);
    println!("Beta(2,3) samples: {:?}", beta_samples);
    println!(
        "Beta(2,3) mean: {:.6}, variance: {:.6}",
        beta_dist.mean(),
        beta_dist.variance()
    );

    // Optimizers Demo / 最適化アルゴリズムデモ
    println!("\n⚡ Enhanced Optimizers:");

    // SGD demonstration
    let mut sgd = SGDWasm::new(0.01, 0.9, 0.0, 0.0001, false);
    let mut params = vec![1.0, 2.0, 3.0];
    let gradients = vec![0.1, 0.2, 0.3];

    println!("SGD before: {:?}", params);
    sgd.step("layer1", &mut params, &gradients);
    println!("SGD after:  {:?}", params);

    // Adam demonstration
    let mut adam = AdamWasm::new(0.001, 0.9, 0.999, 1e-8, 0.0);
    let mut adam_params = vec![1.0, 2.0, 3.0];

    println!("Adam before: {:?}", adam_params);
    adam.step("layer1", &mut adam_params, &gradients);
    println!("Adam after:  {:?}", adam_params);
    println!("Adam step count: {}", adam.get_step_count());

    // Simplified Autograd Demo / 簡素化自動微分デモ
    println!("\n🧮 Simplified Autograd:");

    let mut graph = ComputationGraphWasm::new();

    // Create variables
    let x_id = graph.create_variable(vec![2.0, 3.0], vec![2], true);
    let y_id = graph.create_variable(vec![1.0, 4.0], vec![2], true);

    // Perform operations
    if let Some(sum_id) = graph.add_variables(&x_id, &y_id) {
        if let Some(sum_data) = graph.get_variable_data(&sum_id) {
            println!("x + y = {:?}", sum_data);
        }
    }

    if let Some(mul_id) = graph.mul_variables(&x_id, &y_id) {
        if let Some(mul_data) = graph.get_variable_data(&mul_id) {
            println!("x * y = {:?}", mul_data);
        }
    }

    println!("Total variables in graph: {}", graph.variable_count());

    // Performance benchmark
    println!("\n⏱️  Performance Benchmark:");
    let iterations = 10000;
    let benchmark_results = benchmark_special_functions_wasm(iterations);
    println!("Special functions benchmark ({} iterations):", iterations);
    println!("  Gamma time: {:.2}ms", benchmark_results[0]);
    println!("  Bessel time: {:.2}ms", benchmark_results[1]);
    println!("  Error function time: {:.2}ms", benchmark_results[2]);
    println!("  Total time: {:.2}ms", benchmark_results[3]);

    // Statistical analysis demo / 統計解析デモ
    println!("\n📈 Statistical Analysis:");
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let stats = quick_stats_wasm(&test_data);
    println!("Test data: {:?}", test_data);
    println!(
        "Statistics: mean={:.3}, variance={:.3}, skewness={:.3}, kurtosis={:.3}",
        stats[0], stats[1], stats[2], stats[3]
    );

    // Learning rate scheduling demo / 学習率スケジューリングデモ
    println!("\n📉 Learning Rate Scheduling:");
    let initial_lr = 0.1;
    for step in [0, 10, 50, 100, 500, 1000] {
        let exponential_lr = learning_rate_schedule_wasm(initial_lr, step, 0.95, 100);
        let cosine_lr = cosine_annealing_wasm(initial_lr, step, 1000);
        println!(
            "  Step {}: Exponential LR={:.6}, Cosine LR={:.6}",
            step, exponential_lr, cosine_lr
        );
    }

    println!("\n✅ Enhanced WASM features demo completed!");
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("❌ This demo requires the 'wasm' feature to be enabled.");
    println!("Run with: cargo run --example wasm_enhanced_demo --features wasm");
}
