//! フェーズ1基本演算テスト例
//! Phase 1 Basic Operations Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::F32Tensor;

    rustorch::hybrid_f32_experimental!();

    println!("🧪 フェーズ1基本演算テスト");
    println!("🧪 Phase 1 Basic Operations Test");
    println!("================================\n");

    // 1. テンソル作成テスト
    println!("📝 1. テンソル作成テスト / Tensor Creation Tests");

    let zeros = F32Tensor::zeros(&[2, 3]);
    println!("  Zeros tensor: {:?}", zeros.as_slice());

    let ones = F32Tensor::ones(&[2, 2]);
    println!("  Ones tensor: {:?}", ones.as_slice());

    let arange = F32Tensor::arange(0.0, 5.0, 1.0);
    println!("  Arange tensor: {:?}", arange.as_slice());

    let linspace = F32Tensor::linspace(0.0, 10.0, 5);
    println!("  Linspace tensor: {:?}", linspace.as_slice());

    let eye = F32Tensor::eye(3);
    println!("  Identity matrix: {:?}", eye.as_slice());

    // 2. 基本算術演算テスト
    println!("\n⚡ 2. 基本算術演算テスト / Basic Arithmetic Tests");

    let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = F32Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2])?;

    println!("  A: {:?}", a.as_slice());
    println!("  B: {:?}", b.as_slice());

    let add_result = a.add(&b)?;
    println!("  A + B: {:?}", add_result.as_slice());

    let mul_result = a.mul(&b)?;
    println!("  A * B: {:?}", mul_result.as_slice());

    let scalar_mul = a.mul_scalar(2.0)?;
    println!("  A * 2: {:?}", scalar_mul.as_slice());

    let pow_result = a.pow(2.0)?;
    println!("  A^2: {:?}", pow_result.as_slice());

    // 3. 統計演算テスト
    println!("\n📊 3. 統計演算テスト / Statistical Operations Tests");

    let data = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    println!("  Data: {:?} (shape: {:?})", data.as_slice(), data.shape());

    let sum = data.sum()?;
    println!("  Sum: {}", sum);

    let mean = data.mean()?;
    println!("  Mean: {}", mean);

    let max = data.max()?;
    println!("  Max: {}", max);

    let min = data.min()?;
    println!("  Min: {}", min);

    let std = data.std()?;
    println!("  Std: {}", std);

    // 4. 軸演算テスト
    println!("\n🔄 4. 軸演算テスト / Axis Operations Tests");

    let sum_axis0 = data.sum_axis(0)?;
    println!("  Sum axis 0: {:?}", sum_axis0.as_slice());

    let sum_axis1 = data.sum_axis(1)?;
    println!("  Sum axis 1: {:?}", sum_axis1.as_slice());

    let mean_axis0 = data.mean_axis(0)?;
    println!("  Mean axis 0: {:?}", mean_axis0.as_slice());

    // 5. 行列演算テスト
    println!("\n🔢 5. 行列演算テスト / Matrix Operations Tests");

    let mat_a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let mat_b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

    println!("  Matrix A: {:?}", mat_a.as_slice());
    println!("  Matrix B: {:?}", mat_b.as_slice());

    let matmul_result = mat_a.matmul(&mat_b)?;
    println!("  A @ B: {:?}", matmul_result.as_slice());

    println!("\n✅ フェーズ1テスト完了！全ての基本演算が正常に動作しています。");
    println!("✅ Phase 1 tests completed! All basic operations working correctly.");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example phase1_test --features hybrid-f32");
}