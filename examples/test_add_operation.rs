use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 Testing element-wise addition");

    // Test 1: Simple vector addition
    let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let b = F32Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[4])?;
    let c = a.add(&b)?;
    println!("\nTest 1: Vector addition");
    println!("  a: {:?}", a.as_slice());
    println!("  b: {:?}", b.as_slice());
    println!("  a + b: {:?}", c.as_slice());
    println!("  Expected: [1.5, 3.0, 4.5, 6.0]");

    // Test 2: Residual connection style (larger vector)
    let hidden = F32Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &[8]
    )?;
    let residual = F32Tensor::from_vec(
        vec![0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08],
        &[8]
    )?;
    let result = hidden.add(&residual)?;
    println!("\nTest 2: Residual connection style");
    println!("  hidden: {:?}", hidden.as_slice());
    println!("  residual: {:?}", residual.as_slice());
    println!("  result: {:?}", result.as_slice());
    println!("  Expected: [0.11, 0.18, 0.33, 0.36, 0.55, 0.54, 0.77, 0.72]");

    // Test 3: 2D tensor addition (like transformer layers)
    let t1 = F32Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3]
    )?;
    let t2 = F32Tensor::from_vec(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &[2, 3]
    )?;
    let t3 = t1.add(&t2)?;
    println!("\nTest 3: 2D tensor addition");
    println!("  Shape: {:?}", t3.shape());
    println!("  Result: {:?}", t3.as_slice());
    println!("  Expected: [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]");

    println!("\n✅ All add() tests completed");
    Ok(())
}
