//! Test Metal matmul kernel to verify matrix multiplication correctness
//!
//! This test verifies that the Metal GPU matmul implementation produces
//! correct results for a simple 2x3 × 3x2 matrix multiplication.

use rustorch::error::{RusTorchError, RusTorchResult};

#[cfg(feature = "metal")]
fn main() -> RusTorchResult<()> {
    use rustorch::gpu::metal_kernels::MetalKernelExecutor;

    eprintln!("🧪 Testing Metal matmul operation...");
    eprintln!();

    // Get Metal executor
    let executor_mutex = MetalKernelExecutor::get()?;
    let executor_guard = executor_mutex.lock().unwrap();
    let executor = executor_guard.as_ref()
        .ok_or_else(|| RusTorchError::tensor_op("Metal executor not initialized"))?;

    // Test data: 2x3 matrix × 3x2 matrix = 2x2 matrix
    //
    // A (2x3) = [[1, 2, 3],      B (3x2) = [[1, 2],
    //            [4, 5, 6]]                 [3, 4],
    //                                       [5, 6]]
    //
    // Expected C (2x2) = [[22, 28],
    //                     [49, 64]]
    //
    // Calculation:
    //   C[0][0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    //   C[0][1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    //   C[1][0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    //   C[1][1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64

    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 (row-major)
    let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2 (row-major)
    let mut c = vec![0.0f32; 4]; // 2x2

    eprintln!("📊 Input matrices:");
    eprintln!("   A (2x3, row-major): [[{}, {}, {}], [{}, {}, {}]]",
        a[0], a[1], a[2], a[3], a[4], a[5]);
    eprintln!("   B (3x2, row-major): [[{}, {}], [{}, {}], [{}, {}]]",
        b[0], b[1], b[2], b[3], b[4], b[5]);
    eprintln!();

    // Execute Metal matmul: matmul_f32(a, b, c, m, n, k)
    // where A is m×k, B is k×n, C is m×n
    // m=2 (rows of A), n=2 (cols of B), k=3 (cols of A = rows of B)
    executor.matmul_f32(&a, &b, &mut c, 2, 2, 3)?;

    eprintln!("🔍 Metal GPU Result:");
    eprintln!("   C (2x2): [[{}, {}], [{}, {}]]",
        c[0], c[1], c[2], c[3]);
    eprintln!();

    // Verify result (expected: [22, 28], [49, 64])
    let expected = vec![22.0f32, 28.0, 49.0, 64.0];
    eprintln!("✓ Expected Result:");
    eprintln!("   C (2x2): [[{}, {}], [{}, {}]]",
        expected[0], expected[1], expected[2], expected[3]);
    eprintln!();

    let epsilon = 0.001;
    let mut all_correct = true;

    for i in 0..4 {
        let diff = (c[i] - expected[i]).abs();
        let correct = diff <= epsilon;

        eprintln!("   C[{}]: got {:.6}, expected {:.6}, diff={:.9} {}",
            i, c[i], expected[i], diff,
            if correct { "✅" } else { "❌" });

        if !correct {
            all_correct = false;
        }
    }

    eprintln!();

    if all_correct {
        eprintln!("✅ Metal matmul test PASSED");
        eprintln!("   All values match expected results within epsilon={}", epsilon);
        Ok(())
    } else {
        eprintln!("❌ Metal matmul test FAILED");
        eprintln!("   Some values do not match expected results");
        eprintln!();
        eprintln!("🔍 Diagnosis:");
        eprintln!("   This indicates a bug in the Metal matmul kernel.");
        eprintln!("   Possible causes:");
        eprintln!("   1. Matrix layout mismatch (row-major vs column-major)");
        eprintln!("   2. Incorrect indexing in the kernel");
        eprintln!("   3. Parameter passing issue");
        Err(RusTorchError::tensor_op(
            "Metal matmul produced incorrect results"
        ))
    }
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("⚠️  Metal feature not enabled");
    eprintln!("   Run with: cargo run --example test_metal_matmul --features metal");
}
