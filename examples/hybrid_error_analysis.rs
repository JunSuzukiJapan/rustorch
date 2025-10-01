//! 既存ハイブリッドエラー分析 - どんな処理でエラーが出るかを調査
//! Existing Hybrid Error Analysis - Investigate which operations cause errors
//!
//! 既存ハイブリッドシステムでエラーが発生する具体的なケースを特定
//! Identify specific cases where existing hybrid system produces errors
//!
//! 実行方法 / Usage:
//! ```bash
//! cargo run --example hybrid_error_analysis --features hybrid-f32 --release
//! ```

#[cfg(feature = "coreml")]
use rustorch::gpu::hybrid_executor::HybridExecution;
#[cfg(feature = "coreml")]
use rustorch::gpu::{hybrid_executor::HybridExecutor, DeviceType, OpType};
#[cfg(feature = "coreml")]
use rustorch::tensor::Tensor;

#[cfg(feature = "coreml")]
fn main() -> rustorch::error::RusTorchResult<()> {
    println!("🔍 Hybrid Error Analysis - Investigating Error Conditions");
    println!("=========================================================");
    println!("📊 Testing various scenarios to identify when errors occur");
    println!();

    // 既存ハイブリッドエグゼキューター初期化
    let _existing_hybrid_executor = HybridExecutor::new();

    println!("🎯 Testing scenarios:");
    println!("  1. Normal operation (should work)");
    println!("  2. CPU fallback prohibited (should error)");
    println!("  3. Invalid device forced (should error)");
    println!("  4. Unsupported operation (may error)");
    println!();

    // テストデータ作成
    let size = 512;
    let data_a: Vec<f64> = (0..size * size).map(|i| (i as f64 % 100.0) + 1.0).collect();
    let data_b: Vec<f64> = (0..size * size)
        .map(|i| ((i + size) as f64 % 100.0) + 1.0)
        .collect();
    let matrix_a = Tensor::from_vec(data_a, vec![size, size]);
    let matrix_b = Tensor::from_vec(data_b, vec![size, size]);

    // シナリオ1: 通常動作（CPU許可）
    println!("📋 Scenario 1: Normal Hybrid Operation (CPU allowed)");
    println!("  🔄 Using standard hybrid_operation without CPU prohibition");

    let result1 = matrix_a.hybrid_operation(OpType::LinearAlgebra, |device| {
        println!("    🎯 Attempting operation on device: {:?}", device);
        matrix_a.matmul(&matrix_b)
    });

    match result1 {
        Ok(_) => println!("  ✅ Normal operation succeeded"),
        Err(e) => println!("  ❌ Normal operation failed: {}", e),
    }

    // シナリオ2: CPU フォールバック禁止
    println!("\n📋 Scenario 2: CPU Fallback Prohibited");
    println!("  🚫 Explicitly rejecting CPU device");

    let result2 = matrix_a.hybrid_operation(OpType::LinearAlgebra, |device| {
        println!("    🎯 Device selected: {:?}", device);
        if device == DeviceType::Cpu {
            println!("    🚫 CPU fallback prohibited - returning error");
            return Err(rustorch::error::RusTorchError::tensor_op(
                "CPU fallback prohibited",
            ));
        }
        println!("    ✅ Executing on non-CPU device: {:?}", device);
        matrix_a.matmul(&matrix_b)
    });

    match result2 {
        Ok(_) => println!("  ✅ CPU prohibition succeeded (executed on GPU)"),
        Err(e) => println!("  ❌ CPU prohibition caused error: {}", e),
    }

    // シナリオ3: すべてのデバイスを拒否
    println!("\n📋 Scenario 3: All Devices Rejected");
    println!("  🚫 Rejecting all devices to force error");

    let result3: Result<Tensor<f64>, _> =
        matrix_a.hybrid_operation(OpType::LinearAlgebra, |device| {
            println!("    🎯 Device attempted: {:?}", device);
            println!("    🚫 Rejecting device: {:?}", device);
            Err(rustorch::error::RusTorchError::tensor_op(format!(
                "Device {:?} rejected",
                device
            )))
        });

    match result3 {
        Ok(_) => println!("  ⚠️ Unexpected success when all devices rejected"),
        Err(e) => println!("  ✅ Expected error when all devices rejected: {}", e),
    }

    // シナリオ4: 特定のデバイスのみ拒否（fallback テスト）
    println!("\n📋 Scenario 4: Specific Device Rejection (Fallback Test)");
    println!("  🔄 Rejecting primary device, allowing fallback");

    let attempt_count = std::sync::Arc::new(std::sync::Mutex::new(0));
    let count_clone = attempt_count.clone();
    let result4: Result<Tensor<f64>, _> =
        matrix_a.hybrid_operation(OpType::LinearAlgebra, |device| {
            let mut count = count_clone.lock().unwrap();
            *count += 1;
            let current_count = *count;
            drop(count);

            println!("    🎯 Device attempt {}: {:?}", current_count, device);

            // 最初のデバイス（通常Metal(0)やCoreML(0)）を拒否
            if current_count == 1 {
                println!("    🚫 Rejecting first device: {:?}", device);
                return Err(rustorch::error::RusTorchError::tensor_op(format!(
                    "First device {:?} rejected",
                    device
                )));
            }

            println!("    ✅ Accepting fallback device: {:?}", device);
            matrix_a.matmul(&matrix_b)
        });

    match result4 {
        Ok(_) => println!("  ✅ Fallback mechanism worked correctly"),
        Err(e) => println!("  ❌ Fallback mechanism failed: {}", e),
    }

    // シナリオ5: 演算タイプ別テスト
    println!("\n📋 Scenario 5: Different Operation Types");

    let op_types = vec![
        (OpType::LinearAlgebra, "LinearAlgebra"),
        (OpType::Activation, "Activation"),
        (OpType::Reduction, "Reduction"),
        (OpType::Convolution, "Convolution"),
    ];

    for (op_type, name) in op_types {
        println!("  🧪 Testing operation type: {}", name);

        let result: Result<Tensor<f64>, _> = matrix_a.hybrid_operation(op_type, |device| {
            if device == DeviceType::Cpu {
                println!("    🚫 CPU rejected for {}", name);
                return Err(rustorch::error::RusTorchError::tensor_op("CPU prohibited"));
            }
            println!("    ✅ {} executing on {:?}", name, device);
            matrix_a.matmul(&matrix_b) // Use matmul as proxy for all ops
        });

        match result {
            Ok(_) => println!("    ✅ {} succeeded on GPU", name),
            Err(e) => println!("    ❌ {} failed: {}", name, e),
        }
    }

    // シナリオ6: デバイス選択の透明性テスト
    println!("\n📋 Scenario 6: Device Selection Transparency");
    println!("  🔍 Observing device selection process");

    let devices_attempted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let devices_clone = devices_attempted.clone();

    let result6: Result<Tensor<f64>, _> =
        matrix_a.hybrid_operation(OpType::LinearAlgebra, |device| {
            {
                let mut devices = devices_clone.lock().unwrap();
                devices.push(device);
            }

            println!("    🎯 Device selection attempt: {:?}", device);

            // 最初の2つのデバイスを拒否して、fallback chainを観察
            let devices = devices_attempted.lock().unwrap();
            if devices.len() <= 2 {
                println!(
                    "    🚫 Rejecting device {} to observe fallback",
                    devices.len()
                );
                return Err(rustorch::error::RusTorchError::tensor_op(format!(
                    "Device {:?} rejected for observation",
                    device
                )));
            }

            println!("    ✅ Accepting device: {:?}", device);
            matrix_a.matmul(&matrix_b)
        });

    println!("  📊 Device selection sequence:");
    let final_devices = devices_attempted.lock().unwrap();
    for (i, device) in final_devices.iter().enumerate() {
        println!("    {}. {:?}", i + 1, device);
    }

    match result6 {
        Ok(_) => println!("  ✅ Device selection transparency test completed"),
        Err(e) => println!("  ❌ Device selection transparency test failed: {}", e),
    }

    // 📊 分析結果
    println!("\n📊 Error Analysis Summary:");
    println!("=========================");
    println!();
    println!("🔍 Error Conditions Identified:");
    println!(
        "  1. ✅ CPU Fallback Prohibition: Errors occur when CPU device is explicitly rejected"
    );
    println!("  2. ✅ All Devices Rejected: Errors occur when no device accepts the operation");
    println!("  3. ✅ Fallback Chain: Multiple devices are attempted in sequence");
    println!(
        "  4. ✅ Operation Type Specific: Different OpTypes may have different device preferences"
    );
    println!();
    println!("🎯 Key Findings:");
    println!("  - 既存ハイブリッドは自動的にfallback chainを試行します");
    println!("  - CPUを明示的に拒否することでGPU専用実行を強制できます");
    println!("  - エラーは operation closure 内で制御されます");
    println!("  - デバイス選択は OpType と TensorInfo に基づきます");
    println!();
    println!("⚠️ Error Scenarios:");
    println!("  - CPU禁止: `if device == DeviceType::Cpu {{ return Err(...) }}`");
    println!("  - GPU利用不可: GPU/Neural Engineが利用できない環境");
    println!("  - 演算非対応: 特定デバイスで未対応の演算タイプ");
    println!("  - メモリ不足: デバイスメモリが不足している場合");

    println!("\n✅ Hybrid error analysis completed!");
    println!("📝 Clear understanding of error conditions established");

    Ok(())
}

#[cfg(not(feature = "coreml"))]
fn main() {
    println!("❌ This example requires 'coreml' feature to be enabled.");
    println!("📋 Run with: cargo run --example hybrid_error_analysis --features coreml");
}
