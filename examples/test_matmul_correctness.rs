use rustorch::hybrid_f32::tensor::F32Tensor;
use rustorch::hybrid_f32::models::{F32LlamaModel, DeviceType};
use rustorch::hybrid_f32::error::F32Result;

fn main() -> F32Result<()> {
    println!("🧪 Testing Metal matmul correctness\n");

    // Test 1: Simple 2x3 @ 3x2 matmul
    println!("Test 1: Simple matmul");
    let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let b = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;

    println!("  A (2x3): {:?}", a.as_slice());
    println!("  B (3x2): {:?}", b.as_slice());

    let c = a.matmul(&b)?;
    println!("  C = A @ B: {:?}", c.as_slice());
    println!("  Expected: [22.0, 28.0, 49.0, 64.0]");

    // Manual verification:
    // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22 ✓
    // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28 ✓
    // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49 ✓
    // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64 ✓

    let expected = vec![22.0, 28.0, 49.0, 64.0];
    let result = c.as_slice();
    let mut all_correct = true;
    for i in 0..4 {
        let diff = (result[i] - expected[i]).abs();
        if diff > 1e-5 {
            println!("  ❌ Mismatch at index {}: got {}, expected {}", i, result[i], expected[i]);
            all_correct = false;
        }
    }
    if all_correct {
        println!("  ✅ Test 1 PASSED\n");
    } else {
        println!("  ❌ Test 1 FAILED\n");
    }

    // Test 2: LM head style matmul [1, hidden] @ [hidden, vocab]
    println!("Test 2: LM head style matmul [1, 4] @ [4, 8]");
    let hidden = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])?;
    let weights = F32Tensor::from_vec(
        vec![
            // Row 0 (for token 0): 1, 2, 3, 4, 5, 6, 7, 8
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            // Row 1 (for token 1): 2, 3, 4, 5, 6, 7, 8, 9
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            // Row 2 (for token 2): 3, 4, 5, 6, 7, 8, 9, 10
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            // Row 3 (for token 3): 4, 5, 6, 7, 8, 9, 10, 11
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ],
        &[4, 8]
    )?;

    println!("  Hidden [1, 4]: {:?}", hidden.as_slice());
    println!("  Weights [4, 8] (row-major):");
    for i in 0..4 {
        let row: Vec<f32> = weights.as_slice()[i*8..(i+1)*8].to_vec();
        println!("    Row {}: {:?}", i, row);
    }

    let logits = hidden.matmul(&weights)?;
    println!("  Logits [1, 8]: {:?}", logits.as_slice());

    // Manual calculation for token 0:
    // logit[0] = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // For token 1:
    // logit[1] = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    // For token 7:
    // logit[7] = 1*8 + 2*9 + 3*10 + 4*11 = 8 + 18 + 30 + 44 = 100

    let expected_logits = vec![30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    println!("  Expected: {:?}", expected_logits);

    let logits_slice = logits.as_slice();
    let mut all_correct = true;
    for i in 0..8 {
        let diff = (logits_slice[i] - expected_logits[i]).abs();
        if diff > 1e-5 {
            println!("  ❌ Mismatch at token {}: got {}, expected {}", i, logits_slice[i], expected_logits[i]);
            all_correct = false;
        }
    }
    if all_correct {
        println!("  ✅ Test 2 PASSED\n");
    } else {
        println!("  ❌ Test 2 FAILED\n");
    }

    // Test 3: Extract actual weights from LM head and verify layout
    println!("Test 3: Verify actual model weight layout");
    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    println!("  Loading model...");
    let model = F32LlamaModel::from_gguf_with_device(&model_path, DeviceType::Cpu)?;

    let lm_head_weight = model.get_weight("output.weight")
        .or_else(|| model.get_weight("lm_head.weight"))
        .expect("LM head weight not found");

    println!("  LM head shape: {:?}", lm_head_weight.shape());
    println!("  Layout should be [hidden_size, vocab_size] = [2048, 32000]");

    // Check if weights for token 1552 are accessible
    let lm_data = lm_head_weight.as_slice();
    if lm_head_weight.shape()[0] == 2048 && lm_head_weight.shape()[1] == 32000 {
        // Row-major: weights[row * 32000 + col]
        // For token 1552, we need column 1552
        // First 5 elements of column 1552 (from rows 0-4)
        let mut col_1552_sample: Vec<f32> = Vec::new();
        for row in 0..5 {
            let idx = row * 32000 + 1552;
            col_1552_sample.push(lm_data[idx]);
        }
        println!("  Token 1552 weights (first 5 rows): {:?}", col_1552_sample);
        println!("  These are the weights that should be dotted with hidden state");
        println!("  ✅ Weight layout verified\n");
    } else {
        println!("  ⚠️  Unexpected shape: {:?}", lm_head_weight.shape());
    }

    Ok(())
}
