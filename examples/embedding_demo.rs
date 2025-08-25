//! Embedding layers demonstration
//! 埋め込み層のデモンストレーション

use rustorch::autograd::Variable;
use rustorch::nn::{Embedding, Module, PositionalEmbedding, SinusoidalPositionalEncoding};
use rustorch::tensor::Tensor;

fn main() {
    println!("=== RusTorch Embedding Layers Demo ===");
    println!("埋め込み層のデモンストレーション");

    // Test Word Embedding
    test_word_embedding();

    // Test Positional Embedding (learnable)
    test_positional_embedding();

    // Test Sinusoidal Positional Encoding (fixed)
    test_sinusoidal_encoding();

    // Test combined embedding + positional encoding
    test_combined_embeddings();

    println!("\n✅ All embedding tests completed successfully!");
    println!("✅ すべての埋め込みテストが正常に完了しました！");
}

fn test_word_embedding() {
    println!("\n--- Word Embedding Test ---");
    println!("--- 単語埋め込みテスト ---");

    // Create embedding: vocab_size=1000, embedding_dim=128, padding_idx=0
    let embedding = Embedding::<f32>::new(1000, 128, Some(0), None, None);

    println!(
        "Created embedding: vocab_size={}, embedding_dim={}, padding_idx={:?}",
        embedding.vocab_size(),
        embedding.embedding_dim(),
        embedding.padding_idx()
    );

    // Create input with token indices [1, 5, 10, 0] (0 is padding)
    let input_indices = vec![1.0, 5.0, 10.0, 0.0];
    let input = Variable::new(Tensor::from_vec(input_indices, vec![4]), false);

    // Forward pass
    let output = embedding.forward(&input);
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Input shape: {:?}", input.data().read().unwrap().shape());
    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [4, 128] (4 tokens, 128 embedding dims)");

    // Check that padding index produces zeros
    let output_array = output_data.as_array();
    let padding_embedding_norm: f32 = (0..128)
        .map(|i| output_array[[3, i]].powi(2))
        .sum::<f32>()
        .sqrt();

    println!(
        "Padding embedding norm: {:.6} (should be close to 0)",
        padding_embedding_norm
    );

    // Check parameter count
    let params = embedding.parameters();
    println!("Number of parameters: {} (should be 1)", params.len());

    if let Some(weight) = params.first() {
        let weight_binding = weight.data();
        let weight_data = weight_binding.read().unwrap();
        println!("Weight shape: {:?}", weight_data.shape());
    }
}

fn test_positional_embedding() {
    println!("\n--- Positional Embedding Test ---");
    println!("--- 位置埋め込みテスト ---");

    // Create positional embedding: max_length=100, embedding_dim=64
    let pos_embedding = PositionalEmbedding::<f32>::new(100, 64);

    println!(
        "Created positional embedding: max_length={}, embedding_dim={}",
        pos_embedding.max_length(),
        pos_embedding.embedding_dim()
    );

    // Create input: batch_size=2, seq_length=5, embedding_dim=64
    let batch_size = 2;
    let seq_length = 5;
    let embedding_dim = 64;

    let input_data: Vec<f32> = (0..batch_size * seq_length * embedding_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, seq_length, embedding_dim]),
        false,
    );

    println!("Input shape: {:?}", input.data().read().unwrap().shape());

    // Forward pass
    let output = pos_embedding.forward(&input);
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [2, 5, 64] (batch_size, seq_length, embedding_dim)");

    // Check that output is different from input (positional info added)
    let input_binding = input.data();
    let input_guard = input_binding.read().unwrap();
    let input_array = input_guard.as_array();
    let output_array = output_data.as_array();

    let mut differences = 0;
    for i in 0..10 {
        // Check first 10 elements
        if (input_array.as_slice().unwrap()[i] - output_array.as_slice().unwrap()[i]).abs() > 1e-6 {
            differences += 1;
        }
    }

    println!("Elements with positional differences: {}/10", differences);

    // Check parameter count
    let params = pos_embedding.parameters();
    println!("Number of parameters: {} (should be 1)", params.len());
}

fn test_sinusoidal_encoding() {
    println!("\n--- Sinusoidal Positional Encoding Test ---");
    println!("--- 正弦波位置エンコーディングテスト ---");

    // Create sinusoidal encoding: max_length=50, embedding_dim=32
    let sin_encoding = SinusoidalPositionalEncoding::<f32>::new(50, 32);

    println!(
        "Created sinusoidal encoding: max_length={}, embedding_dim={}",
        sin_encoding.max_length(),
        sin_encoding.embedding_dim()
    );

    // Create input: batch_size=1, seq_length=10, embedding_dim=32
    let batch_size = 1;
    let seq_length = 10;
    let embedding_dim = 32;

    let input_data: Vec<f32> = (0..batch_size * seq_length * embedding_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, seq_length, embedding_dim]),
        false,
    );

    println!("Input shape: {:?}", input.data().read().unwrap().shape());

    // Forward pass
    let output = sin_encoding.forward(&input);
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [1, 10, 32] (batch_size, seq_length, embedding_dim)");

    // Check that sinusoidal encoding has no parameters (fixed encoding)
    let params = sin_encoding.parameters();
    println!(
        "Number of parameters: {} (should be 0 for fixed encoding)",
        params.len()
    );

    // Display first few positional encodings
    let output_array = output_data.as_array();
    println!("First position encoding values:");
    for i in 0..5 {
        print!("  pos[0][{}]: {:.4}", i, output_array[[0, 0, i]]);
    }
    println!();

    println!("Second position encoding values:");
    for i in 0..5 {
        print!("  pos[1][{}]: {:.4}", i, output_array[[0, 1, i]]);
    }
    println!();
}

fn test_combined_embeddings() {
    println!("\n--- Combined Embeddings Test ---");
    println!("--- 結合埋め込みテスト ---");

    // Simulate a simple NLP pipeline:
    // 1. Token IDs -> Word Embeddings
    // 2. Add Positional Embeddings

    let vocab_size = 100;
    let embedding_dim = 32;
    let max_length = 20;
    let seq_length = 8;
    let batch_size = 2;

    // Create word embedding
    let word_embedding = Embedding::<f32>::new(vocab_size, embedding_dim, Some(0), None, None);

    // Create positional embedding
    let pos_embedding = PositionalEmbedding::<f32>::new(max_length, embedding_dim);

    println!(
        "Word embedding: vocab_size={}, embedding_dim={}",
        word_embedding.vocab_size(),
        word_embedding.embedding_dim()
    );
    println!(
        "Positional embedding: max_length={}, embedding_dim={}",
        pos_embedding.max_length(),
        pos_embedding.embedding_dim()
    );

    // Create token indices for two sequences
    let token_indices = vec![
        // Sequence 1: [5, 12, 7, 23, 1, 0, 0, 0] (padded)
        5.0, 12.0, 7.0, 23.0, 1.0, 0.0, 0.0, 0.0,
        // Sequence 2: [8, 15, 3, 9, 11, 2, 0, 0] (padded)
        8.0, 15.0, 3.0, 9.0, 11.0, 2.0, 0.0, 0.0,
    ];

    let token_input = Variable::new(
        Tensor::from_vec(token_indices, vec![batch_size, seq_length]),
        false,
    );

    println!(
        "Token input shape: {:?}",
        token_input.data().read().unwrap().shape()
    );

    // Step 1: Get word embeddings
    // Note: In a full implementation, we'd need to handle batch processing properly
    // For now, we'll process each sequence separately

    let mut word_embeddings_data = Vec::new();
    let token_binding = token_input.data();
    let token_array = token_binding.read().unwrap();

    for b in 0..batch_size {
        for s in 0..seq_length {
            let token_idx = token_array.as_array()[[b, s]];
            let single_token = Variable::new(Tensor::from_vec(vec![token_idx], vec![1]), false);

            let word_emb = word_embedding.forward(&single_token);
            let word_emb_binding = word_emb.data();
            let word_emb_data = word_emb_binding.read().unwrap();
            let word_emb_slice = word_emb_data.as_slice().unwrap();

            word_embeddings_data.extend_from_slice(word_emb_slice);
        }
    }

    let word_embeddings = Variable::new(
        Tensor::from_vec(
            word_embeddings_data,
            vec![batch_size, seq_length, embedding_dim],
        ),
        false,
    );

    println!(
        "Word embeddings shape: {:?}",
        word_embeddings.data().read().unwrap().shape()
    );

    // Step 2: Add positional embeddings
    let final_embeddings = pos_embedding.forward(&word_embeddings);
    let final_binding = final_embeddings.data();
    let final_data = final_binding.read().unwrap();

    println!("Final embeddings shape: {:?}", final_data.shape());
    println!("Expected shape: [2, 8, 32] (batch_size, seq_length, embedding_dim)");

    // Display some embedding values
    let final_array = final_data.as_array();
    println!("Sample embeddings for first token of first sequence:");
    for i in 0..5 {
        print!("  dim[{}]: {:.4}", i, final_array[[0, 0, i]]);
    }
    println!();

    println!("Sample embeddings for second token of first sequence:");
    for i in 0..5 {
        print!("  dim[{}]: {:.4}", i, final_array[[0, 1, i]]);
    }
    println!();

    // Count total parameters
    let word_params = word_embedding.parameters();
    let pos_params = pos_embedding.parameters();
    let total_params = word_params.len() + pos_params.len();

    println!(
        "Total parameters: {} (word: {}, positional: {})",
        total_params,
        word_params.len(),
        pos_params.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_embedding_basic_functionality() {
        let embedding = Embedding::<f32>::new(10, 4, None, None, None);

        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]), false);

        let output = embedding.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        assert_eq!(output_data.shape(), &[3, 4]);
    }

    #[test]
    fn test_positional_embedding_basic() {
        let pos_emb = PositionalEmbedding::<f32>::new(50, 16);

        let input = Variable::new(
            Tensor::from_vec(vec![0.1; 2 * 10 * 16], vec![2, 10, 16]),
            false,
        );

        let output = pos_emb.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        assert_eq!(output_data.shape(), &[2, 10, 16]);
    }

    #[test]
    fn test_sinusoidal_encoding_basic() {
        let sin_enc = SinusoidalPositionalEncoding::<f32>::new(20, 8);

        let input = Variable::new(Tensor::from_vec(vec![0.1; 1 * 5 * 8], vec![1, 5, 8]), false);

        let output = sin_enc.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        assert_eq!(output_data.shape(), &[1, 5, 8]);
        assert_eq!(sin_enc.parameters().len(), 0); // No learnable parameters
    }
}
