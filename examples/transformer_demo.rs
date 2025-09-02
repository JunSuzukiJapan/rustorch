//! Transformer architecture demonstration (Phase 6)
//! Transformerアーキテクチャのデモンストレーション（フェーズ6）

use rustorch::autograd::Variable;
use rustorch::nn::{
    Embedding, LayerNorm, Module, MultiheadAttention, PositionalEncoding,
    TransformerEncoder, TransformerEncoderLayer,
};
use rustorch::tensor::Tensor;
use rustorch::error::{RusTorchError, RusTorchResult};

fn main() -> RusTorchResult<()> {
    println!("=== RusTorch Transformer Demo ===");
    println!("Transformerアーキテクチャのデモンストレーション");

    // Test individual components
    test_layer_norm()?;
    test_multi_head_attention()?;
    test_transformer_encoder_layer()?;
    test_complete_transformer_pipeline()?;

    println!("\n✅ All Transformer tests completed successfully!");
    println!("✅ すべてのTransformerテストが正常に完了しました！");
    
    Ok(())
}

fn test_layer_norm() -> RusTorchResult<()> {
    println!("\n--- Layer Normalization Test ---");
    println!("--- レイヤー正規化テスト ---");

    // Create LayerNorm for embedding dimension 128
    let layer_norm = LayerNorm::<f32>::new(vec![128], None, None);

    println!(
        "Created LayerNorm: normalized_shape={:?}, eps={:.2e}, elementwise_affine={}",
        layer_norm.normalized_shape(),
        layer_norm.eps(),
        layer_norm.elementwise_affine()
    );

    // Create input: batch_size=2, seq_length=10, d_model=128
    let batch_size = 2;
    let seq_length = 10;
    let d_model = 128;

    let input_data: Vec<f32> = (0..batch_size * seq_length * d_model)
        .map(|i| (i as f32) * 0.01 + 1.0) // Add 1.0 to avoid all zeros
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, seq_length, d_model]),
        false,
    );

    println!("Input shape: {:?}", input.data().read().unwrap().shape());

    // Forward pass
    let output = layer_norm.forward(&input);
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [2, 10, 128]");

    // Check parameter count
    let params = layer_norm.parameters();
    println!(
        "Number of parameters: {} (should be 2: weight + bias)",
        params.len()
    );
    
    Ok(())
}

fn test_multi_head_attention() -> RusTorchResult<()> {
    println!("\n--- Multi-Head Attention Test (Phase 6) ---");
    println!("--- マルチヘッドアテンションテスト（フェーズ6） ---");

    // Create MultiheadAttention: embed_dim=256, num_heads=8
    let embed_dim = 256;
    let num_heads = 8;
    let mha = MultiheadAttention::<f32>::new(embed_dim, num_heads, None, None, None, None, None)?;

    println!(
        "Created MultiheadAttention: embed_dim={}, num_heads={}, head_dim={}",
        mha.embed_dim(),
        mha.num_heads(),
        mha.head_dim()
    );

    // Create input: batch_size=2, seq_length=12, embed_dim=256
    let batch_size = 2;
    let seq_length = 12;

    let input_data: Vec<f32> = (0..batch_size * seq_length * embed_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, seq_length, embed_dim]),
        false,
    );

    println!("Input shape: {:?}", input.data().read().unwrap().shape());

    // Forward pass (self-attention) - Phase 6 API
    let (output, _attn_weights) = mha.forward(&input, &input, &input, None, None, None, None)?;
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [2, 12, 256] (same as input)");

    // Check parameter count
    let params = mha.parameters();
    println!(
        "Number of parameters: {} (Phase 6 implementation)",
        params.len()
    );
    
    Ok(())
}

fn test_transformer_encoder_layer() -> RusTorchResult<()> {
    println!("\n--- Transformer Encoder Layer Test (Phase 6) ---");
    println!("--- Transformerエンコーダー層テスト（フェーズ6） ---");

    // Create TransformerEncoderLayer: d_model=128, num_heads=4, d_ff=512
    let d_model = 128;
    let num_heads = 4;
    let d_ff = 512;

    let encoder_layer = TransformerEncoderLayer::<f32>::new(d_model, num_heads, d_ff, None, None)?;

    println!(
        "Created TransformerEncoderLayer: d_model={}, num_heads={}, d_ff={}",
        d_model, num_heads, d_ff
    );

    // Create input: batch_size=1, seq_length=8, d_model=128
    let batch_size = 1;
    let seq_length = 8;

    let input_data: Vec<f32> = (0..batch_size * seq_length * d_model)
        .map(|i| (i as f32) * 0.01)
        .collect();

    let input = Variable::new(
        Tensor::from_vec(input_data, vec![batch_size, seq_length, d_model]),
        false,
    );

    println!("Input shape: {:?}", input.data().read().unwrap().shape());

    // Forward pass - Phase 6 API
    let output = encoder_layer.forward(&input, None, None, None)?;
    let output_binding = output.data();
    let output_data = output_binding.read().unwrap();

    println!("Output shape: {:?}", output_data.shape());
    println!("Expected shape: [1, 8, 128] (same as input)");

    // Check parameter count
    let params = encoder_layer.parameters();
    println!(
        "Number of parameters: {} (multiple components)",
        params.len()
    );
    
    Ok(())
}

fn test_complete_transformer_pipeline() -> RusTorchResult<()> {
    println!("\n--- Complete Transformer Pipeline Test ---");
    println!("--- 完全なTransformerパイプラインテスト ---");

    // Parameters
    let vocab_size = 1000;
    let d_model = 128;
    let num_heads = 4;
    let d_ff = 512;
    let num_layers = 2;
    let max_length = 50;
    let seq_length = 16;
    let batch_size = 2;

    println!("Pipeline parameters:");
    println!(
        "  vocab_size: {}, d_model: {}, num_heads: {}",
        vocab_size, d_model, num_heads
    );
    println!(
        "  d_ff: {}, num_layers: {}, max_length: {}",
        d_ff, num_layers, max_length
    );
    println!("  seq_length: {}, batch_size: {}", seq_length, batch_size);

    // Step 1: Create embedding layers
    let word_embedding = Embedding::<f32>::new(vocab_size, d_model, Some(0), None, None);
    let pos_encoding = PositionalEncoding::<f32>::new(d_model, max_length, None);

    // Step 2: Create Transformer encoder (using Phase 6 implementation)
    // Note: Using individual layers for demonstration as full encoder needs update
    let encoder_layer = TransformerEncoderLayer::<f32>::new(d_model, num_heads, d_ff, None, None)?;

    // Step 3: Create input token sequences
    let token_sequences = vec![
        // Sequence 1: [5, 12, 7, 23, 1, 45, 8, 19, 33, 2, 0, 0, 0, 0, 0, 0]
        5.0, 12.0, 7.0, 23.0, 1.0, 45.0, 8.0, 19.0, 33.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        // Sequence 2: [8, 15, 3, 9, 11, 2, 67, 34, 21, 6, 13, 0, 0, 0, 0, 0]
        8.0, 15.0, 3.0, 9.0, 11.0, 2.0, 67.0, 34.0, 21.0, 6.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    let token_input = Variable::new(
        Tensor::from_vec(token_sequences, vec![batch_size, seq_length]),
        false,
    );

    println!(
        "\nStep 1: Token input shape: {:?}",
        token_input.data().read().unwrap().shape()
    );

    // Step 4: Convert tokens to embeddings (simplified batch processing)
    let mut embeddings_data = Vec::new();
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

            embeddings_data.extend_from_slice(word_emb_slice);
        }
    }

    let word_embeddings = Variable::new(
        Tensor::from_vec(embeddings_data, vec![batch_size, seq_length, d_model]),
        false,
    );

    println!(
        "Step 2: Word embeddings shape: {:?}",
        word_embeddings.data().read().unwrap().shape()
    );

    // Step 5: Add positional encoding
    let positioned_embeddings = pos_encoding.forward(&word_embeddings);

    println!(
        "Step 3: Positioned embeddings shape: {:?}",
        positioned_embeddings.data().read().unwrap().shape()
    );

    // Step 6: Pass through Transformer encoder layer (Phase 6)
    let encoder_output = encoder_layer.forward(&positioned_embeddings, None, None, None)?;

    println!(
        "Step 4: Transformer output shape: {:?}",
        encoder_output.data().read().unwrap().shape()
    );
    println!("Expected final shape: [2, 16, 128]");

    // Display some output values
    let output_binding = encoder_output.data();
    let output_data = output_binding.read().unwrap();
    let output_array = output_data.as_array();

    println!("\nSample output values for first sequence, first position:");
    for i in 0..5 {
        print!("  dim[{}]: {:.4}", i, output_array[[0, 0, i]]);
    }
    println!();

    // Count total parameters
    let word_params = word_embedding.parameters();
    let pos_params = pos_encoding.parameters();
    let encoder_params = encoder_layer.parameters();

    let total_params = word_params.len() + pos_params.len() + encoder_params.len();

    println!("\nParameter summary:");
    println!("  Word embedding: {} parameters", word_params.len());
    println!("  Positional encoding: {} parameters", pos_params.len());
    println!(
        "  Transformer encoder layer: {} parameters",
        encoder_params.len()
    );
    println!("  Total: {} parameters", total_params);

    println!("\n🎉 Complete Transformer pipeline working!");
    println!("🎉 完全なTransformerパイプラインが動作しています！");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_pipeline() -> RusTorchResult<()> {
        let vocab_size = 100;
        let d_model = 64;
        let num_heads = 4;
        let d_ff = 256;
        let num_layers = 1;
        let max_length = 20;

        // Create components with Phase 6 implementation
        let word_embedding = Embedding::<f32>::new(vocab_size, d_model, Some(0), None, None);
        let pos_encoding = PositionalEncoding::<f32>::new(d_model, max_length, None);
        let encoder_layer = TransformerEncoderLayer::<f32>::new(d_model, num_heads, d_ff, None, None)?;

        // Test with small sequence
        let token_input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.0], vec![1, 4]),
            false,
        );

        // Process tokens individually and combine for 3D embedding
        let mut embeddings_data = Vec::new();
        let token_binding = token_input.data();
        let token_array = token_binding.read().unwrap();

        for s in 0..4 {
            let token_idx = token_array.as_array()[[0, s]];
            let single_token = Variable::new(Tensor::from_vec(vec![token_idx], vec![1]), false);

            let word_emb = word_embedding.forward(&single_token);
            let word_emb_binding = word_emb.data();
            let word_emb_data = word_emb_binding.read().unwrap();
            let word_emb_slice = word_emb_data.as_slice().unwrap();

            embeddings_data.extend_from_slice(word_emb_slice);
        }

        let word_embeddings = Variable::new(
            Tensor::from_vec(embeddings_data, vec![1, 4, d_model]),
            false,
        );

        let positioned = pos_encoding.forward(&word_embeddings);

        // Now test full transformer encoder layer forward pass (Phase 6)
        let output = encoder_layer.forward(&positioned, None, None, None)?;

        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();

        // Verify full transformer output
        assert_eq!(output_data.shape(), &[1, 4, d_model]);

        println!("✅ Phase 6 Transformer pipeline successful!");
        println!(
            "✓ Word embedding: {} vocab → {} dimensions",
            vocab_size, d_model
        );
        println!(
            "✓ Positional encoding: max length {} → {} dimensions",
            max_length, d_model
        );
        println!(
            "✓ TransformerEncoderLayer output: batch=1, seq_len=4, d_model={}",
            d_model
        );
        println!("✓ Phase 6 MultiheadAttention working with PyTorch-compatible API!");
        
        Ok(())
    }
}
