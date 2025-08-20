//! 機械学習モデルデモ
//! Machine learning models demonstration

use rustorch::prelude::*;
use rustorch::models::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RusTorch Machine Learning Models Demo");
    println!("=========================================");
    
    // CNN モデルのデモ
    cnn_demo()?;
    
    // RNN/LSTM モデルのデモ
    rnn_demo()?;
    
    // Transformer モデルのデモ
    transformer_demo()?;
    
    // 訓練デモ
    training_demo()?;
    
    // モデル保存・読み込みデモ
    serialization_demo()?;
    
    Ok(())
}

/// CNN モデルのデモ
/// CNN model demonstration
fn cnn_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 CNN Model Demo");
    println!("-----------------");
    
    // 基本的な CNN モデル
    let cnn = CNNBuilder::new()
        .input_channels(3)
        .num_classes(10)
        .hidden_channels(vec![32, 64, 128])
        .dropout_rate(0.5)
        .build();
    
    println!("CNN Model created:");
    println!("{}", cnn.summary());
    println!("Configuration: {:?}", cnn.config());
    
    // ResNet モデル
    let resnet = ResNetBuilder::new()
        .resnet18()
        .num_classes(1000)
        .build();
    
    println!("\nResNet-18 Model created:");
    println!("{}", resnet.summary());
    
    Ok(())
}

/// RNN/LSTM モデルのデモ
/// RNN/LSTM model demonstration
fn rnn_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 RNN/LSTM Model Demo");
    println!("----------------------");
    
    // LSTM モデル
    let lstm = LSTMModelBuilder::new()
        .vocab_size(10000)
        .embedding_dim(128)
        .hidden_size(256)
        .num_layers(2)
        .num_classes(5)
        .dropout_rate(0.3)
        .bidirectional(true)
        .build();
    
    println!("LSTM Model created:");
    println!("{}", lstm.summary());
    
    // RNN モデル
    let rnn = RNNModelBuilder::new()
        .vocab_size(5000)
        .embedding_dim(100)
        .hidden_size(128)
        .num_classes(3)
        .build();
    
    println!("\nRNN Model created:");
    println!("{}", rnn.summary());
    
    Ok(())
}

/// Transformer モデルのデモ
/// Transformer model demonstration
fn transformer_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 Transformer Model Demo");
    println!("--------------------------");
    
    // 基本的な Transformer モデル
    let transformer = TransformerModelBuilder::new()
        .vocab_size(30000)
        .d_model(512)
        .nhead(8)
        .num_encoder_layers(6)
        .dim_feedforward(2048)
        .num_classes(2)
        .dropout_rate(0.1)
        .max_seq_length(512)
        .build();
    
    println!("Transformer Model created:");
    println!("{}", transformer.summary());
    
    // BERT モデル
    let bert = BERTBuilder::new()
        .bert_base()
        .num_labels(3)
        .build();
    
    println!("\nBERT-Base Model created:");
    println!("{}", bert.summary());
    
    // GPT モデル
    let gpt = GPT::gpt2_small();
    println!("\nGPT-2 Small Model created:");
    println!("{}", gpt.summary());
    
    Ok(())
}

/// 訓練デモ
/// Training demonstration
fn training_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏋️ Training Demo");
    println!("----------------");
    
    // モデル作成
    let mut model = CNNBuilder::new()
        .input_channels(1)
        .num_classes(10)
        .hidden_channels(vec![32, 64])
        .build();
    
    // 訓練設定
    let config = TrainingConfig {
        epochs: 5,
        batch_size: 32,
        learning_rate: 0.001,
        validation_frequency: 1,
        early_stopping_patience: Some(3),
        log_frequency: 1,
        ..Default::default()
    };
    
    println!("Training Configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Early stopping patience: {:?}", config.early_stopping_patience);
    
    // 推論エンジンのデモ
    let inference_engine = InferenceEngine::new(model, "cpu".to_string());
    println!("\nInference Engine created for CPU");
    
    Ok(())
}

/// モデル保存・読み込みデモ
/// Model serialization demonstration
fn serialization_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Model Serialization Demo");
    println!("---------------------------");
    
    // モデル作成
    let model = CNNBuilder::new()
        .input_channels(3)
        .num_classes(1000)
        .hidden_channels(vec![64, 128, 256, 512])
        .build();
    
    println!("Model created for serialization:");
    println!("{}", model.summary());
    
    // 保存パス
    let save_path = std::path::Path::new("model_demo.json");
    
    // モデル保存のシミュレーション
    println!("\nSaving model to: {:?}", save_path);
    println!("Format: JSON");
    
    // 実際の保存は省略（ファイルシステムへのアクセスが必要）
    // ModelSaver::save(&model, save_path, SerializationFormat::Json)?;
    
    println!("✅ Model save simulation completed");
    
    // モデル情報表示のシミュレーション
    println!("\nModel Information (simulated):");
    println!("  Parameters: ~1.2M");
    println!("  Size: ~4.8 MB");
    println!("  Format: JSON");
    println!("  Version: 0.1.9");
    
    Ok(())
}

/// パフォーマンステスト
/// Performance test
fn performance_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Performance Test");
    println!("------------------");
    
    use std::time::Instant;
    
    // 異なるサイズのモデルでパフォーマンステスト
    let model_configs = vec![
        ("Small CNN", vec![32, 64]),
        ("Medium CNN", vec![64, 128, 256]),
        ("Large CNN", vec![128, 256, 512, 1024]),
    ];
    
    for (name, channels) in model_configs {
        let start = Instant::now();
        
        let model = CNNBuilder::new()
            .input_channels(3)
            .num_classes(1000)
            .hidden_channels(channels.clone())
            .build();
        
        let creation_time = start.elapsed();
        
        println!("{}: Created in {:?}", name, creation_time);
        println!("  Channels: {:?}", channels);
        println!("  Config: {:?}", model.config());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cnn_creation() {
        let cnn = CNNBuilder::new()
            .input_channels(3)
            .num_classes(10)
            .build();
        
        assert_eq!(cnn.mode(), ModelMode::Train);
        let config = cnn.config();
        assert_eq!(config.get("model_type"), Some(&"CNN".to_string()));
        assert_eq!(config.get("num_classes"), Some(&"10".to_string()));
    }
    
    #[test]
    fn test_lstm_creation() {
        let lstm = LSTMModelBuilder::new()
            .vocab_size(1000)
            .embedding_dim(128)
            .hidden_size(256)
            .num_classes(5)
            .build();
        
        assert_eq!(lstm.mode(), ModelMode::Train);
        let config = lstm.config();
        assert_eq!(config.get("model_type"), Some(&"LSTM".to_string()));
        assert_eq!(config.get("num_classes"), Some(&"5".to_string()));
    }
    
    #[test]
    fn test_transformer_creation() {
        let transformer = TransformerModelBuilder::new()
            .vocab_size(10000)
            .d_model(512)
            .num_classes(2)
            .build();
        
        assert_eq!(transformer.mode(), ModelMode::Train);
        let config = transformer.config();
        assert_eq!(config.get("model_type"), Some(&"Transformer".to_string()));
    }
    
    #[test]
    fn test_bert_creation() {
        let bert = BERTBuilder::new()
            .bert_base()
            .num_labels(3)
            .build();
        
        assert_eq!(bert.mode(), ModelMode::Train);
        let config = bert.config();
        assert_eq!(config.get("model_type"), Some(&"BERT".to_string()));
    }
    
    #[test]
    fn test_model_mode_switching() {
        let mut cnn = CNNBuilder::new()
            .input_channels(3)
            .num_classes(10)
            .build();
        
        assert_eq!(cnn.mode(), ModelMode::Train);
        
        cnn.eval();
        assert_eq!(cnn.mode(), ModelMode::Eval);
        
        cnn.train();
        assert_eq!(cnn.mode(), ModelMode::Train);
    }
}
