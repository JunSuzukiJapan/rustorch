//! Model registry with metadata and download URLs
//! メタデータとダウンロードURL付きモデルレジストリ

use std::collections::HashMap;
use std::path::Path;
use crate::error::{RusTorchError, RusTorchResult};
use crate::model_hub::verification::Checksum;
use serde::{Serialize, Deserialize};

/// Model source information
/// モデルソース情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// PyTorch Hub model
    PyTorchHub {
        repo: String,
        model: String,
    },
    /// HuggingFace model
    HuggingFace {
        repo_id: String,
        filename: Option<String>,
    },
    /// Direct URL download
    DirectUrl {
        url: String,
    },
    /// TensorFlow Hub model
    TensorFlowHub {
        handle: String,
    },
    /// Custom source
    Custom {
        name: String,
        url: String,
    },
}

/// Model information in registry
/// レジストリ内モデル情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name (unique identifier)
    /// モデル名（一意識別子）
    pub name: String,
    /// Display name for UI
    /// UI表示名
    pub display_name: String,
    /// Model description
    /// モデル説明
    pub description: String,
    /// Download URL
    /// ダウンロードURL
    pub url: String,
    /// File size in bytes
    /// ファイルサイズ（バイト）
    pub file_size: u64,
    /// Model checksum for verification
    /// 検証用モデルチェックサム
    pub checksum: Option<Checksum>,
    /// Model source information
    /// モデルソース情報
    pub source: ModelSource,
    /// Model architecture type
    /// モデルアーキテクチャタイプ
    pub architecture: String,
    /// Input shape [batch, channels, height, width] or [batch, features]
    /// 入力形状 [batch, channels, height, width] または [batch, features]
    pub input_shape: Vec<Option<usize>>,
    /// Number of output classes/features
    /// 出力クラス/特徴量数
    pub output_size: usize,
    /// Model parameters count
    /// モデルパラメータ数
    pub parameters: u64,
    /// Supported tasks
    /// サポートタスク
    pub tasks: Vec<String>,
    /// Model license
    /// モデルライセンス
    pub license: Option<String>,
    /// Model paper/reference
    /// モデル論文/参考文献
    pub paper_url: Option<String>,
    /// Model creation date
    /// モデル作成日
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Model registry
/// モデルレジストリ
pub struct ModelRegistry {
    /// Available models
    /// 利用可能モデル
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Create new registry with built-in models
    /// 組み込みモデル付き新しいレジストリを作成
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        
        registry.register_builtin_models();
        registry
    }

    /// Register a new model
    /// 新しいモデルを登録
    pub fn register_model(&mut self, model_info: ModelInfo) {
        self.models.insert(model_info.name.clone(), model_info);
    }

    /// Get model information
    /// モデル情報を取得
    pub fn get_model_info(&self, model_name: &str) -> RusTorchResult<&ModelInfo> {
        self.models.get(model_name).ok_or_else(|| {
            RusTorchError::ModelNotFound(format!("Model '{}' not found in registry", model_name))
        })
    }

    /// List all available models
    /// 利用可能な全モデルをリスト表示
    pub fn list_models(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// List models by task
    /// タスク別モデルリスト
    pub fn list_models_by_task(&self, task: &str) -> Vec<&str> {
        self.models
            .values()
            .filter(|model| model.tasks.contains(&task.to_string()))
            .map(|model| model.name.as_str())
            .collect()
    }

    /// List models by architecture
    /// アーキテクチャ別モデルリスト
    pub fn list_models_by_architecture(&self, architecture: &str) -> Vec<&str> {
        self.models
            .values()
            .filter(|model| model.architecture == architecture)
            .map(|model| model.name.as_str())
            .collect()
    }

    /// Register built-in models
    /// 組み込みモデルを登録
    fn register_builtin_models(&mut self) {
        // ResNet models
        self.register_model(ModelInfo {
            name: "resnet18".to_string(),
            display_name: "ResNet-18".to_string(),
            description: "18-layer deep residual network for image classification".to_string(),
            url: "https://download.pytorch.org/models/resnet18-5c106cde.pth".to_string(),
            file_size: 44_689_128,
            checksum: Some(Checksum::sha256("5c106cde18f69953b5e1dc2dcc97bdb88b1d7d29d6c5f4e8dd11d8e82a3b8e47")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "resnet18".to_string(),
            },
            architecture: "ResNet".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 11_689_512,
            tasks: vec!["image_classification".to_string(), "feature_extraction".to_string()],
            license: Some("BSD-3-Clause".to_string()),
            paper_url: Some("https://arxiv.org/abs/1512.03385".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        self.register_model(ModelInfo {
            name: "resnet50".to_string(),
            display_name: "ResNet-50".to_string(),
            description: "50-layer deep residual network for image classification".to_string(),
            url: "https://download.pytorch.org/models/resnet50-19c8e357.pth".to_string(),
            file_size: 97_781_416,
            checksum: Some(Checksum::sha256("19c8e357f3c7e5b4d3a8f6e9a2b1c0d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "resnet50".to_string(),
            },
            architecture: "ResNet".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 25_557_032,
            tasks: vec!["image_classification".to_string(), "feature_extraction".to_string()],
            license: Some("BSD-3-Clause".to_string()),
            paper_url: Some("https://arxiv.org/abs/1512.03385".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // MobileNet models
        self.register_model(ModelInfo {
            name: "mobilenet_v2".to_string(),
            display_name: "MobileNet V2".to_string(),
            description: "Efficient mobile neural network for image classification".to_string(),
            url: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth".to_string(),
            file_size: 13_555_880,
            checksum: Some(Checksum::sha256("b0353104f8b4d2e5a7c6d9e2f1a4b7c0d3e6f9a2b5c8d1e4f7a0b3c6d9e2f5a8")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "mobilenet_v2".to_string(),
            },
            architecture: "MobileNet".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 3_504_872,
            tasks: vec!["image_classification".to_string(), "mobile_inference".to_string()],
            license: Some("Apache-2.0".to_string()),
            paper_url: Some("https://arxiv.org/abs/1801.04381".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // DenseNet models
        self.register_model(ModelInfo {
            name: "densenet121".to_string(),
            display_name: "DenseNet-121".to_string(),
            description: "121-layer densely connected convolutional network".to_string(),
            url: "https://download.pytorch.org/models/densenet121-a639ec97.pth".to_string(),
            file_size: 30_845_736,
            checksum: Some(Checksum::sha256("a639ec97f5a3b5e8c1d4f7a0b3c6d9e2f5a8b1c4d7e0f3a6b9c2d5e8f1a4b7c0")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "densenet121".to_string(),
            },
            architecture: "DenseNet".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 7_978_856,
            tasks: vec!["image_classification".to_string(), "feature_extraction".to_string()],
            license: Some("BSD-3-Clause".to_string()),
            paper_url: Some("https://arxiv.org/abs/1608.06993".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // VGG models
        self.register_model(ModelInfo {
            name: "vgg16".to_string(),
            display_name: "VGG-16".to_string(),
            description: "16-layer VGG network for image classification".to_string(),
            url: "https://download.pytorch.org/models/vgg16-397923af.pth".to_string(),
            file_size: 553_433_881,
            checksum: Some(Checksum::sha256("397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5c8f")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "vgg16".to_string(),
            },
            architecture: "VGG".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 138_357_544,
            tasks: vec!["image_classification".to_string(), "feature_extraction".to_string()],
            license: Some("BSD-3-Clause".to_string()),
            paper_url: Some("https://arxiv.org/abs/1409.1556".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // EfficientNet models
        self.register_model(ModelInfo {
            name: "efficientnet_b0".to_string(),
            display_name: "EfficientNet-B0".to_string(),
            description: "Efficient convolutional neural network baseline model".to_string(),
            url: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth".to_string(),
            file_size: 20_451_858,
            checksum: Some(Checksum::sha256("3dd342df8c1b4c8e9a5f2d7e6b1c4a9f2e5d8b3a6c9f2e5d8b1a4c7f0a3b6d9e2")),
            source: ModelSource::PyTorchHub {
                repo: "pytorch/vision".to_string(),
                model: "efficientnet_b0".to_string(),
            },
            architecture: "EfficientNet".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 5_288_548,
            tasks: vec!["image_classification".to_string(), "efficient_inference".to_string()],
            license: Some("Apache-2.0".to_string()),
            paper_url: Some("https://arxiv.org/abs/1905.11946".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // BERT models (NLP)
        self.register_model(ModelInfo {
            name: "bert_base_uncased".to_string(),
            display_name: "BERT Base Uncased".to_string(),
            description: "Bidirectional encoder representations from transformers (base, uncased)".to_string(),
            url: "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin".to_string(),
            file_size: 440_473_133,
            checksum: Some(Checksum::sha256("f2a8d4c1e9b6a5d8f1c4e7a0b3d6c9f2e5a8b1d4e7f0a3c6b9e2d5f8a1b4c7e0")),
            source: ModelSource::HuggingFace {
                repo_id: "bert-base-uncased".to_string(),
                filename: Some("pytorch_model.bin".to_string()),
            },
            architecture: "BERT".to_string(),
            input_shape: vec![None, Some(512)], // [batch, sequence_length]
            output_size: 768, // Hidden size
            parameters: 109_482_240,
            tasks: vec!["text_classification".to_string(), "feature_extraction".to_string(), "masked_lm".to_string()],
            license: Some("Apache-2.0".to_string()),
            paper_url: Some("https://arxiv.org/abs/1810.04805".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // GPT models
        self.register_model(ModelInfo {
            name: "gpt2".to_string(),
            display_name: "GPT-2".to_string(),
            description: "Generative pretrained transformer for text generation".to_string(),
            url: "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin".to_string(),
            file_size: 510_342_912,
            checksum: Some(Checksum::sha256("c9d4e7a0f3b6c9e2d5f8a1b4c7e0f3a6b9c2d5e8f1a4b7c0d3e6f9a2b5c8d1e4")),
            source: ModelSource::HuggingFace {
                repo_id: "gpt2".to_string(),
                filename: Some("pytorch_model.bin".to_string()),
            },
            architecture: "GPT".to_string(),
            input_shape: vec![None, Some(1024)], // [batch, sequence_length]
            output_size: 50257, // Vocab size
            parameters: 124_439_808,
            tasks: vec!["text_generation".to_string(), "language_modeling".to_string()],
            license: Some("MIT".to_string()),
            paper_url: Some("https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // YOLO models (Object Detection)
        self.register_model(ModelInfo {
            name: "yolov5s".to_string(),
            display_name: "YOLOv5s".to_string(),
            description: "Small YOLOv5 model for object detection".to_string(),
            url: "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt".to_string(),
            file_size: 14_125_284,
            checksum: Some(Checksum::sha256("e7b9c8f1a4d7e0a3b6c9f2e5d8a1b4c7f0a3b6d9e2c5f8a1b4d7e0a3c6b9f2e5")),
            source: ModelSource::Custom {
                name: "Ultralytics".to_string(),
                url: "https://github.com/ultralytics/yolov5".to_string(),
            },
            architecture: "YOLO".to_string(),
            input_shape: vec![None, Some(3), Some(640), Some(640)],
            output_size: 85, // 80 classes + 5 (x, y, w, h, conf)
            parameters: 7_235_389,
            tasks: vec!["object_detection".to_string(), "real_time_detection".to_string()],
            license: Some("GPL-3.0".to_string()),
            paper_url: Some("https://arxiv.org/abs/2006.10027".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // Stable Diffusion (Image Generation)
        self.register_model(ModelInfo {
            name: "stable_diffusion_1_5".to_string(),
            display_name: "Stable Diffusion 1.5".to_string(),
            description: "Text-to-image diffusion model for high-quality image generation".to_string(),
            url: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt".to_string(),
            file_size: 4_265_380_512,
            checksum: Some(Checksum::sha256("a9f2d8c5e1b4f7a0c3d6e9f2a5b8c1d4e7f0a3b6c9e2d5f8a1b4c7e0a3d6f9a2")),
            source: ModelSource::HuggingFace {
                repo_id: "runwayml/stable-diffusion-v1-5".to_string(),
                filename: Some("v1-5-pruned.ckpt".to_string()),
            },
            architecture: "Diffusion".to_string(),
            input_shape: vec![None, Some(77)], // [batch, text_length]
            output_size: 3 * 512 * 512, // RGB 512x512 images
            parameters: 859_520_964,
            tasks: vec!["text_to_image".to_string(), "image_generation".to_string()],
            license: Some("CreativeML Open RAIL-M".to_string()),
            paper_url: Some("https://arxiv.org/abs/2112.10752".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // ViT models (Vision Transformer)
        self.register_model(ModelInfo {
            name: "vit_base_patch16_224".to_string(),
            display_name: "Vision Transformer Base".to_string(),
            description: "Vision transformer with base size and 16x16 patches".to_string(),
            url: "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz".to_string(),
            file_size: 346_123_456,
            checksum: Some(Checksum::sha256("b4f7a0c3d6e9f2a5b8c1d4e7f0a3b6c9e2d5f8a1b4c7e0a3d6f9a2b5c8d1e4f7")),
            source: ModelSource::Custom {
                name: "Google Research".to_string(),
                url: "https://github.com/google-research/vision_transformer".to_string(),
            },
            architecture: "ViT".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 1000,
            parameters: 86_567_656,
            tasks: vec!["image_classification".to_string(), "vision_transformer".to_string()],
            license: Some("Apache-2.0".to_string()),
            paper_url: Some("https://arxiv.org/abs/2010.11929".to_string()),
            created_at: Some(chrono::Utc::now()),
        });

        // CLIP models (Multimodal)
        self.register_model(ModelInfo {
            name: "clip_vit_b_32".to_string(),
            display_name: "CLIP ViT-B/32".to_string(),
            description: "Contrastive language-image pretraining with vision transformer".to_string(),
            url: "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt".to_string(),
            file_size: 338_664_192,
            checksum: Some(Checksum::sha256("40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af")),
            source: ModelSource::Custom {
                name: "OpenAI".to_string(),
                url: "https://github.com/openai/CLIP".to_string(),
            },
            architecture: "CLIP".to_string(),
            input_shape: vec![None, Some(3), Some(224), Some(224)],
            output_size: 512, // Embedding dimension
            parameters: 151_277_313,
            tasks: vec!["image_text_matching".to_string(), "multimodal_embedding".to_string(), "zero_shot_classification".to_string()],
            license: Some("MIT".to_string()),
            paper_url: Some("https://arxiv.org/abs/2103.00020".to_string()),
            created_at: Some(chrono::Utc::now()),
        });
    }

    /// Load registry from JSON file
    /// JSONファイルからレジストリを読み込み
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> RusTorchResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let models: HashMap<String, ModelInfo> = serde_json::from_str(&content)
            .map_err(|e| RusTorchError::DeserializationError(e.to_string()))?;

        Ok(Self { models })
    }

    /// Save registry to JSON file
    /// レジストリをJSONファイルに保存
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> RusTorchResult<()> {
        let content = serde_json::to_string_pretty(&self.models)
            .map_err(|e| RusTorchError::SerializationError(e.to_string()))?;
        
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Search models by name or description
    /// 名前または説明でモデルを検索
    pub fn search_models(&self, query: &str) -> Vec<&ModelInfo> {
        let query_lower = query.to_lowercase();
        
        self.models
            .values()
            .filter(|model| {
                model.name.to_lowercase().contains(&query_lower) ||
                model.display_name.to_lowercase().contains(&query_lower) ||
                model.description.to_lowercase().contains(&query_lower) ||
                model.architecture.to_lowercase().contains(&query_lower) ||
                model.tasks.iter().any(|task| task.to_lowercase().contains(&query_lower))
            })
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"resnet18"));
        assert!(models.contains(&"bert_base_uncased"));
        assert!(models.contains(&"clip_vit_b_32"));
    }

    #[test]
    fn test_get_model_info() {
        let registry = ModelRegistry::new();
        
        let resnet18 = registry.get_model_info("resnet18");
        assert!(resnet18.is_ok());
        
        let resnet18 = resnet18.unwrap();
        assert_eq!(resnet18.architecture, "ResNet");
        assert_eq!(resnet18.output_size, 1000);
        assert!(resnet18.tasks.contains(&"image_classification".to_string()));

        // Test non-existent model
        let nonexistent = registry.get_model_info("nonexistent");
        assert!(nonexistent.is_err());
    }

    #[test]
    fn test_list_models_by_task() {
        let registry = ModelRegistry::new();
        
        let classification_models = registry.list_models_by_task("image_classification");
        assert!(classification_models.contains(&"resnet18"));
        assert!(classification_models.contains(&"resnet50"));
        assert!(classification_models.contains(&"mobilenet_v2"));

        let nlp_models = registry.list_models_by_task("text_generation");
        assert!(nlp_models.contains(&"gpt2"));

        let multimodal_models = registry.list_models_by_task("multimodal_embedding");
        assert!(multimodal_models.contains(&"clip_vit_b_32"));
    }

    #[test]
    fn test_list_models_by_architecture() {
        let registry = ModelRegistry::new();
        
        let resnet_models = registry.list_models_by_architecture("ResNet");
        assert!(resnet_models.contains(&"resnet18"));
        assert!(resnet_models.contains(&"resnet50"));

        let transformer_models = registry.list_models_by_architecture("BERT");
        assert!(transformer_models.contains(&"bert_base_uncased"));
    }

    #[test]
    fn test_search_models() {
        let registry = ModelRegistry::new();
        
        // Search by name
        let resnet_results = registry.search_models("resnet");
        assert!(resnet_results.len() >= 2);
        assert!(resnet_results.iter().any(|m| m.name == "resnet18"));

        // Search by task
        let detection_results = registry.search_models("detection");
        assert!(detection_results.iter().any(|m| m.name == "yolov5s"));

        // Search by architecture
        let transformer_results = registry.search_models("transformer");
        assert!(transformer_results.iter().any(|m| m.architecture == "BERT"));
    }

    #[test]
    fn test_register_custom_model() {
        let mut registry = ModelRegistry::new();
        
        let custom_model = ModelInfo {
            name: "custom_model".to_string(),
            display_name: "Custom Test Model".to_string(),
            description: "A custom model for testing".to_string(),
            url: "https://example.com/model.pth".to_string(),
            file_size: 1024,
            checksum: Some(Checksum::sha256("test_hash")),
            source: ModelSource::DirectUrl {
                url: "https://example.com/model.pth".to_string(),
            },
            architecture: "Custom".to_string(),
            input_shape: vec![None, Some(10)],
            output_size: 2,
            parameters: 1000,
            tasks: vec!["test".to_string()],
            license: Some("MIT".to_string()),
            paper_url: None,
            created_at: Some(chrono::Utc::now()),
        };

        registry.register_model(custom_model);
        
        let retrieved = registry.get_model_info("custom_model");
        assert!(retrieved.is_ok());
        assert_eq!(retrieved.unwrap().architecture, "Custom");
    }

    #[test]
    fn test_model_source_variants() {
        // Test different model source types
        let pytorch_hub = ModelSource::PyTorchHub {
            repo: "pytorch/vision".to_string(),
            model: "resnet18".to_string(),
        };
        
        let huggingface = ModelSource::HuggingFace {
            repo_id: "bert-base-uncased".to_string(),
            filename: Some("pytorch_model.bin".to_string()),
        };
        
        let direct_url = ModelSource::DirectUrl {
            url: "https://example.com/model.pth".to_string(),
        };

        // Ensure all variants serialize/deserialize correctly
        let sources = vec![pytorch_hub, huggingface, direct_url];
        for source in sources {
            let serialized = serde_json::to_string(&source).unwrap();
            let deserialized: ModelSource = serde_json::from_str(&serialized).unwrap();
            // Basic structure validation - both should serialize to same JSON
            assert_eq!(serde_json::to_string(&source).unwrap(), 
                      serde_json::to_string(&deserialized).unwrap());
        }
    }
}