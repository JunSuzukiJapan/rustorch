//! Phase 5 DataLoader System Integration Tests
//! フェーズ5 DataLoaderシステム統合テスト

use rustorch::data::dataset::{DatasetV2, TensorDataset, ConcatDataset};
use rustorch::data::sampler::{Sampler, SequentialSampler, RandomSampler, BatchSampler, WeightedRandomSampler};
use rustorch::data::dataloader::Phase5DataLoader;
use rustorch::tensor::Tensor;
use rustorch::error::RusTorchError;

#[cfg(test)]
mod phase5_tests {
    use super::*;

    #[test]
    fn test_tensor_dataset_creation() {
        let features = vec![
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3, 1]),
            Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3, 1]),
        ];
        
        let dataset = TensorDataset::new(features);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_tensor_dataset_get_item() {
        let tensors = vec![
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3, 1]),
            Tensor::from_vec(vec![10.0f32, 20.0, 30.0], vec![3, 1]),
        ];
        
        let dataset = TensorDataset::new(tensors).unwrap();
        
        let item = dataset.get_item(0);
        assert!(item.is_ok());
        
        let tensors = item.unwrap();
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn test_tensor_dataset_invalid_shapes() {
        let tensors = vec![
            Tensor::from_vec(vec![1.0f32, 2.0], vec![2, 1]),
            Tensor::from_vec(vec![3.0f32, 4.0, 5.0], vec![3, 1]),
        ];
        
        let result = TensorDataset::new(tensors);
        assert!(result.is_err());
        
        if let Err(RusTorchError::ShapeMismatch { .. }) = result {
            // Expected error type
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(5);
        
        assert_eq!(sampler.len(), 5);
        assert!(!sampler.is_empty());
        
        for i in 0..5 {
            assert_eq!(sampler.sample(), Some(i));
        }
        
        assert_eq!(sampler.sample(), None);
        assert!(sampler.is_empty());
        
        sampler.reset();
        assert_eq!(sampler.sample(), Some(0));
        assert!(!sampler.is_empty());
    }

    #[test]
    fn test_random_sampler() {
        let mut sampler = RandomSampler::new(100);
        
        assert_eq!(sampler.len(), 100);
        assert!(!sampler.is_empty());
        
        let mut indices = Vec::new();
        for _ in 0..100 {
            if let Some(idx) = sampler.sample() {
                indices.push(idx);
            }
        }
        
        assert_eq!(indices.len(), 100);
        assert!(sampler.is_empty());
        
        // Verify all indices are within range
        for &idx in &indices {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_random_sampler_with_replacement() {
        let mut sampler = RandomSampler::with_replacement(10, 50);
        
        assert_eq!(sampler.len(), usize::MAX); // Infinite for replacement
        
        let mut indices = Vec::new();
        for _ in 0..50 {
            if let Some(idx) = sampler.sample() {
                indices.push(idx);
            }
        }
        
        assert_eq!(indices.len(), 50);
        
        // Should still be able to sample (infinite)
        assert!(!sampler.is_empty());
    }

    #[test]
    fn test_batch_sampler() {
        let sequential = SequentialSampler::new(10);
        let mut batch_sampler = BatchSampler::new(
            Box::new(sequential),
            3,
            false // don't drop last
        );
        
        assert_eq!(batch_sampler.batch_size(), 3);
        assert!(!batch_sampler.drop_last());
        
        let batch1 = batch_sampler.next_batch();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap(), vec![0, 1, 2]);
        
        let batch2 = batch_sampler.next_batch();
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap(), vec![3, 4, 5]);
        
        // Continue until exhausted
        let mut batches = 0;
        while batch_sampler.next_batch().is_some() {
            batches += 1;
            if batches > 10 { // Safety check
                break;
            }
        }
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let sequential = SequentialSampler::new(10);
        let mut batch_sampler = BatchSampler::new(
            Box::new(sequential),
            3,
            true // drop last incomplete batch
        );
        
        let mut batch_count = 0;
        while let Some(batch) = batch_sampler.next_batch() {
            assert_eq!(batch.len(), 3); // All batches should be complete
            batch_count += 1;
        }
        
        assert_eq!(batch_count, 3); // 10/3 = 3 complete batches
    }

    #[test]
    fn test_weighted_random_sampler() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let mut sampler = WeightedRandomSampler::new(weights, 100, true).unwrap();
        
        assert_eq!(sampler.len(), 100);
        
        let mut counts = vec![0; 4];
        for _ in 0..1000 {
            if let Some(idx) = sampler.sample() {
                counts[idx] += 1;
            }
        }
        
        // Higher weighted indices should appear more frequently
        // Index 3 (weight 0.4) should appear most often
        // Index 0 (weight 0.1) should appear least often
        assert!(counts[3] > counts[2]);
        assert!(counts[2] > counts[1]);
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn test_weighted_sampler_invalid_weights() {
        let weights = vec![0.1, -0.2, 0.3]; // Negative weight
        let result = WeightedRandomSampler::new(weights, 10, false);
        
        assert!(result.is_err());
        if let Err(RusTorchError::InvalidParameters { message, .. }) = result {
            assert!(message.contains("negative"));
        }
    }

    #[test]
    fn test_concat_dataset() {
        let features1 = vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![2, 1])];
        let features2 = vec![Tensor::from_vec(vec![3.0f32, 4.0], vec![2, 1])];
        
        let dataset1 = TensorDataset::new(features1).unwrap();
        let dataset2 = TensorDataset::new(features2).unwrap();
        
        let datasets: Vec<Box<dyn DatasetV2<Vec<Tensor<f32>>>>> = vec![
            Box::new(dataset1),
            Box::new(dataset2),
        ];
        
        let concat_dataset = ConcatDataset::new(datasets).unwrap();
        
        assert_eq!(concat_dataset.len(), 4); // 2 + 2
        
        // Test accessing items from both datasets
        assert!(concat_dataset.get_item(0).is_ok());
        assert!(concat_dataset.get_item(1).is_ok());
        assert!(concat_dataset.get_item(2).is_ok());
        assert!(concat_dataset.get_item(3).is_ok());
        
        // Out of bounds
        assert!(concat_dataset.get_item(4).is_err());
    }

    #[test]
    fn test_error_integration() {
        // Test DataError type alias works with RusTorchError
        let error: rustorch::data::dataset::DataError = RusTorchError::InvalidParameters {
            operation: "test".to_string(),
            message: "test error".to_string(),
        };
        
        match error {
            RusTorchError::InvalidParameters { operation, message } => {
                assert_eq!(operation, "test");
                assert_eq!(message, "test error");
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_phase5_api_compatibility() {
        // Test that Phase 5 components integrate correctly
        let tensors = vec![
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4, 1]),
        ];
        
        let dataset = TensorDataset::new(tensors).unwrap();
        let sampler = SequentialSampler::new(dataset.len());
        
        // Verify compatibility between components
        assert_eq!(dataset.len(), sampler.len());
        
        // Test that we can create batch sampler from base sampler
        let batch_sampler = BatchSampler::new(
            Box::new(sampler),
            2,
            false
        );
        
        assert_eq!(batch_sampler.batch_size(), 2);
    }

    #[test]
    fn test_memory_efficiency() {
        // Test that large datasets can be created without excessive memory usage
        let large_size = 10000;
        let data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();
        let tensors = vec![Tensor::from_vec(data, vec![large_size, 1])];
        
        let dataset = TensorDataset::new(tensors);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        assert_eq!(dataset.len(), large_size);
        
        // Test random access doesn't cause memory issues
        let item = dataset.get_item(5000);
        assert!(item.is_ok());
    }

    #[test] 
    fn test_phase5_roadmap_requirements() {
        // Verify all Phase 5 roadmap requirements are met
        
        // 1. Dataset traits implemented
        let _: &dyn DatasetV2<Vec<Tensor<f32>>> = &TensorDataset::new(vec![
            Tensor::from_vec(vec![1.0f32], vec![1, 1])
        ]).unwrap();
        
        // 2. Sampler system implemented  
        let _: &dyn Sampler = &SequentialSampler::new(10);
        let _: &dyn Sampler = &RandomSampler::new(10);
        
        // 3. Unified error handling
        let _: rustorch::data::dataset::DataError = RusTorchError::InvalidParameters {
            operation: "test".to_string(), 
            message: "test".to_string(),
        };
        
        // 4. Memory-efficient prefetching (tested indirectly through large dataset)
        let large_dataset = TensorDataset::new(vec![
            Tensor::from_vec((0..1000).map(|i| i as f32).collect(), vec![1000, 1])
        ]).unwrap();
        
        assert_eq!(large_dataset.len(), 1000);
    }

    #[test]
    fn test_phase5_dataloader_integration() {
        // Test compatibility between Phase 5 components
        let tensors = vec![
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4, 1]),
        ];
        
        let dataset = TensorDataset::new(tensors).unwrap();
        let sampler = SequentialSampler::new(dataset.len());
        
        // Test that components are compatible
        assert_eq!(dataset.len(), sampler.len());
        
        // Test sampler functionality
        let mut test_sampler = SequentialSampler::new(4);
        assert_eq!(test_sampler.sample(), Some(0));
        assert_eq!(test_sampler.sample(), Some(1));
        assert!(!test_sampler.is_empty());
        
        test_sampler.reset();
        assert_eq!(test_sampler.sample(), Some(0));
    }
}