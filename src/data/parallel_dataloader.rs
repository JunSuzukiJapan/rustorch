//! Parallel DataLoader implementation for high-performance batch processing
//! 高性能バッチ処理のための並列DataLoader実装

use crate::data::LegacyDataset;
use crate::tensor::parallel_traits::MatrixParallelOp;
use crate::tensor::Tensor;
use num_traits::Float;
use rayon::prelude::*;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Parallel DataLoader for concurrent batch processing
/// 並行バッチ処理のための並列DataLoader
#[deprecated(since = "0.6.0", note = "Use Phase 5 DataLoader with built-in parallelization")]
pub struct ParallelDataLoader<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    D: LegacyDataset<T> + Send + Sync,
> {
    dataset: Arc<D>,
    batch_size: usize,
    num_workers: usize,
    shuffle: bool,
    prefetch_factor: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > ParallelDataLoader<T, D>
{
    /// Creates a new parallel DataLoader
    /// 新しい並列DataLoaderを作成
    pub fn new(
        dataset: D,
        batch_size: usize,
        num_workers: usize,
        shuffle: bool,
        prefetch_factor: usize,
    ) -> Self {
        ParallelDataLoader {
            dataset: Arc::new(dataset),
            batch_size,
            num_workers: num_workers.max(1),
            shuffle,
            prefetch_factor: prefetch_factor.max(1),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a parallel batch iterator
    /// 並列バッチイテレータを作成
    pub fn iter(&self) -> ParallelBatchIterator<T, D> {
        ParallelBatchIterator::new(
            self.dataset.clone(),
            self.batch_size,
            self.num_workers,
            self.shuffle,
            self.prefetch_factor,
        )
    }

    /// Process batches in parallel with a custom function
    /// カスタム関数で並列バッチ処理
    pub fn process_parallel<F, R>(&self, processor: F) -> Vec<R>
    where
        F: Fn((Tensor<T>, Tensor<T>)) -> R + Send + Sync,
        R: Send + 'static,
    {
        let batches: Vec<_> = self.iter().collect();
        batches.into_par_iter().map(processor).collect()
    }

    /// Parallel batch preprocessing
    /// 並列バッチ前処理
    pub fn preprocess_parallel<F>(&self, preprocessor: F) -> Vec<(Tensor<T>, Tensor<T>)>
    where
        F: Fn((Tensor<T>, Tensor<T>)) -> (Tensor<T>, Tensor<T>) + Send + Sync,
    {
        let batches: Vec<_> = self.iter().collect();

        batches.into_par_iter().map(preprocessor).collect()
    }
}

/// Parallel batch iterator with prefetching
/// プリフェッチ機能付き並列バッチイテレータ
#[deprecated(since = "0.6.0", note = "Use Phase 5 DataLoader with built-in parallelization")]
pub struct ParallelBatchIterator<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    D: LegacyDataset<T> + Send + Sync,
> {
    receiver: mpsc::Receiver<Option<(Tensor<T>, Tensor<T>)>>,
    _handles: Vec<thread::JoinHandle<()>>,
    _phantom: std::marker::PhantomData<D>,
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > ParallelBatchIterator<T, D>
{
    fn new(
        dataset: Arc<D>,
        batch_size: usize,
        num_workers: usize,
        shuffle: bool,
        _prefetch_factor: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        let mut handles = Vec::new();

        // Create indices for batching
        let total_samples = dataset.len();
        let _num_batches = total_samples.div_ceil(batch_size);

        let indices = Arc::new(Mutex::new({
            let mut idx: Vec<usize> = (0..total_samples).collect();
            if shuffle {
                use rand::seq::SliceRandom;
                idx.shuffle(&mut rand::thread_rng());
            }
            idx
        }));

        // Spawn worker threads
        for _worker_id in 0..num_workers {
            let dataset = dataset.clone();
            let indices = indices.clone();
            let sender = sender.clone();

            let handle = thread::spawn(move || {
                let mut batch_count = 0;

                loop {
                    // Get next batch indices
                    let batch_indices = {
                        let indices_guard = indices.lock().unwrap();
                        let start_idx = batch_count * batch_size;
                        let end_idx = std::cmp::min(start_idx + batch_size, indices_guard.len());

                        if start_idx >= indices_guard.len() {
                            break;
                        }

                        let batch_indices: Vec<usize> = indices_guard[start_idx..end_idx].to_vec();
                        batch_count += num_workers; // Each worker processes every num_workers-th batch
                        batch_indices
                    };

                    if batch_indices.is_empty() {
                        break;
                    }

                    // Load batch data in parallel
                    let batch_data: Vec<_> = batch_indices
                        .into_par_iter()
                        .filter_map(|idx| dataset.get(idx))
                        .collect();

                    if !batch_data.is_empty() {
                        // Stack tensors to create batch
                        let features: Vec<&Tensor<T>> = batch_data.iter().map(|(f, _)| f).collect();
                        let targets: Vec<&Tensor<T>> = batch_data.iter().map(|(_, t)| t).collect();

                        if let (Ok(batch_features), Ok(batch_targets)) =
                            (Tensor::stack(&features), Tensor::stack(&targets))
                        {
                            if sender.send(Some((batch_features, batch_targets))).is_err() {
                                break;
                            }
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Drop the original sender so the receiver knows when all workers are done
        drop(sender);

        ParallelBatchIterator {
            receiver,
            _handles: handles,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync,
    > Iterator for ParallelBatchIterator<T, D>
{
    type Item = (Tensor<T>, Tensor<T>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(Some(batch)) => Some(batch),
            Ok(None) | Err(_) => None,
        }
    }
}

/// Parallel batch operations for training
/// 訓練用並列バッチ演算
#[deprecated(since = "0.6.0", note = "Use Phase 5 DataLoader with built-in parallelization")]
pub struct ParallelBatchProcessor<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    _phantom: std::marker::PhantomData<T>,
}

impl<
        T: Float + Send + Sync + Clone + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    > ParallelBatchProcessor<T>
{
    /// Parallel gradient computation across batches
    /// バッチ間並列勾配計算
    pub fn compute_gradients_parallel(
        batches: &[(Tensor<T>, Tensor<T>)],
        model_fn: impl Fn(&Tensor<T>) -> Tensor<T> + Send + Sync,
        loss_fn: impl Fn(&Tensor<T>, &Tensor<T>) -> T + Send + Sync,
    ) -> Vec<T> {
        batches
            .par_iter()
            .map(|(features, targets)| {
                let predictions = model_fn(features);
                loss_fn(&predictions, targets)
            })
            .collect()
    }

    /// Parallel data augmentation
    /// 並列データ拡張
    pub fn augment_parallel(
        batches: &[(Tensor<T>, Tensor<T>)],
        augment_fn: impl Fn(&Tensor<T>) -> Tensor<T> + Send + Sync,
    ) -> Vec<(Tensor<T>, Tensor<T>)> {
        batches
            .par_iter()
            .map(|(features, targets)| {
                let augmented_features = augment_fn(features);
                (augmented_features, targets.clone())
            })
            .collect()
    }

    /// Parallel batch normalization statistics
    /// 並列バッチ正規化統計
    pub fn compute_batch_stats_parallel(batches: &[(Tensor<T>, Tensor<T>)]) -> (T, T) {
        let stats: Vec<_> = batches
            .par_iter()
            .map(|(features, _)| {
                if let Some(slice) = features.as_slice() {
                    let sum = slice.iter().fold(T::zero(), |acc, &x| acc + x);
                    let sum_sq = slice.iter().fold(T::zero(), |acc, &x| acc + x * x);
                    let count = T::from(slice.len()).unwrap();
                    (sum, sum_sq, count)
                } else {
                    (T::zero(), T::zero(), T::zero())
                }
            })
            .collect();

        let (total_sum, total_sum_sq, total_count) = stats.iter().fold(
            (T::zero(), T::zero(), T::zero()),
            |(s, sq, c), &(s_i, sq_i, c_i)| (s + s_i, sq + sq_i, c + c_i),
        );

        if total_count > T::zero() {
            let mean = total_sum / total_count;
            let variance = (total_sum_sq / total_count) - mean * mean;
            (mean, variance)
        } else {
            (T::zero(), T::zero())
        }
    }
}

/// Specialized f32 parallel operations with SIMD integration
/// SIMD統合を含むf32特殊化並列演算
impl ParallelBatchProcessor<f32> {
    /// High-performance parallel batch matrix operations
    /// 高性能並列バッチ行列演算
    pub fn simd_batch_operations(
        batches: &[(Tensor<f32>, Tensor<f32>)],
        weights: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        batches
            .par_iter()
            .map(|(features, _)| {
                // Use SIMD-optimized matrix multiplication
                if let Ok(result) = features.batch_simd_matmul_parallel(weights) {
                    result
                } else {
                    features.clone()
                }
            })
            .collect()
    }

    /// Parallel SIMD convolution operations
    /// 並列SIMD畳み込み演算
    pub fn simd_batch_conv2d(
        batches: &[(Tensor<f32>, Tensor<f32>)],
        kernel: &Tensor<f32>,
        stride: usize,
        padding: usize,
    ) -> Vec<Tensor<f32>> {
        batches
            .par_iter()
            .map(|(features, _)| {
                if let Ok(result) = features.batch_conv2d(kernel, stride, padding) {
                    result
                } else {
                    features.clone()
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;

    #[test]
    fn test_parallel_dataloader() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
            Tensor::from_vec(vec![7.0, 8.0], vec![2]),
            Tensor::from_vec(vec![9.0, 10.0], vec![2]),
            Tensor::from_vec(vec![11.0, 12.0], vec![2]),
        ];

        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];

        let dataset = TensorDataset::from_features_targets(features, targets).unwrap();
        let parallel_loader = ParallelDataLoader::new(dataset, 2, 2, false, 2);

        let batches: Vec<_> = parallel_loader.iter().collect();
        assert!(!batches.is_empty());

        // Test parallel processing
        let results = parallel_loader
            .process_parallel(|(features, targets)| (features.size(), targets.size()));

        assert!(!results.is_empty());
    }

    #[test]
    fn test_parallel_batch_processor() {
        let batches = vec![
            (
                Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                Tensor::<f32>::from_vec(vec![0.0, 1.0], vec![2]),
            ),
            (
                Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
                Tensor::<f32>::from_vec(vec![1.0, 0.0], vec![2]),
            ),
        ];

        let losses = ParallelBatchProcessor::compute_gradients_parallel(
            &batches,
            |features| features.clone(),
            |pred, target| {
                // Simple MSE loss
                if let (Some(p_slice), Some(t_slice)) = (pred.as_slice(), target.as_slice()) {
                    p_slice
                        .iter()
                        .zip(t_slice.iter())
                        .map(|(p, t)| (p - t) * (p - t))
                        .sum::<f32>()
                        / p_slice.len() as f32
                } else {
                    0.0
                }
            },
        );

        assert_eq!(losses.len(), 2);
    }

    #[test]
    fn test_batch_stats_parallel() {
        let batches = vec![
            (
                Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                Tensor::<f32>::from_vec(vec![0.0], vec![1]),
            ),
            (
                Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
                Tensor::<f32>::from_vec(vec![1.0], vec![1]),
            ),
        ];

        let (mean, variance) = ParallelBatchProcessor::compute_batch_stats_parallel(&batches);

        // Mean of [1,2,3,4,5,6,7,8] = 4.5
        assert!((mean - 4.5).abs() < 1e-6);

        // Variance should be > 0
        assert!(variance > 0.0);
    }

    #[test]
    fn test_simd_batch_operations() {
        let batches = vec![(
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::<f32>::from_vec(vec![0.0], vec![1]),
        )];

        let weights = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let results = ParallelBatchProcessor::simd_batch_operations(&batches, &weights);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_large_parallel_processing() {
        let batch_count = 100;
        let batch_size = 50;
        let feature_size = 100;

        let mut batches = Vec::new();
        for i in 0..batch_count {
            let features = Tensor::<f32>::from_vec(
                (0..batch_size * feature_size)
                    .map(|j| (i * 1000 + j) as f32)
                    .collect(),
                vec![batch_size, feature_size],
            );
            let targets = Tensor::<f32>::from_vec(vec![i as f32; batch_size], vec![batch_size]);
            batches.push((features, targets));
        }

        let start = std::time::Instant::now();
        let results = ParallelBatchProcessor::compute_gradients_parallel(
            &batches,
            |features| features.clone(),
            |_, _| 1.0f32,
        );
        let duration = start.elapsed();

        assert_eq!(results.len(), batch_count);
        println!("Processed {} batches in {:?}", batch_count, duration);
    }
}
