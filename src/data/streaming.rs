//! Streaming data loading for large datasets
//! 大規模データセット用ストリーミングデータローディング
//!
//! This module provides memory-efficient data loading for datasets that don't fit in memory
//! このモジュールはメモリに収まらないデータセットのためのメモリ効率的なデータ読み込みを提供します

#![allow(deprecated)] // Allow deprecated APIs for backward compatibility

use super::LegacyDataset;
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Streaming dataset that loads data on-demand
/// オンデマンドでデータを読み込むストリーミングデータセット
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 IterableDataset for streaming data"
)]
pub struct StreamingDataset<T: Float + Send + Sync + 'static, D: LegacyDataset<T> + Send + Sync> {
    inner_dataset: Arc<D>,
    buffer_size: usize,
    prefetch_threads: usize,
    buffer: Arc<Mutex<VecDeque<(usize, (Tensor<T>, Tensor<T>))>>>,
    loaded_indices: Arc<Mutex<std::collections::HashSet<usize>>>,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > StreamingDataset<T, D>
{
    /// Create a new streaming dataset
    /// 新しいストリーミングデータセットを作成
    pub fn new(dataset: D, buffer_size: usize, prefetch_threads: usize) -> Self {
        let inner_dataset = Arc::new(dataset);

        StreamingDataset {
            inner_dataset,
            buffer_size: buffer_size.max(1),
            prefetch_threads: prefetch_threads.max(1).min(8),
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            loaded_indices: Arc::new(Mutex::new(std::collections::HashSet::new())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Start background prefetching for given indices
    /// 指定されたインデックスについて背景プリフェッチを開始
    pub fn start_prefetch(&self, indices: Vec<usize>) {
        let dataset = self.inner_dataset.clone();
        let buffer = self.buffer.clone();
        let loaded_indices = self.loaded_indices.clone();
        let buffer_size = self.buffer_size;
        let num_threads = self.prefetch_threads;

        let indices = Arc::new(Mutex::new(indices.into_iter().collect::<VecDeque<_>>()));

        // Spawn prefetch workers
        for _worker_id in 0..num_threads {
            let dataset = dataset.clone();
            let buffer = buffer.clone();
            let loaded_indices = loaded_indices.clone();
            let indices = indices.clone();

            thread::spawn(move || {
                loop {
                    let next_index = {
                        let mut indices_guard = indices.lock().unwrap();
                        indices_guard.pop_front()
                    };

                    match next_index {
                        Some(index) => {
                            // Check if already loaded or buffer is full
                            {
                                let loaded_guard = loaded_indices.lock().unwrap();
                                let buffer_guard = buffer.lock().unwrap();

                                if loaded_guard.contains(&index)
                                    || buffer_guard.len() >= buffer_size
                                {
                                    thread::sleep(Duration::from_millis(10));
                                    continue;
                                }
                            }

                            // Load data
                            if let Some(data) = dataset.get(index) {
                                let mut buffer_guard = buffer.lock().unwrap();
                                let mut loaded_guard = loaded_indices.lock().unwrap();

                                if buffer_guard.len() < buffer_size
                                    && !loaded_guard.contains(&index)
                                {
                                    buffer_guard.push_back((index, data));
                                    loaded_guard.insert(index);
                                }
                            }
                        }
                        None => break, // No more indices to process
                    }

                    thread::sleep(Duration::from_millis(1));
                }
            });
        }
    }

    /// Get data from buffer or load directly
    /// バッファからデータを取得するか直接読み込み
    pub fn get_buffered(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        // First try to get from buffer
        {
            let mut buffer_guard = self.buffer.lock().unwrap();
            if let Some(pos) = buffer_guard.iter().position(|(idx, _)| *idx == index) {
                if let Some((_, data)) = buffer_guard.remove(pos) {
                    return Some(data);
                }
            }
        }

        // If not in buffer, load directly
        if let Some(data) = self.inner_dataset.get(index) {
            // Mark as loaded
            {
                let mut loaded_guard = self.loaded_indices.lock().unwrap();
                loaded_guard.insert(index);
            }
            Some(data)
        } else {
            None
        }
    }

    /// Clear the buffer and loaded indices
    /// バッファと読み込み済みインデックスをクリア
    pub fn clear_buffer(&self) {
        {
            let mut buffer_guard = self.buffer.lock().unwrap();
            buffer_guard.clear();
        }
        {
            let mut loaded_guard = self.loaded_indices.lock().unwrap();
            loaded_guard.clear();
        }
    }

    /// Get buffer statistics
    /// バッファ統計を取得
    pub fn buffer_stats(&self) -> (usize, usize) {
        let buffer_guard = self.buffer.lock().unwrap();
        let loaded_guard = self.loaded_indices.lock().unwrap();
        (buffer_guard.len(), loaded_guard.len())
    }
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > LegacyDataset<T> for StreamingDataset<T, D>
{
    fn len(&self) -> usize {
        self.inner_dataset.len()
    }

    fn get(&self, index: usize) -> Option<(Tensor<T>, Tensor<T>)> {
        self.get_buffered(index)
    }
}

/// Dynamic batch size dataloader that adjusts based on memory usage
/// メモリ使用量に基づいて調整する動的バッチサイズデータローダー
#[deprecated(since = "0.6.0", note = "Use Phase 5 DataLoader with dynamic sampling")]
pub struct DynamicBatchLoader<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    D: LegacyDataset<T>,
> {
    dataset: Arc<D>,
    min_batch_size: usize,
    max_batch_size: usize,
    current_batch_size: usize,
    memory_threshold: usize, // in bytes
    indices: Vec<usize>,
    current_index: usize,
    shuffle: bool,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > DynamicBatchLoader<T, D>
{
    /// Create a new dynamic batch loader
    /// 新しい動的バッチローダーを作成
    pub fn new(
        dataset: D,
        min_batch_size: usize,
        max_batch_size: usize,
        memory_threshold: usize,
        shuffle: bool,
    ) -> Self {
        let dataset_arc = Arc::new(dataset);
        let mut indices: Vec<usize> = (0..dataset_arc.len()).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        DynamicBatchLoader {
            dataset: dataset_arc,
            min_batch_size,
            max_batch_size: max_batch_size.max(min_batch_size),
            current_batch_size: min_batch_size,
            memory_threshold,
            indices,
            current_index: 0,
            shuffle,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Estimate memory usage of a tensor
    /// テンソルのメモリ使用量を推定
    fn estimate_memory_usage(&self, tensor: &Tensor<T>) -> usize {
        tensor.numel() * std::mem::size_of::<T>()
    }

    /// Adjust batch size based on memory usage
    /// メモリ使用量に基づいてバッチサイズを調整
    fn adjust_batch_size(&mut self, memory_used: usize) {
        if memory_used > self.memory_threshold {
            // Reduce batch size
            self.current_batch_size = (self.current_batch_size as f64 * 0.8) as usize;
            self.current_batch_size = self.current_batch_size.max(self.min_batch_size);
        } else if memory_used < self.memory_threshold / 2 {
            // Increase batch size
            self.current_batch_size = (self.current_batch_size as f64 * 1.2) as usize;
            self.current_batch_size = self.current_batch_size.min(self.max_batch_size);
        }
    }

    /// Get next dynamic batch
    /// 次の動的バッチを取得
    pub fn next_batch(&mut self) -> Option<(Tensor<T>, Tensor<T>)> {
        if self.current_index >= self.dataset.len() {
            return None;
        }

        let end_index = std::cmp::min(
            self.current_index + self.current_batch_size,
            self.dataset.len(),
        );

        let mut batch_features = Vec::new();
        let mut batch_targets = Vec::new();
        let mut total_memory = 0;

        for i in self.current_index..end_index {
            let index = self.indices[i];
            if let Some((feature, target)) = self.dataset.get(index) {
                total_memory += self.estimate_memory_usage(&feature);
                total_memory += self.estimate_memory_usage(&target);

                batch_features.push(feature);
                batch_targets.push(target);
            }
        }

        self.current_index = end_index;

        // Adjust batch size for next iteration
        self.adjust_batch_size(total_memory);

        if !batch_features.is_empty() {
            let feature_refs: Vec<&Tensor<T>> = batch_features.iter().collect();
            let target_refs: Vec<&Tensor<T>> = batch_targets.iter().collect();

            match (Tensor::stack(&feature_refs), Tensor::stack(&target_refs)) {
                (Ok(stacked_features), Ok(stacked_targets)) => {
                    Some((stacked_features, stacked_targets))
                }
                _ => Some((batch_features[0].clone(), batch_targets[0].clone())),
            }
        } else {
            None
        }
    }

    /// Reset the loader for a new epoch
    /// 新しいエポック用にローダーをリセット
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }

    /// Get current batch size
    /// 現在のバッチサイズを取得
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Get memory threshold
    /// メモリ閾値を取得
    pub fn memory_threshold(&self) -> usize {
        self.memory_threshold
    }

    /// Set memory threshold
    /// メモリ閾値を設定
    pub fn set_memory_threshold(&mut self, threshold: usize) {
        self.memory_threshold = threshold;
    }
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > Iterator for DynamicBatchLoader<T, D>
{
    type Item = (Tensor<T>, Tensor<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Async data loader using channels for non-blocking data loading
/// ノンブロッキングデータ読み込み用チャネルを使用する非同期データローダー
#[deprecated(
    since = "0.6.0",
    note = "Use Phase 5 IterableDataset for async data loading"
)]
pub struct AsyncDataLoader<
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    D: LegacyDataset<T> + Send + Sync + 'static,
> {
    receiver: mpsc::Receiver<Option<(Tensor<T>, Tensor<T>)>>,
    _handles: Vec<thread::JoinHandle<()>>,
    _phantom: std::marker::PhantomData<D>,
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > AsyncDataLoader<T, D>
{
    /// Create a new async data loader
    /// 新しい非同期データローダーを作成
    pub fn new(
        dataset: D,
        batch_size: usize,
        buffer_size: usize,
        num_workers: usize,
        shuffle: bool,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        let dataset = Arc::new(dataset);
        let mut handles = Vec::new();

        // Create shuffled indices
        let indices = Arc::new(Mutex::new({
            let mut idx: Vec<usize> = (0..dataset.len()).collect();
            if shuffle {
                use rand::seq::SliceRandom;
                idx.shuffle(&mut rand::thread_rng());
            }
            idx.into_iter().collect::<VecDeque<_>>()
        }));

        let batch_counter = Arc::new(Mutex::new(0usize));

        // Spawn worker threads
        for _worker_id in 0..num_workers.max(1).min(8) {
            let dataset = dataset.clone();
            let indices = indices.clone();
            let sender = sender.clone();
            let batch_counter = batch_counter.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Check if we have enough items to form a batch
                    let can_make_batch = {
                        let indices_guard = indices.lock().unwrap();
                        indices_guard.len() >= batch_size
                    };

                    if !can_make_batch {
                        thread::sleep(Duration::from_millis(10));

                        // Check if we should stop (no more data)
                        let should_stop = {
                            let indices_guard = indices.lock().unwrap();
                            indices_guard.is_empty()
                        };

                        if should_stop {
                            break;
                        }
                        continue;
                    }

                    // Get batch indices
                    let batch_indices = {
                        let mut indices_guard = indices.lock().unwrap();
                        let mut batch = Vec::new();
                        for _ in 0..batch_size {
                            if let Some(index) = indices_guard.pop_front() {
                                batch.push(index);
                            } else {
                                break;
                            }
                        }
                        batch
                    };

                    if batch_indices.is_empty() {
                        break;
                    }

                    // Load batch data
                    let batch_data: Vec<_> = batch_indices
                        .into_iter()
                        .filter_map(|idx| dataset.get(idx))
                        .collect();

                    if !batch_data.is_empty() {
                        let features: Vec<&Tensor<T>> = batch_data.iter().map(|(f, _)| f).collect();
                        let targets: Vec<&Tensor<T>> = batch_data.iter().map(|(_, t)| t).collect();

                        if let (Ok(batch_features), Ok(batch_targets)) =
                            (Tensor::stack(&features), Tensor::stack(&targets))
                        {
                            // Check buffer size
                            {
                                let mut counter = batch_counter.lock().unwrap();
                                if *counter >= buffer_size {
                                    thread::sleep(Duration::from_millis(5));
                                    continue;
                                }
                                *counter += 1;
                            }

                            if sender.send(Some((batch_features, batch_targets))).is_err() {
                                break;
                            }
                        }
                    }

                    thread::sleep(Duration::from_millis(1));
                }

                // Signal completion
                let _ = sender.send(None);
            });

            handles.push(handle);
        }

        // Drop original sender
        drop(sender);

        AsyncDataLoader {
            receiver,
            _handles: handles,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get next batch (non-blocking with timeout)
    /// 次のバッチを取得（タイムアウト付きノンブロッキング）
    pub fn next_batch_timeout(&self, timeout: Duration) -> Option<(Tensor<T>, Tensor<T>)> {
        match self.receiver.recv_timeout(timeout) {
            Ok(Some(batch)) => Some(batch),
            Ok(None) | Err(_) => None,
        }
    }
}

impl<
        T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
        D: LegacyDataset<T> + Send + Sync + 'static,
    > Iterator for AsyncDataLoader<T, D>
{
    type Item = (Tensor<T>, Tensor<T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch_timeout(Duration::from_secs(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;

    #[test]
    #[cfg(not(feature = "ci-fast"))]
    fn test_streaming_dataset() {
        let features = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0, 4.0], vec![2]),
            Tensor::from_vec(vec![5.0, 6.0], vec![2]),
            Tensor::from_vec(vec![7.0, 8.0], vec![2]),
        ];

        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];

        let dataset = TensorDataset::from_features_targets(features, targets).unwrap();
        let streaming_dataset = StreamingDataset::new(dataset, 2, 2);

        // Start prefetching
        streaming_dataset.start_prefetch(vec![0, 1, 2, 3]);

        // Wait a bit for prefetching (reduced for CI)
        std::thread::sleep(Duration::from_millis(10));

        // Test accessing data
        if let Some((feature, target)) = streaming_dataset.get(0) {
            assert_eq!(feature.as_slice().unwrap(), &[1.0, 2.0]);
            assert_eq!(target.as_slice().unwrap(), &[0.0]);
        }

        let (buffer_size, loaded_size) = streaming_dataset.buffer_stats();
        assert!(buffer_size <= 2); // Buffer size limit
        assert!(loaded_size > 0); // Something should be loaded
    }

    #[test]
    fn test_dynamic_batch_loader() {
        let features = vec![
            Tensor::from_vec(vec![1.0f32; 1000], vec![1000]), // Large tensor
            Tensor::from_vec(vec![2.0f32; 1000], vec![1000]),
            Tensor::from_vec(vec![3.0f32; 1000], vec![1000]),
            Tensor::from_vec(vec![4.0f32; 1000], vec![1000]),
        ];

        let targets = vec![
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
            Tensor::from_vec(vec![0.0], vec![1]),
            Tensor::from_vec(vec![1.0], vec![1]),
        ];

        let dataset = TensorDataset::from_features_targets(features, targets).unwrap();
        let mut loader = DynamicBatchLoader::new(
            dataset, 1,     // min_batch_size
            4,     // max_batch_size
            8000,  // memory_threshold (8KB)
            false, // shuffle
        );

        let initial_batch_size = loader.current_batch_size();

        // Get first batch - should adjust batch size due to large tensors
        if let Some(_) = loader.next_batch() {
            // Batch size might change based on memory usage
            let new_batch_size = loader.current_batch_size();
            assert!(new_batch_size >= 1);
            assert!(new_batch_size <= 4);
        }
    }

    #[test]
    fn test_async_data_loader() {
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
        let mut async_loader = AsyncDataLoader::new(
            dataset, 2,     // batch_size
            3,     // buffer_size
            2,     // num_workers
            false, // shuffle
        );

        // Wait a bit for async loading to start (reduced for CI)
        std::thread::sleep(Duration::from_millis(10));

        let mut batches_received = 0;
        while let Some((batch_features, batch_targets)) = async_loader.next() {
            assert_eq!(batch_features.batch_size(), 2);
            assert_eq!(batch_targets.batch_size(), 2);
            batches_received += 1;

            // Don't wait forever
            if batches_received >= 3 {
                break;
            }
        }

        assert!(batches_received > 0);
    }

    #[test]
    fn test_memory_estimation() {
        let features = vec![Tensor::from_vec(vec![1.0f32; 100], vec![100])];
        let targets = vec![Tensor::from_vec(vec![0.0f32], vec![1])];
        let dataset = TensorDataset::from_features_targets(features, targets).unwrap();

        let loader = DynamicBatchLoader::new(dataset, 1, 2, 1000, false);

        let test_tensor = Tensor::from_vec(vec![1.0f32; 100], vec![100]);
        let estimated_size = loader.estimate_memory_usage(&test_tensor);
        let expected_size = 100 * std::mem::size_of::<f32>();

        assert_eq!(estimated_size, expected_size);
    }
}
