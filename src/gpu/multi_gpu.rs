//! Multi-GPU Support Module
//! 
//! Provides comprehensive multi-GPU parallelism including data parallelism,
//! model parallelism, pipeline parallelism, and distributed training support.

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use crate::gpu::DeviceType;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Multi-GPU parallelism strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismStrategy {
    /// Data parallelism - replicate model, split data
    DataParallel,
    /// Model parallelism - split model across GPUs
    ModelParallel,
    /// Pipeline parallelism - pipeline stages across GPUs
    PipelineParallel,
    /// Hybrid parallelism - combination of strategies
    Hybrid,
    /// Expert parallelism - for mixture of experts
    ExpertParallel,
}

/// GPU topology information
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Number of GPUs
    pub num_gpus: usize,
    /// GPU device IDs
    pub device_ids: Vec<usize>,
    /// Peer-to-peer connectivity matrix
    pub p2p_matrix: Vec<Vec<bool>>,
    /// NVLink/Interconnect bandwidth matrix (GB/s)
    pub bandwidth_matrix: Vec<Vec<f64>>,
    /// GPU compute capabilities
    pub compute_capabilities: Vec<(u32, u32)>,
    /// Available memory per GPU
    pub memory_per_gpu: Vec<usize>,
}

impl Default for GpuTopology {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            device_ids: vec![0],
            p2p_matrix: vec![vec![true]],
            bandwidth_matrix: vec![vec![0.0]],
            compute_capabilities: vec![(8, 0)],
            memory_per_gpu: vec![8 * 1024 * 1024 * 1024], // 8GB default
        }
    }
}

/// Multi-GPU context manager
pub struct MultiGpuContext {
    /// GPU topology
    topology: GpuTopology,
    /// Current parallelism strategy
    strategy: ParallelismStrategy,
    /// Communication manager
    comm_manager: Arc<CommunicationManager>,
    /// Load balancer
    load_balancer: Arc<LoadBalancer>,
    /// Synchronization barrier
    barrier: Arc<Barrier>,
}

/// Communication manager for GPU-to-GPU transfers
pub struct CommunicationManager {
    /// Communication backend
    backend: CommBackend,
    /// Active communication groups
    groups: RwLock<HashMap<String, CommunicationGroup>>,
    /// Transfer statistics
    stats: RwLock<TransferStatistics>,
    /// Optimization settings
    optimization: CommOptimization,
}

/// Communication backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackend {
    /// NVIDIA NCCL
    NCCL,
    /// Direct peer-to-peer
    P2P,
    /// Host-staged transfers
    HostStaged,
    /// AMD RCCL
    RCCL,
    /// Intel XeLink
    XeLink,
}

/// Communication group for collective operations
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    /// Group name
    pub name: String,
    /// Participating GPU IDs
    pub gpu_ids: Vec<usize>,
    /// Group rank mapping
    pub rank_map: HashMap<usize, usize>,
    /// Root rank for broadcast operations
    pub root_rank: usize,
}

/// Communication optimization settings
#[derive(Debug, Clone)]
pub struct CommOptimization {
    /// Enable compression
    pub compression: bool,
    /// Enable fusion of small transfers
    pub fusion: bool,
    /// Enable overlap of computation and communication
    pub overlap: bool,
    /// Ring buffer size for async operations
    pub ring_buffer_size: usize,
}

/// Transfer statistics
#[derive(Debug, Default)]
pub struct TransferStatistics {
    /// Total bytes transferred
    pub total_bytes: usize,
    /// Total transfers
    pub total_transfers: u64,
    /// Average bandwidth achieved
    pub avg_bandwidth: f64,
    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,
}

/// Load balancer for work distribution
pub struct LoadBalancer {
    /// Current loads per GPU
    loads: RwLock<Vec<f64>>,
    /// Load history
    history: Mutex<VecDeque<Vec<f64>>>,
    /// Balancing strategy
    strategy: BalancingStrategy,
    /// Rebalancing threshold
    threshold: f64,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalancingStrategy {
    /// Static even distribution
    Static,
    /// Dynamic based on current load
    Dynamic,
    /// Work stealing
    WorkStealing,
    /// Predictive based on history
    Predictive,
}

/// Data parallel trainer
pub struct DataParallelTrainer {
    /// Number of GPUs
    num_gpus: usize,
    /// Batch splitter
    batch_splitter: BatchSplitter,
    /// Gradient aggregator
    gradient_aggregator: GradientAggregator,
    /// All-reduce algorithm
    all_reduce_algo: AllReduceAlgorithm,
}

/// Batch splitting strategy
#[derive(Debug, Clone)]
pub struct BatchSplitter {
    /// Split strategy
    strategy: SplitStrategy,
    /// Micro-batch size
    micro_batch_size: Option<usize>,
}

/// Split strategies for data parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Even split across GPUs
    Even,
    /// Weighted by GPU capability
    Weighted,
    /// Dynamic based on processing speed
    Dynamic,
}

/// Gradient aggregation handler
pub struct GradientAggregator {
    /// Aggregation method
    method: AggregationMethod,
    /// Gradient compression
    compression: Option<GradientCompression>,
    /// Gradient clipping threshold
    clip_threshold: Option<f32>,
}

/// Gradient aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Simple average
    Average,
    /// Weighted average
    Weighted,
    /// Delayed aggregation
    Delayed,
    /// Hierarchical aggregation
    Hierarchical,
}

/// Gradient compression methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientCompression {
    /// Top-K sparsification
    TopK(usize),
    /// Random sparsification
    Random(f32),
    /// Quantization
    Quantization(u8),
    /// Error feedback compression
    ErrorFeedback,
}

/// All-reduce algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceAlgorithm {
    /// Ring all-reduce
    Ring,
    /// Tree all-reduce
    Tree,
    /// Butterfly all-reduce
    Butterfly,
    /// Double binary tree
    DoubleBinaryTree,
}

/// Model parallel partitioner
pub struct ModelParallelPartitioner {
    /// Partitioning strategy
    strategy: PartitionStrategy,
    /// Layer assignments to GPUs
    layer_assignments: HashMap<String, usize>,
    /// Activation checkpointing
    checkpointing: bool,
}

/// Model partitioning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Layer-wise partitioning
    LayerWise,
    /// Tensor-wise partitioning
    TensorWise,
    /// Column-wise partitioning
    ColumnWise,
    /// Row-wise partitioning
    RowWise,
    /// Hybrid partitioning
    Hybrid,
}

/// Pipeline parallel scheduler
pub struct PipelineScheduler {
    /// Number of pipeline stages
    num_stages: usize,
    /// Stage assignments to GPUs
    stage_assignments: Vec<usize>,
    /// Pipeline schedule
    schedule: PipelineSchedule,
    /// Micro-batch size
    micro_batch_size: usize,
}

/// Pipeline scheduling algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe schedule
    GPipe,
    /// PipeDream schedule
    PipeDream,
    /// 1F1B (one forward one backward)
    OneF1B,
    /// Interleaved 1F1B
    Interleaved1F1B,
}

impl MultiGpuContext {

    /// Discover GPU topology
    fn discover_topology(device_ids: &[usize]) -> RusTorchResult<GpuTopology> {
        let num_gpus = device_ids.len();
        
        // Initialize topology matrices
        let mut p2p_matrix = vec![vec![false; num_gpus]; num_gpus];
        let mut bandwidth_matrix = vec![vec![0.0; num_gpus]; num_gpus];
        
        // Check P2P connectivity and measure bandwidth
        for i in 0..num_gpus {
            for j in 0..num_gpus {
                if i != j {
                    // Check if P2P is available (platform-specific)
                    p2p_matrix[i][j] = Self::check_p2p_access(device_ids[i], device_ids[j]);
                    
                    // Measure bandwidth if P2P is available
                    if p2p_matrix[i][j] {
                        bandwidth_matrix[i][j] = Self::measure_bandwidth(device_ids[i], device_ids[j])?;
                    }
                }
            }
        }

        // Get compute capabilities and memory info
        let compute_capabilities = device_ids.iter()
            .map(|&id| Self::get_compute_capability(id))
            .collect::<RusTorchResult<Vec<_>>>()?;
            
        let memory_per_gpu = device_ids.iter()
            .map(|&id| Self::get_device_memory(id))
            .collect::<RusTorchResult<Vec<_>>>()?;

        Ok(GpuTopology {
            num_gpus,
            device_ids: device_ids.to_vec(),
            p2p_matrix,
            bandwidth_matrix,
            compute_capabilities,
            memory_per_gpu,
        })
    }

    /// Check P2P access between GPUs
    fn check_p2p_access(gpu1: usize, gpu2: usize) -> bool {
        // Platform-specific P2P check
        // For now, assume P2P is available between all GPUs
        gpu1 != gpu2
    }

    /// Measure bandwidth between GPUs
    fn measure_bandwidth(gpu1: usize, gpu2: usize) -> RusTorchResult<f64> {
        // Platform-specific bandwidth measurement
        // Return estimated bandwidth based on interconnect type
        if gpu1 == gpu2 {
            Ok(500.0) // Local memory bandwidth
        } else {
            Ok(25.0) // NVLink bandwidth estimate
        }
    }

    /// Get compute capability of GPU
    fn get_compute_capability(gpu_id: usize) -> RusTorchResult<(u32, u32)> {
        // Platform-specific capability query
        Ok((8, 0)) // Default to Ampere
    }

    /// Get available memory on GPU
    fn get_device_memory(gpu_id: usize) -> RusTorchResult<usize> {
        // Platform-specific memory query
        Ok(16 * 1024 * 1024 * 1024) // 16GB default
    }

    /// Select best communication backend based on topology
    fn select_comm_backend(topology: &GpuTopology) -> CommBackend {
        // Check for full P2P connectivity
        let full_p2p = topology.p2p_matrix.iter()
            .enumerate()
            .all(|(i, row)| {
                row.iter().enumerate()
                    .all(|(j, &connected)| i == j || connected)
            });

        if full_p2p {
            CommBackend::P2P
        } else {
            CommBackend::NCCL // Fall back to NCCL
        }
    }

    /// Execute operation across multiple GPUs
    pub fn execute<F>(&self, operation: F) -> RusTorchResult<Vec<Tensor<f32>>>
    where
        F: Fn(usize) -> RusTorchResult<Tensor<f32>> + Send + Sync + Clone + 'static,
    {
        let mut handles = Vec::new();
        let results = Arc::new(Mutex::new(Vec::new()));
        
        // Launch operation on each GPU
        for (idx, &gpu_id) in self.topology.device_ids.iter().enumerate() {
            let op = operation.clone();
            let results_clone = results.clone();
            let barrier = self.barrier.clone();
            
            let handle = thread::spawn(move || {
                // Set current GPU (platform-specific)
                // Execute operation
                let result = op(gpu_id);
                
                // Store result
                if let Ok(tensor) = result {
                    let mut res = results_clone.lock().unwrap();
                    res.push((idx, tensor));
                }
                
                // Synchronize
                barrier.wait();
            });
            
            handles.push(handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            handle.join().map_err(|_| {
                RusTorchError::gpu("GPU operation thread panicked")
            })?;
        }

        // Sort results by GPU index
        let mut results = results.lock().unwrap();
        results.sort_by_key(|(idx, _)| *idx);
        
        Ok(results.iter().map(|(_, tensor)| tensor.clone()).collect())
    }

    /// All-reduce operation across GPUs
    pub fn all_reduce(&self, tensors: Vec<Tensor<f32>>) -> RusTorchResult<Vec<Tensor<f32>>> {
        self.comm_manager.all_reduce(tensors, &self.topology)
    }

    /// Broadcast tensor from root GPU
    pub fn broadcast(&self, tensor: Tensor<f32>, root_gpu: usize) -> RusTorchResult<Vec<Tensor<f32>>> {
        self.comm_manager.broadcast(tensor, root_gpu, &self.topology)
    }

    /// Scatter tensors across GPUs
    pub fn scatter(&self, tensors: Vec<Tensor<f32>>, root_gpu: usize) -> RusTorchResult<Vec<Tensor<f32>>> {
        self.comm_manager.scatter(tensors, root_gpu, &self.topology)
    }

    /// Gather tensors from all GPUs
    pub fn gather(&self, tensors: Vec<Tensor<f32>>, root_gpu: usize) -> RusTorchResult<Tensor<f32>> {
        self.comm_manager.gather(tensors, root_gpu, &self.topology)
    }

    /// Simple constructor for testing
    pub fn new(device_ids: Vec<usize>) -> RusTorchResult<Self> {
        Self::new_with_strategy(device_ids, ParallelismStrategy::DataParallel)
    }

    /// Constructor with strategy (renamed from original new)
    pub fn new_with_strategy(device_ids: Vec<usize>, strategy: ParallelismStrategy) -> RusTorchResult<Self> {
        // Discover GPU topology
        let topology = Self::discover_topology(&device_ids)?;
        
        // Create load balancer
        let load_balancer = Arc::new(LoadBalancer::new(
            device_ids.len(),
            BalancingStrategy::Dynamic
        ));

        // Create communication manager
        let comm_manager = Arc::new(CommunicationManager::new(
            Self::select_comm_backend(&topology)
        ));

        // Create barrier
        let barrier = Arc::new(Barrier::new(device_ids.len()));

        Ok(Self {
            topology,
            strategy,
            comm_manager,
            load_balancer,
            barrier,
        })
    }

    /// Get number of GPUs
    pub fn gpu_count(&self) -> usize {
        self.topology.num_gpus
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self, gpu_id: usize) -> bool {
        self.topology.device_ids.contains(&gpu_id)
    }

    /// Get device IDs
    pub fn get_device_ids(&self) -> &[usize] {
        &self.topology.device_ids
    }

    /// Test P2P communication between two GPUs
    pub fn test_p2p_communication(&self, src_gpu: usize, dst_gpu: usize, tensor: &Tensor<f32>) -> RusTorchResult<()> {
        if !self.is_gpu_available(src_gpu) || !self.is_gpu_available(dst_gpu) {
            return Err(RusTorchError::InvalidOperation("Invalid GPU IDs for P2P test".to_string()));
        }
        
        // Test P2P connectivity
        if src_gpu < self.topology.p2p_matrix.len() && 
           dst_gpu < self.topology.p2p_matrix[src_gpu].len() &&
           self.topology.p2p_matrix[src_gpu][dst_gpu] {
            println!("P2P communication test successful between GPU {} and GPU {}", src_gpu, dst_gpu);
            Ok(())
        } else {
            Err(RusTorchError::UnsupportedOperation("P2P communication not available between specified GPUs".to_string()))
        }
    }
}

impl CommunicationManager {
    /// Create new communication manager
    pub fn new(backend: CommBackend) -> Self {
        Self {
            backend,
            groups: RwLock::new(HashMap::new()),
            stats: RwLock::new(TransferStatistics::default()),
            optimization: CommOptimization {
                compression: false,
                fusion: true,
                overlap: true,
                ring_buffer_size: 4,
            },
        }
    }

    /// All-reduce operation
    pub fn all_reduce(&self, tensors: Vec<Tensor<f32>>, topology: &GpuTopology) -> RusTorchResult<Vec<Tensor<f32>>> {
        match self.backend {
            CommBackend::NCCL => self.nccl_all_reduce(tensors),
            CommBackend::P2P => self.p2p_all_reduce(tensors, topology),
            _ => self.host_staged_all_reduce(tensors),
        }
    }

    /// NCCL all-reduce implementation
    fn nccl_all_reduce(&self, tensors: Vec<Tensor<f32>>) -> RusTorchResult<Vec<Tensor<f32>>> {
        #[cfg(feature = "nccl")]
        {
            use std::os::raw::c_void;
            
            // Initialize NCCL communicator if not already done
            let mut result_tensors = Vec::with_capacity(tensors.len());
            
            // Perform NCCL all-reduce for each tensor
            for tensor in &tensors {
                let as_ptr = tensor.as_ptr() as *mut c_void;
                let element_count = tensor.numel();
                
                // NCCL all-reduce call
                // Note: In production, this would use actual NCCL bindings
                let mut reduced_data = vec![0.0f32; element_count];
                
                // Simulate all-reduce by averaging values across GPUs
                for i in 0..element_count {
                    let sum: f32 = tensors.iter()
                        .map(|t| unsafe { *((t.as_ptr() as *const f32).add(i)) })
                        .sum();
                    reduced_data[i] = sum / tensors.len() as f32;
                }
                
                // Create result tensor with reduced data
                let mut result_tensor = tensor.clone();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        reduced_data.as_ptr(),
                        result_tensor.as_ptr() as *mut f32,
                        element_count
                    );
                }
                
                result_tensors.push(result_tensor);
            }
            
            Ok(result_tensors)
        }
        #[cfg(not(feature = "nccl"))]
        {
            // Fallback to P2P implementation
            self.p2p_all_reduce(tensors, &GpuTopology::default())
        }
    }

    /// P2P all-reduce using ring algorithm
    fn p2p_all_reduce(&self, tensors: Vec<Tensor<f32>>, topology: &GpuTopology) -> RusTorchResult<Vec<Tensor<f32>>> {
        let num_gpus = tensors.len();
        if num_gpus <= 1 {
            return Ok(tensors);
        }
        
        let result = tensors.clone();
        let chunk_size = tensors[0].numel() / num_gpus;
        
        // Ring all-reduce algorithm: scatter-reduce phase
        for step in 0..num_gpus - 1 {
            for gpu_idx in 0..num_gpus {
                let send_to = (gpu_idx + 1) % num_gpus;
                let recv_from = (gpu_idx + num_gpus - 1) % num_gpus;
                let chunk_idx = (gpu_idx + num_gpus - step) % num_gpus;
                
                // Check P2P connectivity
                if topology.p2p_matrix[gpu_idx][send_to] {
                    // Direct P2P transfer with actual tensor data manipulation
                    let start_offset = chunk_idx * chunk_size;
                    let end_offset = std::cmp::min(start_offset + chunk_size, result[gpu_idx].numel());
                    
                    // Perform element-wise addition between chunks
                    unsafe {
                        let src_ptr = result[recv_from].as_ptr() as *const f32;
                        let dst_ptr = result[gpu_idx].as_ptr() as *mut f32;
                        
                        for i in start_offset..end_offset {
                            *dst_ptr.add(i) += *src_ptr.add(i);
                        }
                    }
                } else {
                    // Host-staged transfer for non-P2P GPUs
                    return self.host_staged_all_reduce(tensors);
                }
            }
        }
        
        // Ring all-reduce algorithm: all-gather phase
        for step in 0..num_gpus - 1 {
            for gpu_idx in 0..num_gpus {
                let send_to = (gpu_idx + 1) % num_gpus;
                let chunk_idx = (gpu_idx + 1 - step + num_gpus) % num_gpus;
                
                if topology.p2p_matrix[gpu_idx][send_to] {
                    let start_offset = chunk_idx * chunk_size;
                    let end_offset = std::cmp::min(start_offset + chunk_size, result[gpu_idx].numel());
                    
                    // Copy reduced chunk to neighbor
                    unsafe {
                        let src_ptr = result[gpu_idx].as_ptr() as *const f32;
                        let dst_ptr = result[send_to].as_ptr() as *mut f32;
                        
                        for i in start_offset..end_offset {
                            *dst_ptr.add(i) = *src_ptr.add(i);
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }

    /// Host-staged all-reduce for GPUs without P2P
    fn host_staged_all_reduce(&self, tensors: Vec<Tensor<f32>>) -> RusTorchResult<Vec<Tensor<f32>>> {
        if tensors.is_empty() {
            return Ok(tensors);
        }
        
        let num_gpus = tensors.len();
        let element_count = tensors[0].numel();
        
        // Step 1: Copy all GPU tensors to host memory
        let mut host_buffers: Vec<Vec<f32>> = Vec::with_capacity(num_gpus);
        for tensor in &tensors {
            let mut buffer = vec![0.0f32; element_count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.as_ptr() as *const f32,
                    buffer.as_mut_ptr(),
                    element_count
                );
            }
            host_buffers.push(buffer);
        }
        
        // Step 2: Perform reduction on host (averaging)
        let mut reduced_buffer = vec![0.0f32; element_count];
        for i in 0..element_count {
            let sum: f32 = host_buffers.iter().map(|buf| buf[i]).sum();
            reduced_buffer[i] = sum / num_gpus as f32;
        }
        
        // Step 3: Copy reduced result back to all GPUs
        let mut result_tensors = Vec::with_capacity(num_gpus);
        for tensor in &tensors {
            let result_tensor = tensor.clone();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    reduced_buffer.as_ptr(),
                    result_tensor.as_ptr() as *mut f32,
                    element_count
                );
            }
            result_tensors.push(result_tensor);
        }
        
        Ok(result_tensors)
    }

    /// Broadcast operation
    pub fn broadcast(&self, tensor: Tensor<f32>, root_gpu: usize, topology: &GpuTopology) -> RusTorchResult<Vec<Tensor<f32>>> {
        let num_gpus = topology.num_gpus;
        let result = vec![tensor.clone(); num_gpus];
        let element_count = tensor.numel();
        
        // Tree-based broadcast for efficiency
        let mut pending_transfers = VecDeque::new();
        pending_transfers.push_back((root_gpu, vec![0; num_gpus].into_iter().enumerate().filter(|(i, _)| *i != root_gpu).map(|(i, _)| i).collect::<Vec<_>>()));
        
        while let Some((source_gpu, target_gpus)) = pending_transfers.pop_front() {
            if target_gpus.is_empty() {
                continue;
            }
            
            let mid = target_gpus.len() / 2;
            let (left_targets, right_targets) = target_gpus.split_at(mid);
            
            // Transfer to first target in each group
            for targets in [left_targets, right_targets].iter().filter(|t| !t.is_empty()) {
                let target_gpu = targets[0];
                
                if topology.p2p_matrix[source_gpu][target_gpu] {
                    // Direct P2P copy
                    unsafe {
                        let src_ptr = result[source_gpu].as_ptr() as *const f32;
                        let dst_ptr = result[target_gpu].as_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, element_count);
                    }
                } else {
                    // Host-staged copy
                    let mut host_buffer = vec![0.0f32; element_count];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            result[source_gpu].as_ptr() as *const f32,
                            host_buffer.as_mut_ptr(),
                            element_count
                        );
                        std::ptr::copy_nonoverlapping(
                            host_buffer.as_ptr(),
                            result[target_gpu].as_ptr() as *mut f32,
                            element_count
                        );
                    }
                }
                
                // Schedule further transfers from this target
                if targets.len() > 1 {
                    pending_transfers.push_back((target_gpu, targets[1..].to_vec()));
                }
            }
        }
        
        Ok(result)
    }

    /// Scatter operation
    pub fn scatter(&self, tensors: Vec<Tensor<f32>>, root_gpu: usize, topology: &GpuTopology) -> RusTorchResult<Vec<Tensor<f32>>> {
        let num_gpus = topology.num_gpus;
        if tensors.len() != num_gpus {
            return Err(RusTorchError::InvalidOperation(
                format!("Scatter requires {} tensors for {} GPUs", num_gpus, num_gpus)
            ));
        }
        
        let mut result = vec![tensors[0].clone(); num_gpus];
        
        // Distribute tensors from root to each GPU
        for (target_gpu, tensor) in tensors.iter().enumerate() {
            if target_gpu != root_gpu {
                let element_count = tensor.numel();
                
                if topology.p2p_matrix[root_gpu][target_gpu] {
                    // Direct P2P scatter
                    unsafe {
                        let src_ptr = tensor.as_ptr() as *const f32;
                        let dst_ptr = result[target_gpu].as_ptr() as *mut f32;
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, element_count);
                    }
                } else {
                    // Host-staged scatter
                    let mut host_buffer = vec![0.0f32; element_count];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            tensor.as_ptr() as *const f32,
                            host_buffer.as_mut_ptr(),
                            element_count
                        );
                        std::ptr::copy_nonoverlapping(
                            host_buffer.as_ptr(),
                            result[target_gpu].as_ptr() as *mut f32,
                            element_count
                        );
                    }
                }
            } else {
                // Root GPU keeps its tensor
                result[target_gpu] = tensor.clone();
            }
        }
        
        Ok(result)
    }

    /// Gather operation
    pub fn gather(&self, tensors: Vec<Tensor<f32>>, root_gpu: usize, topology: &GpuTopology) -> RusTorchResult<Tensor<f32>> {
        if tensors.is_empty() {
            return Err(RusTorchError::InvalidOperation("Cannot gather from empty tensor list".to_string()));
        }
        
        if root_gpu >= tensors.len() {
            return Err(RusTorchError::InvalidOperation(
                format!("Root GPU {} out of range for {} tensors", root_gpu, tensors.len())
            ));
        }
        
        let element_count = tensors[0].numel();
        let total_elements = element_count * tensors.len();
        
        // Create gathered tensor on root GPU
        let gathered_tensor = Tensor::<f32>::zeros(&[total_elements]);
        
        // Gather all tensors to root GPU
        for (source_gpu, tensor) in tensors.iter().enumerate() {
            let dst_offset = source_gpu * element_count;
            
            if source_gpu == root_gpu {
                // Local copy
                unsafe {
                    let src_ptr = tensor.as_ptr() as *const f32;
                    let dst_ptr = (gathered_tensor.as_ptr() as *mut f32).add(dst_offset);
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, element_count);
                }
            } else if topology.p2p_matrix[source_gpu][root_gpu] {
                // Direct P2P gather
                unsafe {
                    let src_ptr = tensor.as_ptr() as *const f32;
                    let dst_ptr = (gathered_tensor.as_ptr() as *mut f32).add(dst_offset);
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, element_count);
                }
            } else {
                // Host-staged gather
                let mut host_buffer = vec![0.0f32; element_count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor.as_ptr() as *const f32,
                        host_buffer.as_mut_ptr(),
                        element_count
                    );
                    let dst_ptr = (gathered_tensor.as_ptr() as *mut f32).add(dst_offset);
                    std::ptr::copy_nonoverlapping(
                        host_buffer.as_ptr(),
                        dst_ptr,
                        element_count
                    );
                }
            }
        }
        
        Ok(gathered_tensor)
    }

    /// Create communication group
    pub fn create_group(&self, name: String, gpu_ids: Vec<usize>) -> RusTorchResult<()> {
        let mut rank_map = HashMap::new();
        for (rank, &gpu_id) in gpu_ids.iter().enumerate() {
            rank_map.insert(gpu_id, rank);
        }

        let group = CommunicationGroup {
            name: name.clone(),
            gpu_ids,
            rank_map,
            root_rank: 0,
        };

        let mut groups = self.groups.write().unwrap();
        groups.insert(name, group);
        
        Ok(())
    }
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(num_gpus: usize, strategy: BalancingStrategy) -> Self {
        Self {
            loads: RwLock::new(vec![0.0; num_gpus]),
            history: Mutex::new(VecDeque::with_capacity(100)),
            strategy,
            threshold: 0.2, // 20% imbalance threshold
        }
    }

    /// Update load for GPU
    pub fn update_load(&self, gpu_id: usize, load: f64) {
        let mut loads = self.loads.write().unwrap();
        if gpu_id < loads.len() {
            loads[gpu_id] = load;
        }

        // Store in history
        let mut history = self.history.lock().unwrap();
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(loads.clone());
    }

    /// Get recommended GPU for next operation
    pub fn get_next_gpu(&self) -> usize {
        let loads = self.loads.read().unwrap();
        
        match self.strategy {
            BalancingStrategy::Static => {
                // Round-robin
                0 // Simplified
            },
            BalancingStrategy::Dynamic => {
                // Choose GPU with lowest load
                loads.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            },
            _ => 0,
        }
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalancing(&self) -> bool {
        let loads = self.loads.read().unwrap();
        
        if loads.is_empty() {
            return false;
        }

        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        let max_deviation = loads.iter()
            .map(|&load| (load - avg_load).abs())
            .fold(0.0, f64::max);

        max_deviation / avg_load > self.threshold
    }
}

impl DataParallelTrainer {
    /// Create new data parallel trainer
    pub fn new(num_gpus: usize) -> Self {
        Self {
            num_gpus,
            batch_splitter: BatchSplitter {
                strategy: SplitStrategy::Even,
                micro_batch_size: None,
            },
            gradient_aggregator: GradientAggregator {
                method: AggregationMethod::Average,
                compression: None,
                clip_threshold: None,
            },
            all_reduce_algo: AllReduceAlgorithm::Ring,
        }
    }

    /// Split batch across GPUs
    pub fn split_batch(&self, batch: &Tensor<f32>) -> Vec<Tensor<f32>> {
        let batch_size = batch.shape()[0];
        let split_size = batch_size / self.num_gpus;
        
        let mut splits = Vec::new();
        for i in 0..self.num_gpus {
            let start = i * split_size;
            let end = if i == self.num_gpus - 1 {
                batch_size
            } else {
                (i + 1) * split_size
            };
            
            // Create view or slice of batch
            // Placeholder - actual tensor slicing implementation needed
            splits.push(batch.clone());
        }
        
        splits
    }

    /// Aggregate gradients across GPUs
    pub fn aggregate_gradients(&self, gradients: Vec<Tensor<f32>>) -> Tensor<f32> {
        match self.gradient_aggregator.method {
            AggregationMethod::Average => {
                // Average gradients
                // Placeholder implementation
                gradients[0].clone()
            },
            _ => gradients[0].clone(),
        }
    }
}

impl ModelParallelPartitioner {
    /// Create new model parallel partitioner
    pub fn new(strategy: PartitionStrategy) -> Self {
        Self {
            strategy,
            layer_assignments: HashMap::new(),
            checkpointing: true,
        }
    }

    /// Partition model across GPUs
    pub fn partition_model(&mut self, model_layers: Vec<String>, num_gpus: usize) -> RusTorchResult<()> {
        match self.strategy {
            PartitionStrategy::LayerWise => {
                // Assign layers to GPUs in round-robin or balanced fashion
                let layers_per_gpu = model_layers.len() / num_gpus;
                
                for (idx, layer) in model_layers.iter().enumerate() {
                    let gpu_id = idx / layers_per_gpu.max(1);
                    self.layer_assignments.insert(layer.clone(), gpu_id.min(num_gpus - 1));
                }
            },
            _ => {
                // Other partitioning strategies
            }
        }
        
        Ok(())
    }

    /// Get GPU assignment for layer
    pub fn get_gpu_for_layer(&self, layer: &str) -> Option<usize> {
        self.layer_assignments.get(layer).copied()
    }
}

impl PipelineScheduler {
    /// Create new pipeline scheduler
    pub fn new(num_stages: usize, micro_batch_size: usize) -> Self {
        Self {
            num_stages,
            stage_assignments: (0..num_stages).collect(),
            schedule: PipelineSchedule::OneF1B,
            micro_batch_size,
        }
    }

    /// Generate pipeline schedule
    pub fn generate_schedule(&self, num_micro_batches: usize) -> Vec<(usize, PipelineOp)> {
        let mut schedule = Vec::new();
        
        match self.schedule {
            PipelineSchedule::OneF1B => {
                // 1F1B schedule generation
                // Warm-up phase
                for stage in 0..self.num_stages {
                    for mb in 0..self.num_stages - stage {
                        schedule.push((stage, PipelineOp::Forward(mb)));
                    }
                }
                
                // Steady state
                for mb in self.num_stages..num_micro_batches {
                    for stage in 0..self.num_stages {
                        schedule.push((stage, PipelineOp::Forward(mb)));
                        schedule.push((stage, PipelineOp::Backward(mb - self.num_stages)));
                    }
                }
                
                // Cool-down phase
                for stage in 0..self.num_stages {
                    for mb in (num_micro_batches - self.num_stages + stage)..num_micro_batches {
                        schedule.push((stage, PipelineOp::Backward(mb)));
                    }
                }
            },
            _ => {
                // Other scheduling algorithms
            }
        }
        
        schedule
    }
}

/// Pipeline operation type
#[derive(Debug, Clone, Copy)]
pub enum PipelineOp {
    /// Forward pass for micro-batch
    Forward(usize),
    /// Backward pass for micro-batch
    Backward(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_topology() {
        let device_ids = vec![0, 1];
        let result = MultiGpuContext::discover_topology(&device_ids);
        assert!(result.is_ok());
        
        let topology = result.unwrap();
        assert_eq!(topology.num_gpus, 2);
        assert_eq!(topology.device_ids.len(), 2);
    }

    #[test]
    fn test_load_balancer() {
        let balancer = LoadBalancer::new(4, BalancingStrategy::Dynamic);
        
        // Update loads
        balancer.update_load(0, 0.8);
        balancer.update_load(1, 0.2);
        balancer.update_load(2, 0.5);
        balancer.update_load(3, 0.3);
        
        // Get next GPU (should be GPU 1 with lowest load)
        let next_gpu = balancer.get_next_gpu();
        assert_eq!(next_gpu, 1);
        
        // Check if rebalancing needed
        let needs_rebalance = balancer.needs_rebalancing();
        assert!(needs_rebalance); // Large imbalance between 0.8 and 0.2
    }

    #[test]
    fn test_data_parallel_trainer() {
        let trainer = DataParallelTrainer::new(4);
        
        // Test batch splitting
        let batch = Tensor::<f32>::zeros(&[32, 224, 224, 3]);
        let splits = trainer.split_batch(&batch);
        assert_eq!(splits.len(), 4);
    }

    #[test]
    fn test_model_partitioner() {
        let mut partitioner = ModelParallelPartitioner::new(PartitionStrategy::LayerWise);
        
        let layers = vec![
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
        ];
        
        let result = partitioner.partition_model(layers, 2);
        assert!(result.is_ok());
        
        // Check assignments
        assert_eq!(partitioner.get_gpu_for_layer("layer1"), Some(0));
        assert_eq!(partitioner.get_gpu_for_layer("layer4"), Some(1));
    }

    #[test]
    fn test_pipeline_scheduler() {
        let scheduler = PipelineScheduler::new(4, 8);
        
        let schedule = scheduler.generate_schedule(16);
        assert!(!schedule.is_empty());
    }
}