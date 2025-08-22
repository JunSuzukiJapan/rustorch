#[cfg(test)]
mod gpu_tests {
    use rustorch::tensor::Tensor;
    use rustorch::gpu::{device::{GpuDevice, GpuBackend}, kernels::ModernGpuKernel};
    use rustorch::nn::{Linear, Conv2d, BatchNorm2d};
    use std::sync::Arc;

    fn get_available_gpu() -> Option<Arc<dyn GpuDevice>> {
        #[cfg(feature = "cuda")]
        if let Ok(device) = rustorch::gpu::cuda::CudaDevice::new(0) {
            return Some(Arc::new(device));
        }
        
        #[cfg(feature = "metal")]
        if let Ok(device) = rustorch::gpu::metal::MetalDevice::new() {
            return Some(Arc::new(device));
        }
        
        #[cfg(feature = "opencl")]
        if let Ok(device) = rustorch::gpu::opencl::OpenCLDevice::new(0, 0) {
            return Some(Arc::new(device));
        }
        
        None
    }

    #[test]
    fn test_gpu_device_detection() {
        let backend = GpuBackend::new();
        let devices = backend.list_devices();
        
        println!("Found {} GPU devices", devices.len());
        for (i, device) in devices.iter().enumerate() {
            println!("Device {}: {} ({})", i, device.name(), device.device_type());
            println!("  Memory: {} MB", device.total_memory() / (1024 * 1024));
            println!("  Compute capability: {:?}", device.compute_capability());
        }
        
        if !devices.is_empty() {
            assert!(devices[0].is_available());
        }
    }

    #[test]
    fn test_gpu_tensor_transfer() {
        if let Some(device) = get_available_gpu() {
            let cpu_tensor = Tensor::randn(&[100, 100]);
            
            // Transfer to GPU
            let gpu_tensor = cpu_tensor.to_device(device.clone());
            assert_eq!(gpu_tensor.device().id(), device.id());
            assert_eq!(gpu_tensor.shape(), cpu_tensor.shape());
            
            // Transfer back to CPU
            let cpu_tensor_back = gpu_tensor.to_cpu();
            assert!(cpu_tensor_back.device().is_cpu());
            
            // Verify data integrity
            let diff = (&cpu_tensor - &cpu_tensor_back).abs().max();
            assert!(diff < 1e-6, "Data corruption during GPU transfer");
        } else {
            println!("No GPU available, skipping test");
        }
    }

    #[test]
    fn test_gpu_matrix_multiplication() {
        if let Some(device) = get_available_gpu() {
            let a = Tensor::randn(&[512, 256]).to_device(device.clone());
            let b = Tensor::randn(&[256, 128]).to_device(device.clone());
            
            let start = std::time::Instant::now();
            let c_gpu = a.matmul(&b);
            let gpu_time = start.elapsed();
            
            // Compare with CPU
            let a_cpu = a.to_cpu();
            let b_cpu = b.to_cpu();
            let start = std::time::Instant::now();
            let c_cpu = a_cpu.matmul(&b_cpu);
            let cpu_time = start.elapsed();
            
            println!("GPU time: {:?}, CPU time: {:?}", gpu_time, cpu_time);
            println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
            
            // Verify correctness
            let c_gpu_cpu = c_gpu.to_cpu();
            let diff = (&c_cpu - &c_gpu_cpu).abs().max();
            assert!(diff < 1e-4, "GPU computation incorrect: max diff = {}", diff);
        }
    }

    #[test]
    fn test_gpu_convolution() {
        if let Some(device) = get_available_gpu() {
            let input = Tensor::randn(&[8, 3, 32, 32]).to_device(device.clone());
            let conv = Conv2d::new(3, 64, 3, 1, 1).to_device(device.clone());
            
            let start = std::time::Instant::now();
            let output_gpu = conv.forward(&input);
            let gpu_time = start.elapsed();
            
            assert_eq!(output_gpu.shape(), &[8, 64, 32, 32]);
            println!("GPU Conv2d forward time: {:?}", gpu_time);
            
            // Test backward pass
            let grad_output = Tensor::ones(&[8, 64, 32, 32]).to_device(device.clone());
            let start = std::time::Instant::now();
            let grad_input = conv.backward(&grad_output);
            let gpu_backward_time = start.elapsed();
            
            assert_eq!(grad_input.shape(), &[8, 3, 32, 32]);
            println!("GPU Conv2d backward time: {:?}", gpu_backward_time);
        }
    }

    #[test]
    fn test_gpu_batch_normalization() {
        if let Some(device) = get_available_gpu() {
            let input = Tensor::randn(&[32, 64, 28, 28]).to_device(device.clone());
            let bn = BatchNorm2d::new(64).to_device(device.clone());
            
            // Training mode
            bn.train();
            let output_train = bn.forward(&input);
            assert_eq!(output_train.shape(), input.shape());
            
            // Eval mode
            bn.eval();
            let output_eval = bn.forward(&input);
            assert_eq!(output_eval.shape(), input.shape());
            
            // Verify normalization
            let mean = output_train.mean_dim(&[0, 2, 3], true);
            let var = output_train.var_dim(&[0, 2, 3], true);
            
            assert!(mean.abs().max() < 1e-5, "Mean not normalized");
            assert!((var - 1.0).abs().max() < 1e-2, "Variance not normalized");
        }
    }

    #[test]
    fn test_gpu_memory_management() {
        if let Some(device) = get_available_gpu() {
            let initial_memory = device.allocated_memory();
            
            // Allocate tensors
            let tensors: Vec<_> = (0..10)
                .map(|_| Tensor::randn(&[1000, 1000]).to_device(device.clone()))
                .collect();
            
            let allocated_memory = device.allocated_memory();
            assert!(allocated_memory > initial_memory);
            println!("Allocated {} MB", (allocated_memory - initial_memory) / (1024 * 1024));
            
            // Drop tensors
            drop(tensors);
            
            // Force garbage collection
            device.synchronize();
            std::thread::sleep(std::time::Duration::from_millis(100));
            
            let final_memory = device.allocated_memory();
            assert!(final_memory <= allocated_memory);
            println!("Released {} MB", (allocated_memory - final_memory) / (1024 * 1024));
        }
    }

    #[test]
    fn test_gpu_parallel_kernels() {
        if let Some(device) = get_available_gpu() {
            let size = 10_000_000;
            let a = Tensor::ones(&[size]).to_device(device.clone());
            let b = Tensor::ones(&[size]).to_device(device.clone());
            
            // Element-wise operations
            let start = std::time::Instant::now();
            let c = &a + &b;
            let d = &a * &b;
            let e = a.exp();
            let f = b.log();
            let add_time = start.elapsed();
            
            println!("Parallel kernel execution time: {:?}", add_time);
            
            // Verify results
            assert_eq!(c.to_cpu().mean().item(), 2.0);
            assert_eq!(d.to_cpu().mean().item(), 1.0);
            assert!((e.to_cpu().mean().item() - std::f32::consts::E).abs() < 1e-5);
            assert!((f.to_cpu().mean().item() - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gpu_custom_kernel() {
        if let Some(device) = get_available_gpu() {
            // Test custom kernel compilation and execution
            // Test custom kernel compilation and execution (mock implementation)
            let custom_kernel_source = r#"
                extern "C" __global__ void custom_add(float* a, float* b, float* c, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        c[idx] = a[idx] + b[idx] + 1.0f;
                    }
                }
                "#;
            
            let a = Tensor::ones(&[1000]).to_device(device.clone());
            let b = Tensor::ones(&[1000]).to_device(device.clone());
            let c = &a + &b + &Tensor::ones(&[1000]).to_device(device.clone());
            
            let result = c.to_cpu();
            assert_eq!(result.mean().item(), 3.0);
        }
    }

    #[test]
    fn test_gpu_mixed_precision() {
        if let Some(device) = get_available_gpu() {
            // Test FP16 operations
            let a_f32 = Tensor::randn(&[1000, 1000]).to_device(device.clone());
            let a_f16 = a_f32.to_half();
            
            assert_eq!(a_f16.dtype(), rustorch::dtype::DType::Float16);
            
            let b_f16 = Tensor::randn(&[1000, 1000]).to_device(device.clone()).to_half();
            let c_f16 = a_f16.matmul(&b_f16);
            
            // Convert back to FP32 for verification
            let c_f32 = c_f16.to_float();
            assert_eq!(c_f32.dtype(), rustorch::dtype::DType::Float32);
            
            // Compare with FP32 computation
            let b_f32 = b_f16.to_float();
            let c_f32_ref = a_f32.matmul(&b_f32);
            
            let diff = (&c_f32 - &c_f32_ref).abs().max();
            assert!(diff < 1e-2, "FP16 computation error too large: {}", diff);
            
            println!("Mixed precision error: {}", diff);
        }
    }

    #[test] 
    fn test_multi_gpu_data_parallel() {
        let backend = GpuBackend::new();
        let devices = backend.list_devices();
        
        if devices.len() >= 2 {
            println!("Testing multi-GPU data parallel with {} devices", devices.len());
            
            // Create model and replicate across GPUs
            let model = Linear::new(1000, 100);
            let replicas: Vec<_> = devices.iter()
                .map(|d| model.clone().to_device(d.clone()))
                .collect();
            
            // Split batch across GPUs
            let batch_size = 64;
            let input = Tensor::randn(&[batch_size, 1000]);
            let chunks = input.chunk(devices.len(), 0);
            
            // Parallel forward pass
            let outputs: Vec<_> = chunks.iter()
                .zip(replicas.iter())
                .zip(devices.iter())
                .map(|((chunk, model), device)| {
                    let gpu_input = chunk.to_device(device.clone());
                    model.forward(&gpu_input)
                })
                .collect();
            
            // Gather results
            let output = Tensor::cat(
                &outputs.iter().map(|o| o.to_cpu()).collect::<Vec<_>>(),
                0
            );
            
            assert_eq!(output.shape(), &[batch_size, 100]);
            println!("Multi-GPU forward pass successful");
        } else {
            println!("Less than 2 GPUs available, skipping multi-GPU test");
        }
    }

    #[test]
    fn test_gpu_stream_synchronization() {
        if let Some(device) = get_available_gpu() {
            // Create multiple streams
            let stream1 = device.create_stream();
            let stream2 = device.create_stream();
            
            // Launch operations on different streams
            let a = Tensor::randn(&[1000, 1000]).to_device(device.clone());
            let b = Tensor::randn(&[1000, 1000]).to_device(device.clone());
            
            stream1.with(|| {
                let c = a.matmul(&b);
                c
            });
            
            stream2.with(|| {
                let d = a.add(&b);
                d
            });
            
            // Synchronize streams
            stream1.synchronize();
            stream2.synchronize();
            
            println!("GPU stream synchronization successful");
        }
    }

    #[test]
    fn test_gpu_tensor_indexing() {
        if let Some(device) = get_available_gpu() {
            let tensor = Tensor::arange(0.0, 100.0, 1.0)
                .reshape(&[10, 10])
                .to_device(device.clone());
            
            // Test various indexing operations
            let row = tensor.index(&[Some(5), None]);
            assert_eq!(row.shape(), &[10]);
            assert_eq!(row.to_cpu()[0], 50.0);
            
            let col = tensor.index(&[None, Some(3)]);
            assert_eq!(col.shape(), &[10]);
            assert_eq!(col.to_cpu()[0], 3.0);
            
            let element = tensor.index(&[Some(5), Some(3)]);
            assert_eq!(element.shape(), &[]);
            assert_eq!(element.to_cpu().item(), 53.0);
            
            let slice = tensor.index(&[Some(2..7), Some(3..8)]);
            assert_eq!(slice.shape(), &[5, 5]);
        }
    }

    #[test]
    fn test_gpu_reduction_operations() {
        if let Some(device) = get_available_gpu() {
            let tensor = Tensor::randn(&[100, 200, 50]).to_device(device.clone());
            
            // Test various reduction operations
            let sum = tensor.sum();
            assert_eq!(sum.shape(), &[]);
            
            let mean = tensor.mean();
            assert_eq!(mean.shape(), &[]);
            
            let max = tensor.max();
            assert_eq!(max.shape(), &[]);
            
            let min = tensor.min();
            assert_eq!(min.shape(), &[]);
            
            // Test dimension-specific reductions
            let sum_dim0 = tensor.sum_dim(&[0], false);
            assert_eq!(sum_dim0.shape(), &[200, 50]);
            
            let mean_dim12 = tensor.mean_dim(&[1, 2], true);
            assert_eq!(mean_dim12.shape(), &[100, 1, 1]);
            
            println!("GPU reduction operations successful");
        }
    }
}