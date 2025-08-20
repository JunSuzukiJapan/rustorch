use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use rustorch::gpu::{DeviceType, kernels::{AddKernel, MatMulKernel, KernelExecutor, GpuKernel}};

fn bench_gpu_elementwise_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_elementwise_add");
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144];
    let devices = vec![
        DeviceType::Cpu,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(0),
        #[cfg(feature = "metal")]
        DeviceType::Metal(0),
        #[cfg(feature = "opencl")]
        DeviceType::OpenCl(0),
    ];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        for device in &devices {
            // Skip GPU devices if they're not available
            if !device.is_available() {
                continue;
            }
            
            let device_name = match device {
                DeviceType::Cpu => "CPU",
                DeviceType::Cuda(_) => "CUDA",
                DeviceType::Metal(_) => "Metal",
                DeviceType::OpenCl(_) => "OpenCL",
            };
            
            group.bench_with_input(
                BenchmarkId::new(device_name, size),
                &size,
                |bencher, &size| {
                    let a = vec![1.0f32; size];
                    let b = vec![2.0f32; size];
                    let mut c = vec![0.0f32; size];
                    
                    let executor = KernelExecutor::new(*device);
                    let kernel = AddKernel;
                    
                    bencher.iter(|| {
                        let inputs = [a.as_slice(), b.as_slice()];
                        let mut outputs = [c.as_mut_slice()];
                        
                        executor.execute_kernel(&kernel, &inputs, &mut outputs)
                            .expect("Kernel execution failed");
                        
                        black_box(&c);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_gpu_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_matrix_multiplication");
    
    let matrix_sizes = vec![64, 128, 256, 512];
    let devices = vec![
        DeviceType::Cpu,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(0),
        #[cfg(feature = "metal")]
        DeviceType::Metal(0),
        #[cfg(feature = "opencl")]
        DeviceType::OpenCl(0),
    ];
    
    for n in matrix_sizes {
        let size = n * n;
        group.throughput(Throughput::Elements((2 * n * n * n) as u64)); // FLOPS for matrix multiplication
        
        for device in &devices {
            // Skip GPU devices if they're not available
            if !device.is_available() {
                continue;
            }
            
            let device_name = match device {
                DeviceType::Cpu => "CPU",
                DeviceType::Cuda(_) => "CUDA",
                DeviceType::Metal(_) => "Metal",
                DeviceType::OpenCl(_) => "OpenCL",
            };
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}-{}", n, n, device_name), size),
                &size,
                |bencher, &_size| {
                    let a = vec![1.0f32; size];
                    let b = vec![2.0f32; size];
                    let mut c = vec![0.0f32; size];
                    
                    let executor = KernelExecutor::new(*device);
                    let kernel = MatMulKernel;
                    
                    bencher.iter(|| {
                        let inputs = [a.as_slice(), b.as_slice()];
                        let mut outputs = [c.as_mut_slice()];
                        
                        executor.execute_kernel(&kernel, &inputs, &mut outputs)
                            .expect("Matrix multiplication failed");
                        
                        black_box(&c);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_gpu_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_transfer");
    
    let sizes = vec![1024, 4096, 16384, 65536, 262144];
    
    for size in sizes {
        group.throughput(Throughput::Bytes((size * std::mem::size_of::<f32>()) as u64));
        
        // CUDA memory transfer benchmark
        #[cfg(feature = "cuda")]
        {
            use rustorch::gpu::cuda_kernels::{CudaBuffer, CudaKernelExecutor};
            
            if CudaKernelExecutor::new(0).is_ok() {
                group.bench_with_input(
                    BenchmarkId::new("CUDA-H2D", size),
                    &size,
                    |b, &size| {
                        let data = vec![1.0f32; size];
                        
                        b.iter(|| {
                            let buffer = CudaBuffer::from_host_data(&data)
                                .expect("Failed to create CUDA buffer");
                            black_box(buffer);
                        });
                    },
                );
                
                group.bench_with_input(
                    BenchmarkId::new("CUDA-D2H", size),
                    &size,
                    |bencher, &size| {
                        let data = vec![1.0f32; size];
                        let buffer = CudaBuffer::from_host_data(&data)
                            .expect("Failed to create CUDA buffer");
                        
                        bencher.iter(|| {
                            let mut result = vec![0.0f32; size];
                            buffer.copy_to_host(&mut result)
                                .expect("Failed to copy from device");
                            black_box(result);
                        });
                    },
                );
            }
        }
        
        // Metal memory transfer benchmark
        #[cfg(feature = "metal")]
        {
            use rustorch::gpu::metal_kernels::{MetalBuffer, MetalKernelExecutor};
            
            if MetalKernelExecutor::new().is_ok() {
                group.bench_with_input(
                    BenchmarkId::new("Metal-H2D", size),
                    &size,
                    |b, &size| {
                        let data = vec![1.0f32; size];
                        
                        b.iter(|| {
                            let buffer = MetalBuffer::from_host_data(&data)
                                .expect("Failed to create Metal buffer");
                            black_box(buffer);
                        });
                    },
                );
                
                group.bench_with_input(
                    BenchmarkId::new("Metal-D2H", size),
                    &size,
                    |bencher, &size| {
                        let data = vec![1.0f32; size];
                        let buffer = MetalBuffer::from_host_data(&data)
                            .expect("Failed to create Metal buffer");
                        
                        bencher.iter(|| {
                            let mut result = vec![0.0f32; size];
                            buffer.copy_to_host(&mut result)
                                .expect("Failed to copy from device");
                            black_box(result);
                        });
                    },
                );
            }
        }
        
        // OpenCL memory transfer benchmark
        #[cfg(feature = "opencl")]
        {
            use rustorch::gpu::opencl_kernels::{OpenClBuffer, OpenClKernelExecutor};
            
            if OpenClKernelExecutor::new(0).is_ok() {
                group.bench_with_input(
                    BenchmarkId::new("OpenCL-H2D", size),
                    &size,
                    |b, &size| {
                        let data = vec![1.0f32; size];
                        
                        b.iter(|| {
                            let buffer = OpenClBuffer::from_host_data(&data)
                                .expect("Failed to create OpenCL buffer");
                            black_box(buffer);
                        });
                    },
                );
                
                group.bench_with_input(
                    BenchmarkId::new("OpenCL-D2H", size),
                    &size,
                    |bencher, &size| {
                        let data = vec![1.0f32; size];
                        let buffer = OpenClBuffer::from_host_data(&data)
                            .expect("Failed to create OpenCL buffer");
                        
                        bencher.iter(|| {
                            let mut result = vec![0.0f32; size];
                            buffer.copy_to_host(&mut result)
                                .expect("Failed to copy from device");
                            black_box(result);
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

fn bench_gpu_kernel_launch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_kernel_launch_overhead");
    
    let devices = vec![
        DeviceType::Cpu,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(0),
        #[cfg(feature = "metal")]
        DeviceType::Metal(0),
        #[cfg(feature = "opencl")]
        DeviceType::OpenCl(0),
    ];
    
    // Small problem size to measure launch overhead
    let size = 1024;
    
    for device in &devices {
        // Skip GPU devices if they're not available
        if !device.is_available() {
            continue;
        }
        
        let device_name = match device {
            DeviceType::Cpu => "CPU",
            DeviceType::Cuda(_) => "CUDA",
            DeviceType::Metal(_) => "Metal",
            DeviceType::OpenCl(_) => "OpenCL",
        };
        
        group.bench_function(
            BenchmarkId::new("launch_overhead", device_name),
            |b| {
                let a = vec![1.0f32; size];
                let b_vec = vec![2.0f32; size];
                let mut c = vec![0.0f32; size];
                
                let executor = KernelExecutor::new(*device);
                let kernel = AddKernel;
                
                b.iter(|| {
                    let inputs = [a.as_slice(), b_vec.as_slice()];
                    let mut outputs = [c.as_mut_slice()];
                    
                    executor.execute_kernel(&kernel, &inputs, &mut outputs)
                        .expect("Kernel execution failed");
                    
                    black_box(&c);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_gpu_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_batch_operations");
    
    let batch_sizes = vec![1, 4, 16, 64];
    let element_size = 4096;
    let devices = vec![
        DeviceType::Cpu,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(0),
        #[cfg(feature = "metal")]
        DeviceType::Metal(0),
        #[cfg(feature = "opencl")]
        DeviceType::OpenCl(0),
    ];
    
    for batch_size in batch_sizes {
        let total_elements = batch_size * element_size;
        group.throughput(Throughput::Elements(total_elements as u64));
        
        for device in &devices {
            // Skip GPU devices if they're not available
            if !device.is_available() {
                continue;
            }
            
            let device_name = match device {
                DeviceType::Cpu => "CPU",
                DeviceType::Cuda(_) => "CUDA",
                DeviceType::Metal(_) => "Metal",
                DeviceType::OpenCl(_) => "OpenCL",
            };
            
            group.bench_with_input(
                BenchmarkId::new(format!("batch_{}-{}", batch_size, device_name), total_elements),
                &total_elements,
                |bencher, &_total_elements| {
                    let a = vec![1.0f32; total_elements];
                    let b_vec = vec![2.0f32; total_elements];
                    let mut c = vec![0.0f32; total_elements];
                    
                    let executor = KernelExecutor::new(*device);
                    let kernel = AddKernel;
                    
                    bencher.iter(|| {
                        // Simulate batch processing by running multiple smaller operations
                        for i in 0..batch_size {
                            let start = i * element_size;
                            let end = start + element_size;
                            
                            let inputs = [&a[start..end], &b_vec[start..end]];
                            let mut outputs = [&mut c[start..end]];
                            
                            executor.execute_kernel(&kernel, &inputs, &mut outputs)
                                .expect("Batch kernel execution failed");
                        }
                        
                        black_box(&c);
                    });
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    gpu_kernel_benches,
    bench_gpu_elementwise_add,
    bench_gpu_matrix_multiplication,
    bench_gpu_memory_transfer,
    bench_gpu_kernel_launch_overhead,
    bench_gpu_batch_operations
);

criterion_main!(gpu_kernel_benches);
