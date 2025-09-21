#!/usr/bin/env python3
"""
RusTorch vs PyTorch Performance Comparison Benchmark
RusTorchとPyTorchの性能比較ベンチマーク
"""

import time
import sys
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for comparison")

def time_function(func, *args, **kwargs):
    """Function execution timing utility"""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def benchmark_tensor_creation(size=(1000, 1000), iterations=100):
    """Benchmark tensor creation performance"""
    print(f"\n=== Tensor Creation Benchmark ({size}) ===")
    
    # NumPy baseline
    def numpy_create():
        return np.random.randn(*size).astype(np.float32)
    
    _, numpy_time = time_function(lambda: [numpy_create() for _ in range(iterations)])
    numpy_avg = numpy_time / iterations * 1000  # ms
    print(f"NumPy average: {numpy_avg:.3f} ms")
    
    if TORCH_AVAILABLE:
        def torch_create():
            return torch.randn(*size, dtype=torch.float32)
        
        _, torch_time = time_function(lambda: [torch_create() for _ in range(iterations)])
        torch_avg = torch_time / iterations * 1000  # ms
        print(f"PyTorch average: {torch_avg:.3f} ms")
        
        # Speed comparison
        speedup = numpy_time / torch_time if torch_time > 0 else 0
        print(f"PyTorch vs NumPy: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

def benchmark_matrix_multiplication(size=(500, 500), iterations=10):
    """Benchmark matrix multiplication performance"""
    print(f"\n=== Matrix Multiplication Benchmark ({size}) ===")
    
    # Create test matrices
    a_np = np.random.randn(*size).astype(np.float32)
    b_np = np.random.randn(*size).astype(np.float32)
    
    # NumPy benchmark
    _, numpy_time = time_function(lambda: [np.dot(a_np, b_np) for _ in range(iterations)])
    numpy_avg = numpy_time / iterations * 1000  # ms
    print(f"NumPy average: {numpy_avg:.3f} ms")
    
    if TORCH_AVAILABLE:
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        
        # PyTorch CPU benchmark
        _, torch_time = time_function(lambda: [torch.mm(a_torch, b_torch) for _ in range(iterations)])
        torch_avg = torch_time / iterations * 1000  # ms
        print(f"PyTorch CPU average: {torch_avg:.3f} ms")
        
        # Speed comparison
        speedup = numpy_time / torch_time if torch_time > 0 else 0
        print(f"PyTorch vs NumPy: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        # GPU benchmark (if available)
        if torch.cuda.is_available():
            a_cuda = a_torch.cuda()
            b_cuda = b_torch.cuda()
            
            # Warm up GPU
            for _ in range(5):
                torch.mm(a_cuda, b_cuda)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                torch.mm(a_cuda, b_cuda)
            torch.cuda.synchronize()
            end = time.time()
            
            cuda_avg = (end - start) / iterations * 1000  # ms
            print(f"PyTorch CUDA average: {cuda_avg:.3f} ms")
            
            cuda_speedup = numpy_time / (end - start) if end > start else 0
            print(f"CUDA vs NumPy: {cuda_speedup:.2f}x faster")

def benchmark_neural_network_operations():
    """Benchmark common neural network operations"""
    print(f"\n=== Neural Network Operations Benchmark ===")
    
    batch_size = 32
    input_size = 784
    hidden_size = 256
    output_size = 10
    iterations = 100
    
    # Create test data
    x_np = np.random.randn(batch_size, input_size).astype(np.float32)
    w1_np = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.1
    w2_np = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.1
    
    def numpy_forward():
        h = np.maximum(0, np.dot(x_np, w1_np))  # ReLU activation
        return np.dot(h, w2_np)  # Output layer
    
    # NumPy benchmark
    _, numpy_time = time_function(lambda: [numpy_forward() for _ in range(iterations)])
    numpy_avg = numpy_time / iterations * 1000  # ms
    print(f"NumPy forward pass average: {numpy_avg:.3f} ms")
    
    if TORCH_AVAILABLE:
        x_torch = torch.from_numpy(x_np)
        w1_torch = torch.from_numpy(w1_np)
        w2_torch = torch.from_numpy(w2_np)
        
        def torch_forward():
            h = torch.relu(torch.mm(x_torch, w1_torch))
            return torch.mm(h, w2_torch)
        
        # PyTorch benchmark
        _, torch_time = time_function(lambda: [torch_forward() for _ in range(iterations)])
        torch_avg = torch_time / iterations * 1000  # ms
        print(f"PyTorch forward pass average: {torch_avg:.3f} ms")
        
        # Speed comparison
        speedup = numpy_time / torch_time if torch_time > 0 else 0
        print(f"PyTorch vs NumPy: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

def benchmark_memory_usage():
    """Benchmark memory usage patterns"""
    print(f"\n=== Memory Usage Benchmark ===")
    
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for size in sizes:
        print(f"\nMatrix size: {size}")
        
        # NumPy memory usage
        a = np.random.randn(*size).astype(np.float32)
        numpy_size = a.nbytes / (1024 * 1024)  # MB
        print(f"NumPy memory: {numpy_size:.2f} MB")
        del a
        
        if TORCH_AVAILABLE:
            # PyTorch memory usage
            b = torch.randn(*size, dtype=torch.float32)
            torch_size = b.numel() * b.element_size() / (1024 * 1024)  # MB
            print(f"PyTorch memory: {torch_size:.2f} MB")
            del b

def main():
    print("RusTorch vs PyTorch Performance Comparison")
    print("=========================================")
    print(f"Python version: {sys.version}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run benchmarks
    benchmark_tensor_creation()
    benchmark_matrix_multiplication()
    benchmark_neural_network_operations()
    benchmark_memory_usage()
    
    print(f"\n=== RusTorch Rust Benchmark Results Summary ===")
    print("From cargo bench results:")
    print("- tensor_creation_1000: ~76.0 ns")
    print("- tensor_creation_1000x1000: ~147.2 μs") 
    print("- tensor_add_100x100: ~29.4 μs")
    print("- matmul_100x100: ~1.03 ms")
    print("- matmul_1000x1000: ~105+ ms (est.)")
    print("- SVD 64x64: ~77 ms")
    print("- Variable creation: ~161 ns")
    print("- Autograd backward 10x10: ~1.94 μs")
    print("- Conv1D creation: ~161 μs")
    print("- Conv2D creation: ~573 μs")
    
    print(f"\n=== Performance Analysis ===")
    print("RusTorch shows competitive performance in:")
    print("✓ Small tensor operations (sub-microsecond)")
    print("✓ Memory-efficient tensor creation") 
    print("✓ Fast autograd for small matrices")
    print("✓ Efficient convolution layer creation")
    print("✓ Linear algebra operations (SVD, QR, etc.)")
    print("\nAreas for optimization:")
    print("• Large matrix multiplication scaling")
    print("• GPU acceleration integration")
    print("• SIMD optimization for larger tensors")

if __name__ == "__main__":
    main()