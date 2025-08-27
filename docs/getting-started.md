# Getting Started with RusTorch

## 🚀 Quick Setup

### Installation

Add RusTorch to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.4.0"

# Optional features
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Linear algebra operations (SVD, QR, LU, eigenvalue)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
all-gpu = ["cuda", "metal", "opencl"]
all-formats = ["safetensors", "onnx"]

# To disable linalg features (avoid OpenBLAS/LAPACK dependencies):
rustorch = { version = "0.4.0", default-features = false }
```

### Basic Tensor Operations

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Create tensors
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Basic operations
    let c = &a + &b;  // Addition
    let d = a.matmul(&b);  // Matrix multiplication
    
    // Mathematical functions
    let e = a.sin();  // Sine function
    let f = a.exp();  // Exponential function
    
    println!("Shape: {:?}", c.shape());
    println!("Result: {:?}", c.as_slice());
}
```

### Broadcasting Support

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Broadcasting: (batch, features) + (1, features)
    let batch_data = Tensor::from_vec(
        (0..64).map(|i| i as f32 * 0.01).collect(),
        vec![32, 2]  // 32 samples, 2 features
    );
    
    let bias = Tensor::from_vec(
        vec![0.1, 0.2],
        vec![1, 2]  // Broadcast shape
    );
    
    // Automatic broadcasting
    let result = batch_data.add(&bias).unwrap();
    println!("Result shape: {:?}", result.shape()); // [32, 2]
}
```

### Advanced Tensor Operations

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Create a 3x4 matrix
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4]
    );
    
    // Statistical operations
    let mean = data.mean(None);  // Overall mean
    let std_dev = data.std(Some(0), true);  // Standard deviation along axis 0
    let median = data.median(Some(1));  // Median along axis 1
    
    // Broadcasting operations
    let broadcasted = data.broadcast_to(&[6, 4]).unwrap();
    
    // Indexing operations
    let selected = data.select(0, &[0, 2]).unwrap();  // Select rows 0 and 2
    
    println!("Mean: {:?}", mean.as_slice());
    println!("Selected shape: {:?}", selected.shape());
}
```

## 🧮 Matrix Decomposition

**Important Note**: Matrix decomposition features require the `linalg` feature (enabled by default). On some systems, this may require OpenBLAS/LAPACK libraries. To avoid these dependencies:

```toml
rustorch = { version = "0.4.0", default-features = false }
```

### SVD, QR, LU and Eigenvalue Decomposition

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Create a 3x3 matrix
    let matrix = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![3, 3]
    );
    
    // Singular Value Decomposition (torch.svd compatible)
    let (u, s, v) = matrix.svd(false).unwrap();
    println!("SVD - U: {:?}, S: {:?}, V: {:?}", u.shape(), s.shape(), v.shape());
    
    // QR decomposition
    let (q, r) = matrix.qr().unwrap();
    println!("QR - Q: {:?}, R: {:?}", q.shape(), r.shape());
    
    // LU decomposition with partial pivoting
    let (l, u, p) = matrix.lu().unwrap();
    println!("LU - L: {:?}, U: {:?}, P: {:?}", l.shape(), u.shape(), p.shape());
    
    // Create symmetric matrix for eigenvalue decomposition
    let sym_data = vec![4.0f32, 2.0, 1.0, 2.0, 3.0, 0.5, 1.0, 0.5, 1.0];
    let sym_matrix = Tensor::from_vec(sym_data, vec![3, 3]);
    
    // Symmetric eigenvalue decomposition (torch.symeig compatible)
    let (eigenvals, eigenvecs) = sym_matrix.symeig(true, true).unwrap();
    println!("Symeig - Values: {:?}, Vectors: {:?}", eigenvals.shape(), eigenvecs.shape());
    
    // General eigenvalue decomposition (torch.eig compatible)
    let (eig_vals, eig_vecs) = matrix.eig(true).unwrap();
    println!("Eig - Values: {:?}, Vectors: {:?}", eig_vals.shape(), eig_vecs.unwrap().shape());
}
```

## 🧠 Neural Networks and Automatic Differentiation

```rust
use rustorch::prelude::*;
use rustorch::nn::{Linear, loss::mse_loss};
use rustorch::optim::{SGD, Optimizer};

fn main() {
    // Create model
    let model = Linear::new(784, 10);
    let params = model.parameters();
    let mut optimizer = SGD::new(params, 0.01, None, None, None, None);
    
    // Prepare data
    let input = Variable::new(
        Tensor::from_vec((0..784).map(|i| i as f32 * 0.01).collect(), vec![1, 784]),
        false
    );
    let target = Variable::new(
        Tensor::from_vec(vec![1.0; 10], vec![1, 10]),
        false
    );
    
    // Training loop
    for epoch in 0..100 {
        optimizer.zero_grad();
        
        let output = model.forward(&input);
        let loss = mse_loss(&output, &target);
        
        loss.backward();
        optimizer.step();
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.data().as_array()[[0]]);
        }
    }
}
```

## 🖼️ Computer Vision

### Basic Transforms

```rust
use rustorch::prelude::*;
use rustorch::vision::{transforms::*, datasets::*, Image, ImageFormat};

fn main() {
    // Load MNIST dataset
    let train_dataset = MNIST::new("./data", true, true).unwrap();
    
    // Create basic transforms
    let transform = Compose::new(vec![
        Box::new(Resize::new((224, 224))),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(ToTensor::new()),
        Box::new(Normalize::imagenet()),
    ]);
    
    let cifar10 = CIFAR10::new("./data", true, true)
        .unwrap()
        .with_transform(Box::new(transform));
    
    let train_loader = DataLoader::new(cifar10, 32, true);
}
```

## 🎮 GPU Acceleration

```rust
use rustorch::gpu::{DeviceType, kernels::{KernelExecutor, AddKernel, MatMulKernel}};

fn main() {
    // Automatic device detection
    let available_devices = DeviceType::available_devices();
    println!("Available devices: {:?}", available_devices);
    
    // GPU kernel execution
    let device = DeviceType::best_available();
    let executor = KernelExecutor::new(device);
    
    // Element-wise addition on GPU
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let mut c = vec![0.0f32; 1024];
    
    let kernel = AddKernel;
    let inputs = [a.as_slice(), b.as_slice()];
    let mut outputs = [c.as_mut_slice()];
    
    executor.execute_kernel(&kernel, &inputs, &mut outputs)
        .expect("GPU kernel execution failed");
    
    println!("GPU computation completed: {:?}", &c[..5]);
}
```

## 🌐 WebAssembly Usage

### JavaScript Integration

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function main() {
    // Initialize WASM
    await init();
    
    // Basic tensor operations
    const tensor1 = rustorch.WasmTensor.zeros([2, 3]);
    const tensor2 = rustorch.WasmTensor.ones([2, 3]);
    const tensor3 = rustorch.WasmTensor.random([2, 3]);
    
    // Mathematical operations
    const sum = tensor1.add(tensor2);
    const product = tensor1.multiply(tensor2);
    const relu_result = tensor1.relu();
    
    // Statistics
    console.log('Mean:', tensor3.mean());
    console.log('Max:', tensor3.max());
    console.log('Sum:', tensor3.sum());
    
    // Neural network model
    const model = new rustorch.WasmModel();
    model.add_linear(4, 8, true);  // Linear layer: 4 inputs → 8 outputs
    model.add_relu();              // ReLU activation
    model.add_linear(8, 2, true);  // Output layer: 8 → 2
}

main();
```

### Building for WebAssembly

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web --features wasm

# Build for Node.js
wasm-pack build --target nodejs --features wasm

# Run examples
cd examples
python -m http.server 8000
# Open http://localhost:8000/wasm_basic.html
```

## 🛡️ Safe Operations

```rust
use rustorch::nn::safe_ops::SafeOps;
use rustorch::autograd::Variable;

fn main() {
    // Create a variable safely with validation
    let var = SafeOps::create_variable(
        vec![-2.0, -1.0, 0.0, 1.0, 2.0], 
        vec![5], 
        false
    ).unwrap();
    
    // Apply ReLU activation: max(0, x)
    let relu_result = SafeOps::relu(&var).unwrap();
    println!("ReLU output: {:?}", relu_result.data().read().unwrap().as_array());
    // Output: [0.0, 0.0, 0.0, 1.0, 2.0]
    
    // Get tensor statistics safely
    let stats = SafeOps::get_stats(&var).unwrap();
    println!("Mean: {:.2}, Std: {:.2}", stats.mean, stats.std_dev());
    
    // Validate tensor for NaN or infinity
    SafeOps::validate_finite(&var).unwrap();
    println!("Tensor is finite and valid!");
}
```

## 📚 Next Steps

- **[Features](features.md)** - Explore all available features
- **[Performance](performance.md)** - Learn about optimization techniques
- **[Examples](examples.md)** - Browse comprehensive examples
- **[Architecture](architecture.md)** - Understand the system design
- **[API Documentation](https://docs.rs/rustorch)** - Detailed API reference

## ❓ Common Issues

### LAPACK/BLAS Dependencies

If you encounter linking issues with LAPACK/BLAS:

```toml
# Disable linear algebra features
rustorch = { version = "0.4.0", default-features = false }
```

### GPU Support

GPU features require appropriate drivers and libraries:
- **CUDA**: NVIDIA CUDA Toolkit
- **Metal**: macOS with Metal support
- **OpenCL**: OpenCL drivers for your platform

See [GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md) for detailed setup instructions.