# RusTorch API Documentation

## 📚 Complete API Reference

This document provides comprehensive API documentation for RusTorch v0.5.15, organized by module and functionality. Features unified error handling with `RusTorchError` and `RusTorchResult<T>` for consistent error management across all 1060+ tests. **Phase 8 COMPLETED** adds advanced tensor utilities including conditional operations, indexing, and statistical functions. **Phase 9 COMPLETED** introduces comprehensive serialization system with model save/load, JIT compilation, and multiple format support including PyTorch compatibility.

## 🏗️ Core Architecture

### Module Structure

```
rustorch/
├── tensor/              # Core tensor operations and data structures
├── nn/                  # Neural network layers and functions
├── autograd/            # Automatic differentiation engine
├── optim/               # Optimizers and learning rate schedulers
├── special/             # Special mathematical functions
├── distributions/       # Statistical distributions
├── vision/              # Computer vision transforms
├── linalg/              # Linear algebra operations (BLAS/LAPACK)
├── gpu/                 # GPU acceleration (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Sparse tensor operations and pruning (Phase 12)
├── serialization/       # Model serialization and JIT compilation (Phase 9)
└── wasm/                # WebAssembly bindings (see [WASM API Documentation](WASM_API_DOCUMENTATION.md))
```

## 📊 Tensor Module

### Core Tensor Creation

```rust
use rustorch::tensor::Tensor;

// Basic creation
let tensor = Tensor::new(vec![2, 3]);               // Shape-based creation
let tensor = Tensor::from_vec(data, vec![2, 3]);    // From data vector
let tensor = Tensor::zeros(vec![10, 10]);           // Zero-filled tensor
let tensor = Tensor::ones(vec![5, 5]);              // One-filled tensor
let tensor = Tensor::randn(vec![3, 3]);             // Random normal distribution
let tensor = Tensor::rand(vec![3, 3]);              // Random uniform [0,1)
let tensor = Tensor::eye(5);                        // Identity matrix
let tensor = Tensor::full(vec![2, 2], 3.14);       // Fill with specific value
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Range tensor
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Linear spacing
```

### Tensor Operations

```rust
// Arithmetic operations
let result = a.add(&b);                             // Element-wise addition
let result = a.sub(&b);                             // Element-wise subtraction
let result = a.mul(&b);                             // Element-wise multiplication
let result = a.div(&b);                             // Element-wise division
let result = a.pow(&b);                             // Element-wise power
let result = a.rem(&b);                             // Element-wise remainder

// Matrix operations
let result = a.matmul(&b);                          // Matrix multiplication
let result = a.transpose();                         // Matrix transpose
let result = a.dot(&b);                             // Dot product

// Mathematical functions
let result = tensor.exp();                          // Exponential
let result = tensor.ln();                           // Natural logarithm
let result = tensor.log10();                        // Base-10 logarithm
let result = tensor.sqrt();                         // Square root
let result = tensor.abs();                          // Absolute value
let result = tensor.sin();                          // Sine function
let result = tensor.cos();                          // Cosine function
let result = tensor.tan();                          // Tangent function
let result = tensor.asin();                         // Arcsine
let result = tensor.acos();                         // Arccosine
let result = tensor.atan();                         // Arctangent
let result = tensor.sinh();                         // Hyperbolic sine
let result = tensor.cosh();                         // Hyperbolic cosine
let result = tensor.tanh();                         // Hyperbolic tangent
let result = tensor.floor();                        // Floor function
let result = tensor.ceil();                         // Ceiling function
let result = tensor.round();                        // Round function
let result = tensor.sign();                         // Sign function
let result = tensor.max();                          // Maximum value
let result = tensor.min();                          // Minimum value
let result = tensor.sum();                          // Sum all elements
let result = tensor.mean();                         // Mean value
let result = tensor.std();                          // Standard deviation
let result = tensor.var();                          // Variance

// Shape manipulation
let result = tensor.reshape(vec![6, 4]);            // Reshape tensor
let result = tensor.squeeze();                      // Remove size-1 dimensions
let result = tensor.unsqueeze(1);                   // Add dimension at index
let result = tensor.permute(vec![1, 0, 2]);         // Permute dimensions
let result = tensor.expand(vec![10, 10, 5]);        // Expand tensor dimensions

// Advanced shape operations (Phase 1)
let result = tensor.squeeze_dim(1);                 // Remove specific size-1 dimension
let result = tensor.flatten_owned();                // Flatten to 1D tensor
let result = tensor.flatten_range(1, Some(3));      // Flatten dimensions 1-3
let result = tensor.unflatten(0, &[2, 3]);         // Reverse flatten operation
let result = tensor.expand_as(&other_tensor);       // Expand to match another tensor
let result = tensor.repeat(&[2, 3, 1]);            // Repeat tensor along dimensions
let result = tensor.repeat_interleave_scalar(3, Some(0)); // Interleave elements
let result = tensor.roll_1d(2, Some(1));           // Roll elements along dimension
let result = tensor.rot90(1, &[0, 1]);             // Rotate 90 degrees
let result = tensor.flip(&[0]);                    // Flip along dimensions
let result = tensor.fliplr();                      // Flip left-right
let result = tensor.flipud();                      // Flip up-down
let result = tensor.view_shape(&[6, 4]);           // Create view with different shape

// Builder Pattern for Chainable Operations (NEW)
use rustorch::tensor::ops::shape_operations::{ShapeOps, shape_ops};

// Method 1: Builder pattern with explicit calls
let result = tensor
    .shape_builder()
    .squeeze().unwrap()                          // Remove singleton dimensions  
    .unsqueeze(1).unwrap()                       // Add dimension at index 1
    .flatten().unwrap()                          // Flatten to 1D
    .build();

// Method 2: Fluent interface
let result = tensor
    .shapes()                                    // Start fluent operations
    .squeeze().unwrap()
    .expand(&[10, 5]).unwrap()
    .flip(&[0]).unwrap()
    .build();

// Method 3: Macro for concise chaining
let result = shape_ops!(tensor,
    squeeze,                                     // Operations without parameters
    unsqueeze(1),                                // Operations with parameters
    flatten
).unwrap();

// Builder pattern supports all shape operations:
let advanced_result = tensor
    .shape_builder()
    .squeeze_dim(2).unwrap()                     // Remove specific dimension
    .repeat(&[2, 1, 3]).unwrap()                // Repeat along dimensions
    .rot90(1, &[0, 1]).unwrap()                 // Rotate 90 degrees
    .flip(&[0, 2]).unwrap()                     // Flip along multiple dimensions
    .build();

// Peek at intermediate results without consuming builder
let builder = tensor.shape_builder().squeeze().unwrap();
println!("Intermediate shape: {:?}", builder.current_shape());
let final_result = builder.flatten().unwrap().build();

// Indexing and slicing
let result = tensor.slice(0, 1, 3);                 // Slice along dimension
let result = tensor.index_select(0, &indices);      // Select indices
let result = tensor.masked_select(&mask);           // Boolean masking
let result = tensor.gather(1, &indices);            // Gather operation

// Comparison operations
let result = a.eq(&b);                              // Element-wise equality
let result = a.ne(&b);                              // Element-wise not equal
let result = a.lt(&b);                              // Element-wise less than
let result = a.le(&b);                              // Element-wise less or equal
let result = a.gt(&b);                              // Element-wise greater than
let result = a.ge(&b);                              // Element-wise greater or equal

// Logical operations
let result = a.logical_and(&b);                     // Logical AND
let result = a.logical_or(&b);                      // Logical OR
let result = a.logical_not();                       // Logical NOT
let result = a.logical_xor(&b);                     // Logical XOR

// Reduction operations
let result = tensor.sum_dim(1, false);              // Sum along dimension
let result = tensor.mean_dim(0, false);             // Mean along dimension
let result = tensor.max_dim(2, false);              // Max along dimension
let result = tensor.min_dim(2, false);              // Min along dimension
let result = tensor.prod();                         // Product of all elements
let result = tensor.prod_dim(1, false);             // Product along dimension

// Sorting operations
let (sorted, indices) = tensor.sort(0, false);     // Sort along dimension
let result = tensor.argsort(0, false);              // Sort indices
let (values, indices) = tensor.topk(5, 0, true);   // Top-k values

// Random operations
let result = tensor.uniform(0.0, 1.0);              // Uniform distribution
let result = tensor.normal(0.0, 1.0);               // Normal distribution
let result = tensor.bernoulli(0.5);                 // Bernoulli distribution
```

### Phase 8: Advanced Tensor Utilities

Phase 8 introduces powerful tensor utility operations for conditional selection, indexing, and statistical analysis.

```rust
use rustorch::tensor::Tensor;
use ndarray::ArrayD;

// Conditional Operations
let mask = ArrayD::from_shape_vec(vec![2, 3], vec![true, false, true, true, false, false])?;
let selected = tensor.masked_select(&mask)?;           // Select elements where mask is true
let filled = tensor.masked_fill(&mask, 999.0)?;       // Fill masked positions with value

let condition = ArrayD::from_shape_vec(vec![2, 3], vec![true, false, true, false, true, false])?;
let result = tensor.where_(&condition, &other_tensor)?; // Select from two tensors based on condition

// Index Operations
let index = ArrayD::from_shape_vec(vec![2], vec![0i64, 2])?;
let gathered = tensor.gather(1, &index)?;             // Gather values along axis using indices
let selected_idx = tensor.index_select(1, &index)?;   // Select values along axis using index
let scattered = tensor.scatter(1, &index, &src_tensor)?; // Scatter values to positions

// Statistical Operations
let (top_values, top_indices) = tensor.topk_util(5, 1, true, true)?; // Top-k elements
let (kth_val, kth_idx) = tensor.kthvalue(3, 1, false)?;             // k-th smallest/largest value

// Advanced Operations
let (unique_vals, inverse, counts) = tensor.unique(true, true, true)?; // Unique elements with options
let (hist_counts, hist_edges) = tensor.histogram(10, Some((0.0, 1.0)))?; // Histogram computation
```

### Complex Number Support

```rust
use rustorch::tensor::{ComplexTensor, Complex64};

// Complex tensor creation
let complex_tensor = ComplexTensor::new(vec![2, 2]);
let from_real_imag = ComplexTensor::from_real_imag(&real_tensor, &imag_tensor);

// Complex operations
let result = complex_tensor.conj();                 // Complex conjugate
let real_part = complex_tensor.real();              // Real part
let imag_part = complex_tensor.imag();              // Imaginary part
let magnitude = complex_tensor.abs();               // Magnitude
let phase = complex_tensor.angle();                 // Phase angle
```

## 🧠 Neural Network Module

### Linear Layers

```rust
use rustorch::nn::Linear;

// Basic linear layer
let linear = Linear::<f32>::new(784, 128);          // Input: 784, Output: 128
let output = linear.forward(&input);                // Forward pass

// Linear layer without bias
let linear = Linear::<f32>::with_bias(784, 128, false);
```

### Convolutional Layers

```rust
use rustorch::nn::{Conv1d, Conv2d, Conv3d, ConvTranspose2d};

// 2D Convolution
let conv2d = Conv2d::<f32>::new(3, 64, (3, 3));    // in_channels, out_channels, kernel_size
let conv2d_full = Conv2d::<f32>::with_params(
    3, 64, (3, 3),                                  // channels and kernel
    Some((1, 1)),                                   // stride
    Some((1, 1)),                                   // padding
    Some((1, 1)),                                   // dilation
    1,                                              // groups
    true                                            // bias
);

// 1D and 3D Convolutions
let conv1d = Conv1d::<f32>::new(1, 16, 5);         // 1D convolution
let conv3d = Conv3d::<f32>::new(1, 8, (3, 3, 3));  // 3D convolution

// Transpose convolution (deconvolution)
let conv_transpose = ConvTranspose2d::<f32>::new(64, 3, (4, 4));
```

### Pooling Layers

```rust
use rustorch::nn::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d};

// Pooling layers
let maxpool = MaxPool2d::<f32>::new((2, 2));       // 2x2 max pooling
let avgpool = AvgPool2d::<f32>::new((2, 2));       // 2x2 average pooling

// Adaptive pooling
let adaptive_avg = AdaptiveAvgPool2d::<f32>::new((7, 7));  // Output size 7x7
let adaptive_max = AdaptiveMaxPool2d::<f32>::new((1, 1));  // Global pooling
```

### Recurrent Layers

```rust
use rustorch::nn::{RNN, LSTM, GRU};

// LSTM layer
let lstm = LSTM::<f32>::new(100, 256, 2, true, true, 0.5); // input_size, hidden_size, num_layers, bias, batch_first, dropout
let (output, (hidden, cell)) = lstm.forward(&input, &(h0, c0));

// GRU layer
let gru = GRU::<f32>::new(100, 256, 2, true, true, 0.5);
let (output, hidden) = gru.forward(&input, &h0);

// Basic RNN
let rnn = RNN::<f32>::new(100, 256, 2, "tanh", true, true, 0.5);
```

### Activation Functions

```rust
use rustorch::nn;

// Function-style activations (recommended)
let relu_out = nn::relu(&input);                    // ReLU activation
let sigmoid_out = nn::sigmoid(&input);              // Sigmoid activation
let tanh_out = nn::tanh(&input);                    // Tanh activation
let softmax_out = nn::softmax(&input, 1);          // Softmax along dimension 1
let log_softmax_out = nn::log_softmax(&input, 1);  // Log-softmax
let elu_out = nn::elu(&input, 1.0);                // ELU with alpha
let selu_out = nn::selu(&input);                    // SELU activation
let swish_out = nn::swish(&input);                  // Swish activation
let mish_out = nn::mish(&input);                    // Mish activation
let gelu_out = nn::gelu(&input);                    // GELU activation

// Module-style activations
use rustorch::nn::{ReLU, LeakyReLU, ELU, SELU, Softmax, LogSoftmax};

let relu = ReLU::<f32>::new();
let leaky_relu = LeakyReLU::<f32>::new(0.01);      // Negative slope
let elu = ELU::<f32>::new(1.0);                    // Alpha parameter
let selu = SELU::<f32>::new();
let softmax = Softmax::<f32>::new(1);              // Dimension
let log_softmax = LogSoftmax::<f32>::new(1);       // Dimension
```

### Normalization Layers

```rust
use rustorch::nn::{BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, GroupNorm, InstanceNorm2d};

// Batch normalization
let bn1d = BatchNorm1d::<f32>::new(128);           // num_features
let bn2d = BatchNorm2d::<f32>::new(64);            // num_features
let bn3d = BatchNorm3d::<f32>::new(32);            // num_features

// Layer normalization
let layer_norm = LayerNorm::<f32>::new(vec![128]); // normalized_shape

// Group normalization
let group_norm = GroupNorm::<f32>::new(8, 64);     // num_groups, num_channels

// Instance normalization
let instance_norm = InstanceNorm2d::<f32>::new(64); // num_features
```

### Regularization Layers

```rust
use rustorch::nn::{Dropout, Dropout2d, Dropout3d, AlphaDropout};

// Dropout layers
let dropout = Dropout::<f32>::new(0.5);            // Drop probability
let dropout2d = Dropout2d::<f32>::new(0.2);        // 2D dropout
let dropout3d = Dropout3d::<f32>::new(0.1);        // 3D dropout
let alpha_dropout = AlphaDropout::<f32>::new(0.5); // Alpha dropout for SELU
```

### Loss Functions

```rust
use rustorch::nn::loss;

// Classification losses
let mse_loss = loss::mse_loss(&predictions, &targets);
let cross_entropy = loss::cross_entropy(&logits, &targets);
let nll_loss = loss::nll_loss(&log_probs, &targets);
let binary_cross_entropy = loss::binary_cross_entropy(&predictions, &targets);

// Regression losses
let l1_loss = loss::l1_loss(&predictions, &targets);
let smooth_l1_loss = loss::smooth_l1_loss(&predictions, &targets);
let huber_loss = loss::huber_loss(&predictions, &targets, 1.0);

// Advanced losses
let kl_div_loss = loss::kl_div(&input, &target);
let cosine_embedding_loss = loss::cosine_embedding_loss(&input1, &input2, &target);
```

## ⚡ Automatic Differentiation Module

### Variable and Gradient Computation

```rust
use rustorch::autograd::{Variable, backward, grad};

// Create variable with gradient tracking
let var = Variable::<f32>::new(tensor, true);       // requires_grad = true
let var = Variable::<f32>::with_grad(tensor);       // Always requires grad

// Gradient computation
let gradients = grad(&[output], &[var], true, true, None);
backward(&[output], &[gradient_tensor], true, true);

// Access gradients
if let Some(grad_tensor) = var.grad() {
    println!("Gradient: {:?}", grad_tensor);
}

// Gradient control
var.zero_grad();                                    // Clear gradients
var.detach();                                       // Detach from computation graph
```

### Custom Autograd Functions

```rust
use rustorch::autograd::{Function, FunctionCtx};

struct CustomSquare;

impl Function<f32> for CustomSquare {
    fn forward(ctx: &mut FunctionCtx<f32>, input: &Tensor<f32>) -> Tensor<f32> {
        ctx.save_for_backward(&[input.clone()]);
        input.pow(&Tensor::from_scalar(2.0))
    }
    
    fn backward(ctx: &FunctionCtx<f32>, grad_output: &Tensor<f32>) -> Vec<Tensor<f32>> {
        let saved = ctx.get_saved_tensors();
        let input = &saved[0];
        vec![grad_output * (input * 2.0)]
    }
}
```

## 🔧 Optimization Module

### Optimizers

#### Core Optimizers

```rust
use rustorch::optim::{SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta};

// SGD optimizer
let sgd = SGD::<f32>::new(parameters, 0.01);        // learning_rate
let sgd_momentum = SGD::<f32>::with_momentum(parameters, 0.01, 0.9); // lr, momentum
let sgd_full = SGD::<f32>::with_params(parameters, 0.01, 0.9, 0.0001, false); // lr, momentum, weight_decay, nesterov

// Adam optimizer
let adam = Adam::<f32>::new(parameters, 0.001);     // learning_rate
let adam_full = Adam::<f32>::with_params(
    parameters, 0.001,                               // learning_rate
    0.9, 0.999,                                     // betas (beta1, beta2)
    1e-8,                                           // eps
    0.0,                                            // weight_decay
    false                                           // amsgrad
);

// AdamW optimizer
let adamw = AdamW::<f32>::new(parameters, 0.001);   // learning_rate
let adamw_full = AdamW::<f32>::with_params(parameters, 0.001, 0.9, 0.999, 1e-8, 0.01);

// RMSprop optimizer
let rmsprop = RMSprop::<f32>::new(parameters, 0.01); // learning_rate
let rmsprop_full = RMSprop::<f32>::with_params(parameters, 0.01, 0.99, 1e-8, 0.0, false, false);

// Other optimizers
let adagrad = Adagrad::<f32>::new(parameters, 0.01); // learning_rate
let adadelta = Adadelta::<f32>::new(parameters);     // Default parameters
```

#### Phase 2: Advanced Adam Variants (NEW in v0.5.13)

**🚀 GenericAdamOptimizer Architecture**: Phase 2 introduces a unified architecture that provides consistent API and improved performance for all Adam-based optimizers.

```rust
use rustorch::optim::{NAdam, RAdam, Adamax, LBFGS, LineSearchMethod};
use rustorch::optim::common::{AdamConfig, AdamState, AdamUtils};
```

##### NAdam (Nesterov-accelerated Adam)

```rust
// Basic usage - excellent for most deep learning tasks
let nadam = NAdam::default_params(0.002)?;

// With weight decay for regularization
let nadam_wd = NAdam::with_weight_decay(0.001, 0.01)?;

// Full configuration (advanced users)
let nadam_full = NAdam::new(
    0.002,    // learning_rate
    0.9,      // beta1 (momentum coefficient)
    0.999,    // beta2 (RMSprop coefficient) 
    1e-8,     // eps (numerical stability)
    0.01,     // weight_decay
    0.004,    // momentum_decay (NAdam-specific)
    0.004,    // schedule_decay (NAdam-specific)
)?;

// Training step
nadam.step(&param, &grad);
```

**Key Benefits:**
- Combines Adam's adaptive learning with Nesterov momentum acceleration
- Time-dependent beta1 scheduling for improved late-stage convergence
- Superior performance on NLP tasks and fine-tuning scenarios
- **Performance**: 18,976+ steps/sec in benchmarks

##### RAdam (Rectified Adam)

```rust
// Basic usage - recommended for stable training
let radam = RAdam::default_params(0.001)?;

// With weight decay
let radam_wd = RAdam::with_weight_decay(0.001, 0.01)?;

// Custom rectification threshold
let radam_custom = RAdam::with_custom_threshold(
    0.001,    // learning_rate
    0.9,      // beta1
    0.999,    // beta2
    1e-8,     // eps
    0.01,     // weight_decay
    4.0,      // rectification_threshold (default: 4.0)
)?;

// Training step with automatic variance rectification
radam.step(&param, &grad);
```

**Key Benefits:**
- Automatically handles Adam's variance issue during early training phases
- No manual warmup scheduling required - built-in rectification logic
- Exceptional stability for transformer architectures and large models
- Falls back to SGD with momentum when variance is not rectifiable
- **Performance**: 21,939+ steps/sec with optimized rectification caching

##### Adamax (Adam with Infinity Norm)

```rust
// Basic usage - ideal for sparse gradients
let adamax = Adamax::default_params(0.002)?;

// With weight decay
let adamax_wd = Adamax::with_weight_decay(0.001, 0.01)?;

// Custom epsilon for numerical stability
let adamax_eps = Adamax::with_infinity_eps(
    0.002,    // learning_rate
    0.9,      // beta1
    0.999,    // beta2
    1e-7,     // eps (smaller default for stability)
    0.01,     // weight_decay
    1e-7,     // infinity_eps (Adamax-specific)
)?;

// Optimized training step with infinity norm
adamax.step(&param, &grad);
```

**Key Benefits:**
- Uses infinity norm (max) instead of L2 norm for second moment estimation
- More stable than Adam for embeddings and sparse feature learning
- No bias correction needed for the infinity norm component
- Excellent handling of outlier gradients and numerical stability
- **Performance**: 33,632+ steps/sec (highest performance optimizer)

##### Enhanced L-BFGS with Advanced Line Search

```rust
// Basic usage with Strong Wolfe line search (recommended)
let lbfgs = LBFGS::new(1.0)?;

// Advanced configuration with specific line search method
let lbfgs_advanced = LBFGS::with_params(
    1.0,                                 // learning_rate
    20,                                  // max_iter
    20,                                  // max_eval
    1e-5,                                // tolerance_grad
    1e-9,                                // tolerance_change
    10,                                  // history_size
    LineSearchMethod::StrongWolfe {      // Enhanced line search
        c1: 1e-4,                        // Armijo condition parameter
        c2: 0.9,                         // Curvature condition parameter
    }
)?;

// Backtracking line search for conservative steps
let lbfgs_backtrack = LBFGS::with_params(
    1.0, 20, 20, 1e-5, 1e-9, 10,
    LineSearchMethod::Backtracking {
        c1: 1e-4,                        // Armijo parameter
        rho: 0.5,                        // Step reduction factor
    }
)?;

// Fixed step size for simple problems
let lbfgs_fixed = LBFGS::with_params(
    0.1, 20, 20, 1e-5, 1e-9, 10,
    LineSearchMethod::None
)?;

// Advanced usage with convergence monitoring
lbfgs.set_tolerance_grad(1e-6)?;
lbfgs.step(&param, &grad);

// Access convergence information
let state = lbfgs.state_dict();
println!("Iterations: {}, Function evaluations: {}", 
         state["step"], state["n_iter"]);
```

**Enhanced Features:**
- **Modular LBFGSMemory**: Efficient storage of gradient history
- **Advanced line search**: Strong Wolfe conditions with numerical safeguards
- **Enhanced convergence detection**: Multiple criteria for robust stopping
- **Memory optimization**: Automatic cleanup and size management
- **Numerical stability**: Improved handling of ill-conditioned problems

#### Unified Adam Architecture Benefits

The GenericAdamOptimizer provides several architectural advantages:

```rust
use rustorch::optim::common::{GenericAdamOptimizer, AdamVariant};

// All Adam variants share common infrastructure:
// - Consistent state management (AdamState)
// - Unified configuration (AdamConfig) 
// - Shared utilities (AdamUtils)
// - Error handling with RusTorchResult<T>

// Example: Create optimizers with consistent API
let nadam: NAdam = GenericAdamOptimizer::from_config_variant(
    config.learning_rate,
    NAdam::create_variant(config.momentum_decay, config.schedule_decay)?
)?;

// All optimizers support the same state operations
nadam.zero_grad();
let state_dict = nadam.state_dict();
nadam.load_state_dict(&saved_state)?;
```

**Architectural Benefits:**
- **50%+ code reduction**: Eliminated duplicate implementations
- **Consistent API**: Uniform interface across all Adam variants
- **Improved maintainability**: Single source of truth for Adam logic
- **Enhanced performance**: Shared optimizations benefit all variants
- **Better error handling**: Comprehensive RusTorchError integration

### Learning Rate Schedulers

```rust
use rustorch::optim::{StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau};

// Step learning rate scheduler
let step_scheduler = StepLR::<f32>::new(optimizer, 30, 0.1); // step_size, gamma

// Exponential learning rate scheduler
let exp_scheduler = ExponentialLR::<f32>::new(optimizer, 0.95); // gamma

// Cosine annealing scheduler
let cosine_scheduler = CosineAnnealingLR::<f32>::new(optimizer, 100, 0.0); // T_max, eta_min

// Reduce on plateau scheduler
let plateau_scheduler = ReduceLROnPlateau::<f32>::new(
    optimizer,
    "min",                                          // mode
    0.1,                                            // factor
    10,                                             // patience
    1e-8,                                           // threshold
    3,                                              // cooldown
    0.0                                             // min_lr
);

// Update learning rate
step_scheduler.step();
plateau_scheduler.step(validation_loss);
```

#### Phase 2: Advanced Optimization Utilities (NEW in v0.5.13)

##### OptimizerUtils - Enhanced Stability and Performance

```rust
use rustorch::optim::utils::{OptimizerUtils, StabilityConfig, OptimizerMetrics};

// Gradient stabilization and clipping
let clipped_grad = OptimizerUtils::clip_gradient_norm(&grad, 1.0);
let stable_grad = OptimizerUtils::sanitize_tensor(&grad, 0.0, 1e6);

// Enhanced momentum and velocity updates with numerical stability
let momentum = OptimizerUtils::enhanced_momentum_update(
    &momentum, &grad, 0.9, 1e-8  // beta1, eps
);
let velocity = OptimizerUtils::enhanced_velocity_update(
    &velocity, &grad, 0.999, 1e-8  // beta2, eps
);

// Bias correction with improved numerics
let corrected_momentum = OptimizerUtils::bias_correction(&momentum, 0.9, step);
let corrected_velocity = OptimizerUtils::bias_correction(&velocity, 0.999, step);

// Advanced mathematical utilities
let l1_norm = OptimizerUtils::l1_norm(&tensor);
let l2_norm = OptimizerUtils::l2_norm(&tensor);
let safe_sqrt = OptimizerUtils::stable_sqrt(&tensor, 1e-8);

// Tensor stability operations
let clamped = OptimizerUtils::clamp_for_stability(&tensor, -1e6, 1e6);
let normalized = OptimizerUtils::normalize_tensor(&tensor);
```

##### StabilityConfig - Comprehensive Training Stability

```rust
use rustorch::optim::utils::StabilityConfig;

// Create stability configuration
let config = StabilityConfig {
    min_eps: 1e-8,                    // Numerical stability threshold
    max_grad_norm: 10.0,              // Gradient clipping threshold
    max_param_value: 1e6,             // Parameter value limit
    auto_nan_correction: true,        // Automatic NaN/Inf handling
    gradient_clipping: true,          // Enable gradient clipping
};

// Default configuration for most use cases
let default_config = StabilityConfig::default();

// Apply stability measures
let stabilized_grad = config.stabilize_gradient(&grad);
let stabilized_param = config.stabilize_parameter(&param);

// Check for numerical issues
if config.needs_stabilization(&grad) {
    println!("Applying gradient stabilization");
}
```

##### OptimizerMetrics - Performance and Convergence Monitoring

```rust
use rustorch::optim::utils::OptimizerMetrics;

// Create metrics tracker
let mut metrics = OptimizerMetrics::new(1000); // History size

// Update metrics during training
metrics.update_gradient_norm(OptimizerUtils::l2_norm(&grad));
metrics.update_parameter_norm(OptimizerUtils::l2_norm(&param));
metrics.update_learning_rate(optimizer.get_learning_rate());

// Get performance statistics
let stats = metrics.get_statistics();
println!("Average gradient norm: {:.6}", stats.avg_gradient_norm);
println!("Parameter stability: {:.6}", stats.parameter_stability);

// Convergence detection
if metrics.detect_convergence(1e-6, 100) {
    println!("Training has converged!");
}

// Issue detection
match metrics.detect_issues() {
    Some(issue) => println!("Training issue detected: {:?}", issue),
    None => println!("Training proceeding normally"),
}
```

##### OptimizerFactory - Configuration Management and Parameter Suggestions

```rust
use rustorch::optim::utils::{OptimizerFactory, OptimizerType, ProblemType, DatasetSize};

// Get suggested parameters for your problem
let suggestions = OptimizerFactory::suggest_parameters(
    OptimizerType::RAdam,           // Optimizer choice
    ProblemType::ImageClassification, // Problem type
    DatasetSize::Large,              // Dataset scale
    Some(0.001)                     // Base learning rate
);

println!("Suggested learning rate: {}", suggestions.learning_rate);
println!("Suggested weight decay: {}", suggestions.weight_decay);
println!("Suggested batch size: {}", suggestions.batch_size);

// Validate optimizer configuration
let config = AdamConfig {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,
};

match OptimizerFactory::validate_config(&config) {
    Ok(_) => println!("Configuration is valid"),
    Err(e) => println!("Configuration error: {:?}", e),
}
```

##### Advanced Learning Rate Scheduling

```rust
use rustorch::optim::utils::OptimizerUtils;

// Enhanced cosine annealing with restarts
let lr = OptimizerUtils::cosine_annealing_lr(
    0.001,    // base_lr
    50,       // T_max (period)
    step,     // current_step
    0.0,      // eta_min
    1         // T_mult (restart multiplier)
);

// Warm-up scheduling
let warmup_lr = OptimizerUtils::warmup_learning_rate(
    0.001,    // target_lr
    step,     // current_step
    1000,     // warmup_steps
    "linear"  // warmup_type: "linear", "constant", or "cosine"
);
```

#### Performance Benchmarking System (NEW in v0.5.13)

##### OptimizerBenchmark - Comprehensive Performance Testing

```rust
use rustorch::optim::benchmarks::{OptimizerBenchmark, BenchmarkConfig, BenchmarkResult};

// Create benchmark suite
let mut benchmark = OptimizerBenchmark::new();

// Run Adam variant comparison
let results = benchmark.run_adam_comparison();
for (config_name, result) in results {
    println!("{}: {:.2}μs/step, {:.1} steps/sec, {}MB memory",
             config_name, result.avg_step_time_us, 
             result.steps_per_second, result.peak_memory_mb);
}

// Specific L-BFGS benchmarks
let lbfgs_results = benchmark.run_lbfgs_benchmark();

// Generate detailed performance report
let report = benchmark.generate_report(&results);
println!("{}", report);

// Custom benchmark configuration
let custom_config = BenchmarkConfig {
    name: "custom_test".to_string(),
    tensor_shape: vec![1024, 1024],
    iterations: 100,
    warmup_iterations: 10,
    learning_rate: 0.001,
};

let custom_result = benchmark.run_custom_benchmark(&custom_config);
```

##### Quick Performance Testing

```rust
use rustorch::optim::benchmarks::quick_performance_test;

// Rapid optimizer comparison (for development/CI)
quick_performance_test(); // Outputs comparison results

// Results show relative performance:
// Adamax : 33,632 steps/sec ⚡ (fastest)
// RAdam  : 21,939 steps/sec 🚀
// NAdam  : 18,976 steps/sec ✨
```

## 🎯 Special Mathematical Functions

### Gamma and Related Functions

```rust
use rustorch::special;

// Gamma function family
let gamma_result = special::gamma(&tensor);          // Gamma function
let lgamma_result = special::lgamma(&tensor);        // Log gamma
let digamma_result = special::digamma(&tensor);      // Digamma (psi function)
let polygamma_result = special::polygamma(1, &tensor); // Polygamma function

// Beta functions
let beta_result = special::beta(&a, &b);            // Beta function
let lbeta_result = special::lbeta(&a, &b);          // Log beta function

// Incomplete gamma/beta functions
let gammainc_result = special::gammainc(&a, &x);    // Lower incomplete gamma
let gammaincc_result = special::gammaincc(&a, &x);  // Upper incomplete gamma
let betainc_result = special::betainc(&a, &b, &x);  // Incomplete beta
```

### Bessel Functions

```rust
// Bessel functions of the first kind
let j0_result = special::j0(&tensor);               // Order 0
let j1_result = special::j1(&tensor);               // Order 1
let jn_result = special::jn(2, &tensor);            // Order n

// Bessel functions of the second kind
let y0_result = special::y0(&tensor);               // Order 0
let y1_result = special::y1(&tensor);               // Order 1
let yn_result = special::yn(2, &tensor);            // Order n

// Modified Bessel functions
let i0_result = special::i0(&tensor);               // Modified first kind, order 0
let i1_result = special::i1(&tensor);               // Modified first kind, order 1
let iv_result = special::iv(2.0, &tensor);          // Modified first kind, order v
let k0_result = special::k0(&tensor);               // Modified second kind, order 0
let k1_result = special::k1(&tensor);               // Modified second kind, order 1
let kv_result = special::kv(2.0, &tensor);          // Modified second kind, order v
```

### Error Functions

```rust
// Error function family
let erf_result = special::erf(&tensor);             // Error function
let erfc_result = special::erfc(&tensor);           // Complementary error function
let erfcx_result = special::erfcx(&tensor);         // Scaled complementary error function
let erfi_result = special::erfi(&tensor);           // Imaginary error function
let erfcinv_result = special::erfcinv(&tensor);     // Inverse complementary error function
let erfinv_result = special::erfinv(&tensor);       // Inverse error function

// Dawson function
let dawson_result = special::dawson(&tensor);       // Dawson's integral
```

### Exponential Integrals

```rust
// Exponential integral functions
let ei_result = special::ei(&tensor);               // Exponential integral
let expi_result = special::expi(&tensor);           // Exponential integral for complex args
let exp1_result = special::exp1(&tensor);           // E1 exponential integral
let expn_result = special::expn(2, &tensor);        // En exponential integral
```

### Hypergeometric Functions

```rust
// Hypergeometric functions
let hyp0f1_result = special::hyp0f1(&b, &z);        // 0F1 hypergeometric function
let hyp1f1_result = special::hyp1f1(&a, &b, &z);   // 1F1 confluent hypergeometric
let hyp2f1_result = special::hyp2f1(&a, &b, &c, &z); // 2F1 Gauss hypergeometric
let hyperu_result = special::hyperu(&a, &b, &z);    // Tricomi confluent hypergeometric
```

### Elliptic Integrals

```rust
// Complete elliptic integrals
let ellipk_result = special::ellipk(&m);            // Complete elliptic integral K
let ellipe_result = special::ellipe(&m);            // Complete elliptic integral E

// Incomplete elliptic integrals
let ellipf_result = special::ellipf(&phi, &m);      // Incomplete elliptic integral F
let ellipinc_result = special::ellipinc(&phi, &m);  // Incomplete elliptic integral E
```

## 📈 Statistical Distributions

### Continuous Distributions

```rust
use rustorch::distributions::*;

// Normal distribution
let normal = Normal::<f32>::new(0.0, 1.0);          // mean, std
let samples = normal.sample(&[1000]);
let prob = normal.pdf(&tensor);                     // Probability density
let cdf = normal.cdf(&tensor);                      // Cumulative distribution
let icdf = normal.icdf(&tensor);                    // Inverse CDF

// Other continuous distributions
let uniform = Uniform::<f32>::new(0.0, 1.0);        // low, high
let exponential = Exponential::<f32>::new(1.0);     // rate
let gamma = Gamma::<f32>::new(2.0, 1.0);            // alpha, beta
let beta = Beta::<f32>::new(2.0, 3.0);              // alpha, beta
let cauchy = Cauchy::<f32>::new(0.0, 1.0);          // location, scale
let laplace = Laplace::<f32>::new(0.0, 1.0);        // location, scale
let logistic = Logistic::<f32>::new(0.0, 1.0);      // location, scale
let lognormal = LogNormal::<f32>::new(0.0, 1.0);    // mean, std of log
let pareto = Pareto::<f32>::new(1.0, 1.0);          // scale, alpha
let weibull = Weibull::<f32>::new(1.0, 2.0);        // scale, concentration
let chi2 = Chi2::<f32>::new(5.0);                   // degrees of freedom
let studentt = StudentT::<f32>::new(10.0);          // degrees of freedom
let fisher_snedecor = FisherSnedecor::<f32>::new(5.0, 10.0); // df1, df2
```

### Discrete Distributions

```rust
// Discrete distributions
let bernoulli = Bernoulli::<f32>::new(0.5);         // probability
let binomial = Binomial::<f32>::new(10, 0.3);       // trials, probability
let categorical = Categorical::<f32>::new(&probs);   // probabilities
let geometric = Geometric::<f32>::new(0.5);         // probability
let poisson = Poisson::<f32>::new(3.0);             // rate
let multinomial = Multinomial::<f32>::new(10, &probs); // trials, probabilities
```

### Multivariate Distributions

```rust
// Multivariate distributions
let mvn = MultivariateNormal::<f32>::new(&mean_vec, &covariance_matrix);
let dirichlet = Dirichlet::<f32>::new(&concentration);
let wishart = Wishart::<f32>::new(10.0, &scale_matrix); // df, scale
let inv_wishart = InverseWishart::<f32>::new(10.0, &scale_matrix);
```

## 🖼️ Computer Vision Module

### Image Transforms

```rust
use rustorch::vision::transforms::*;

// Basic transforms
let resize = Resize::<f32>::new((224, 224));        // Target size
let crop = CenterCrop::<f32>::new((224, 224));      // Crop size
let random_crop = RandomCrop::<f32>::new((224, 224)); // Random crop
let random_flip = RandomHorizontalFlip::<f32>::new(0.5); // Probability

// Advanced transforms
let color_jitter = ColorJitter::<f32>::new(0.1, 0.1, 0.1, 0.05); // brightness, contrast, saturation, hue
let gaussian_blur = GaussianBlur::<f32>::new((3, 3), (0.1, 2.0)); // kernel_size, sigma_range
let normalize = Normalize::<f32>::new(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225]); // ImageNet stats

// Geometric transforms
let rotate = RandomRotation::<f32>::new((-30.0, 30.0)); // Angle range
let affine = RandomAffine::<f32>::new(
    (-15.0, 15.0),                                  // degrees
    Some((0.1, 0.1)),                               // translate
    Some((0.8, 1.2)),                               // scale
    Some((-10.0, 10.0))                             // shear
);

// Apply transforms
let transformed = resize.forward(&image_tensor);
```

### Data Augmentation

```rust
use rustorch::vision::augmentation::*;

// Compose transforms
let transform_pipeline = Compose::<f32>::new(vec![
    Box::new(RandomCrop::<f32>::new((224, 224))),
    Box::new(RandomHorizontalFlip::<f32>::new(0.5)),
    Box::new(ColorJitter::<f32>::new(0.2, 0.2, 0.2, 0.1)),
    Box::new(Normalize::<f32>::new(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])),
]);

let augmented = transform_pipeline.forward(&image);

// Random apply
let random_apply = RandomApply::<f32>::new(
    Box::new(GaussianBlur::<f32>::new((3, 3), (0.1, 2.0))),
    0.3                                             // Probability
);
```

## 🔢 Linear Algebra Module (Feature: "linalg")

### Matrix Decomposition

```rust
use rustorch::linalg;

// Singular Value Decomposition
let (u, s, vt) = linalg::svd(&matrix, true);        // full_matrices
let (u, s, vt) = linalg::svd_lowrank(&matrix, None, None); // Low-rank approximation

// QR Decomposition
let (q, r) = linalg::qr(&matrix, "reduced");        // mode: "reduced" or "complete"

// Eigenvalue Decomposition
let (eigenvals, eigenvecs) = linalg::eig(&matrix);  // General eigenvalues
let (eigenvals, eigenvecs) = linalg::eigh(&matrix); // Symmetric/Hermitian eigenvalues

// Cholesky Decomposition
let l = linalg::cholesky(&matrix);                  // Lower triangular factor
let l = linalg::cholesky_ex(&matrix, true);         // Check positive definite

// LU Decomposition with Pivoting
let (p, l, u) = linalg::lu(&matrix);               // P, L, U matrices
let (lu, pivots) = linalg::lu_factor(&matrix);     // Factored form
```

### Matrix Operations

```rust
// Matrix inverse and pseudo-inverse
let inv = linalg::inv(&matrix);                     // Matrix inverse
let pinv = linalg::pinv(&matrix, 1e-15);           // Pseudo-inverse with tolerance

// Matrix norms
let norm = linalg::norm(&matrix, "fro");            // Frobenius norm
let norm = linalg::vector_norm(&vector, 2.0);      // Vector p-norm
let norm = linalg::matrix_norm(&matrix, "nuclear"); // Nuclear norm

// Determinant and rank
let det = linalg::det(&matrix);                     // Determinant
let rank = linalg::matrix_rank(&matrix, None);      // Matrix rank

// Solving linear systems
let solution = linalg::solve(&a, &b);               // Solve Ax = b
let solution = linalg::solve_triangular(&a, &b, true, false, false); // upper, left, unit_diagonal
```

## 🎯 GPU Acceleration

### Device Management

```rust
use rustorch::gpu::{Device, DeviceType};

// Device selection
let cuda_device = Device::cuda(0);                  // CUDA device 0
let metal_device = Device::metal();                 // Metal device (macOS)
let opencl_device = Device::opencl(0);              // OpenCL device 0
let cpu_device = Device::cpu();                     // CPU device
let webgpu_device = Device::webgpu();               // WebGPU device (browser)

// Check device availability
let is_cuda_available = Device::is_cuda_available();
let is_metal_available = Device::is_metal_available();
let cuda_count = Device::cuda_device_count();

// Move tensors to device
let gpu_tensor = tensor.to_device(&cuda_device);
let cpu_tensor = gpu_tensor.to_cpu();
```

### GPU Matrix Operations (UPDATED Phase 9)

```rust
use rustorch::gpu::matrix_ops::GpuLinearAlgebra;

// High-performance GPU matrix multiplication
let a = Tensor::rand(vec![1024, 1024]);
let b = Tensor::rand(vec![1024, 1024]);

// Automatic device selection (CUDA > Metal > OpenCL > CPU)
let result = a.gpu_matmul(&b)?;                         // Auto-select best GPU
let batch_result = a.gpu_batch_matmul(&b)?;             // Batch operations

// Metal GPU (macOS) - Up to 4875x speedup on Apple Silicon
#[cfg(feature = "metal")]
{
    use rustorch::gpu::metal_kernels::MetalKernelExecutor;
    let executor = MetalKernelExecutor::new()?;
    let result = executor.matrix_multiply_f32(&a_f32, &b_f32, m, n, k)?;
}

// CUDA GPU (NVIDIA) - cuBLAS integration
#[cfg(feature = "cuda")]
{
    use rustorch::gpu::cuda_enhanced::CudaMatrixExecutor;
    let executor = CudaMatrixExecutor::new(0)?;
    let result = executor.matmul_f32(&a, &b, &mut c, m, n, k, false)?;
}

// OpenCL GPU (Cross-platform)
#[cfg(feature = "opencl")]
{
    use rustorch::gpu::opencl_kernels::OpenClKernelExecutor;
    let executor = OpenClKernelExecutor::new(0)?;
    let result = executor.matrix_multiply_f32(&a, &b, m, n, k)?;
}
```

### Performance Benchmarking

```rust
// Metal performance (Apple M4 Pro results)
// 64x64 matrices: ~20x speedup
// 512x512 matrices: ~4875x speedup  
// 1024x1024 matrices: ~57.764 GFLOPS

let start = std::time::Instant::now();
let result = a.gpu_matmul(&b)?;
let duration = start.elapsed();
println!("GPU matmul: {:.2}ms", duration.as_millis());
```

### GPU Memory Management

```rust
use rustorch::gpu::memory::{GpuMemoryPool, MemoryStats};

// Memory pool
let mut pool = GpuMemoryPool::new(&cuda_device);
let gpu_tensor = pool.allocate_tensor(&[1024, 1024]);
pool.deallocate_tensor(gpu_tensor);

// Memory statistics
let stats = MemoryStats::current(&cuda_device);
println!("Used: {}MB, Available: {}MB", stats.used_mb(), stats.available_mb());

// Memory cleanup
pool.empty_cache();                                 // Clear unused memory
pool.synchronize();                                 // Wait for operations
```

## 💾 Serialization System (Phase 9)

### Model Save/Load Operations

```rust
use rustorch::serialization::{ModelIO, SerializationFormat, SerializationError};

// Save model in various formats
let model = MyNeuralNetwork::new();
let state_dict = model.state_dict();

// PyTorch compatible format
ModelIO::save(&state_dict, "model.pth", SerializationFormat::PyTorch)?;

// Native RusTorch format (optimized)
ModelIO::save(&state_dict, "model.rusttorch", SerializationFormat::Native)?;

// ONNX format
ModelIO::save(&state_dict, "model.onnx", SerializationFormat::Onnx)?;

// Load model
let loaded_state = ModelIO::load("model.pth")?;
model.load_state_dict(loaded_state)?;
```

### Tensor Serialization

```rust
use rustorch::serialization::TensorSerializer;

// Serialize single tensor
let tensor = Tensor::rand(vec![100, 100]);
let serialized = TensorSerializer::serialize(&tensor)?;

// Deserialize
let restored = TensorSerializer::deserialize(&serialized)?;
assert_eq!(tensor.shape(), restored.shape());

// Batch serialization
let tensors = vec![tensor1, tensor2, tensor3];
let batch_data = TensorSerializer::serialize_batch(&tensors)?;
```

### JIT Compilation System

```rust
use rustorch::serialization::{JitCompiler, JitModule, CompilationOptions};

// Compile model to JIT format
let options = CompilationOptions {
    optimization_level: 2,
    target_platform: "cpu",
    enable_fusion: true,
};

let jit_module = JitCompiler::compile(model, options)?;

// Execute JIT model
let input = Tensor::rand(vec![1, 784]);
let output = jit_module.forward(&input)?;

// Save JIT module
jit_module.save("model.jit")?;

// Load JIT module
let loaded_jit = JitModule::load("model.jit")?;
```

### Custom Serialization Formats

```rust
use rustorch::serialization::{SerializableModel, CustomFormat};

// Implement custom serialization
impl SerializableModel for MyModel {
    fn serialize_custom(&self) -> Result<Vec<u8>, SerializationError> {
        // Custom binary format
        let mut buffer = Vec::new();
        self.write_header(&mut buffer)?;
        self.write_parameters(&mut buffer)?;
        self.write_metadata(&mut buffer)?;
        Ok(buffer)
    }
    
    fn deserialize_custom(data: &[u8]) -> Result<Self, SerializationError> {
        let mut reader = BinaryReader::new(data);
        let header = reader.read_header()?;
        let params = reader.read_parameters()?;
        let metadata = reader.read_metadata()?;
        Ok(Self::from_components(header, params, metadata))
    }
}
```

### Cross-Platform Compatibility

```rust
// Platform-specific optimizations
let serializer = TensorSerializer::with_platform_optimization(true);

// Ensure compatibility across systems
let compat_options = SerializationOptions {
    use_portable_format: true,
    endianness: Endianness::LittleEndian,
    float_precision: FloatPrecision::F32,
};

ModelIO::save_with_options(&model, "portable_model.bin", compat_options)?;
```

## 🌐 WebAssembly Support

> 📋 **Complete WASM API Reference**: [WASM API Documentation](WASM_API_DOCUMENTATION.md)

For browser-based machine learning applications, RusTorch provides comprehensive WebAssembly bindings including:

- **Core tensor operations** - WASM-optimized tensor arithmetic and manipulation
- **Neural network layers** - Linear layers, activations, and model inference
- **Browser integration** - Storage persistence, canvas rendering, performance monitoring
- **WebGPU acceleration** - Chrome-optimized GPU acceleration for enhanced performance
- **Advanced features** - Optimizers, data transforms, quality metrics, anomaly detection
- **Memory management** - Efficient allocation, cleanup, and garbage collection
- **Signal processing** - FFT operations and filtering for audio/signal data
- **JavaScript interoperability** - Seamless data exchange with browser environments

### Quick Start Example

```rust
use rustorch::wasm::*;
use wasm_bindgen::prelude::*;

// Create WASM tensor
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = WasmTensor::new(data, vec![2, 2]);

// Neural network inference
let model = WasmModel::new();
let predictions = model.predict(&input_data);
```

**📚 For detailed WASM API documentation, examples, and browser integration guides, see:** [WASM API Documentation](WASM_API_DOCUMENTATION.md)

## 📊 FFT and Signal Processing

### Fast Fourier Transform

```rust
use rustorch::fft;

// 1D FFT
let fft_result = fft::fft(&signal);                 // Forward FFT
let ifft_result = fft::ifft(&spectrum);             // Inverse FFT
let rfft_result = fft::rfft(&real_signal);          // Real FFT
let irfft_result = fft::irfft(&real_spectrum);      // Inverse real FFT

// 2D FFT
let fft2_result = fft::fft2(&image);                // 2D forward FFT
let ifft2_result = fft::ifft2(&spectrum);           // 2D inverse FFT

// N-dimensional FFT
let fftn_result = fft::fftn(&tensor, &[0, 1, 2]);  // FFT along specified axes
let ifftn_result = fft::ifftn(&spectrum, &[0, 1, 2]); // Inverse N-D FFT

// FFT with normalization
let fft_norm = fft::fft_with_norm(&signal, "ortho"); // Orthonormal scaling
```

### Window Functions

```rust
use rustorch::signal;

// Window functions for signal processing
let hann = signal::hann_window(512);                // Hann window
let hamming = signal::hamming_window(512);          // Hamming window
let blackman = signal::blackman_window(512);        // Blackman window
let bartlett = signal::bartlett_window(512);        // Bartlett window
let kaiser = signal::kaiser_window(512, 8.6);       // Kaiser window with beta
```

## 🔄 Model Import/Export

### Model Serialization

```rust
use rustorch::models::{save_model, load_model, ModelFormat};

// Save model
save_model(&model, "model.pt", ModelFormat::PyTorch)?;
save_model(&model, "model.safetensors", ModelFormat::SafeTensors)?;
save_model(&model, "model.onnx", ModelFormat::Onnx)?;

// Load model
let loaded_model = load_model::<MyModel>("model.pt", ModelFormat::PyTorch)?;

// Model state dict
let state_dict = model.state_dict();
model.load_state_dict(&state_dict)?;
```

### Checkpoint Management

```rust
use rustorch::training::{Checkpoint, CheckpointManager};

// Save checkpoint
let checkpoint = Checkpoint::new(&model, &optimizer, epoch, loss);
checkpoint.save("checkpoint_epoch_10.pt")?;

// Load checkpoint
let checkpoint = Checkpoint::load("checkpoint_epoch_10.pt")?;
checkpoint.restore_model(&mut model)?;
checkpoint.restore_optimizer(&mut optimizer)?;

// Checkpoint manager
let mut manager = CheckpointManager::new("./checkpoints", 5); // Keep 5 latest
manager.save(&model, &optimizer, epoch, loss)?;
```

## 🌍 Distributed Training

### Data Parallel

```rust
use rustorch::distributed::{DataParallel, DistributedDataParallel};

// Data parallel on multiple GPUs
let devices = vec![Device::cuda(0), Device::cuda(1)];
let parallel_model = DataParallel::new(model, devices);
let output = parallel_model.forward(&input);

// Distributed data parallel (multi-node)
let ddp_model = DistributedDataParallel::new(
    model,
    device_ids,
    output_device,
    broadcast_buffers
);
```

### Communication Backend

```rust
use rustorch::distributed::{ProcessGroup, Backend, init_process_group};

// Initialize distributed training
init_process_group(Backend::NCCL, "env://", 0, 4).await?; // rank 0 of 4

// Create process group
let process_group = ProcessGroup::new(Backend::NCCL, vec![0, 1, 2, 3]);

// Collective operations
process_group.all_reduce(&mut tensor, "sum").await?;
process_group.broadcast(&mut tensor, 0).await?;    // Root rank 0
process_group.all_gather(&tensors, &tensor).await?;
```

## 🔧 Advanced Features

### Custom Kernels

```rust
use rustorch::gpu::kernels::{CustomKernel, KernelBuilder};

// Define custom CUDA kernel
let kernel_source = r#"
extern "C" __global__ void custom_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"#;

let kernel = CustomKernel::from_source(kernel_source, "custom_add")?;
let result = kernel.launch(&[&tensor_a, &tensor_b], (256, 1, 1), (1024, 1, 1))?;
```

### Memory-Mapped Tensors

```rust
use rustorch::memory::{MmapTensor, MmapOptions};

// Memory-mapped tensor for large datasets
let mmap_options = MmapOptions::new()
    .read_only(true)
    .huge_tlb(true);
    
let mmap_tensor = MmapTensor::<f32>::open("large_dataset.bin", &[1000000, 512], mmap_options)?;
let batch = mmap_tensor.slice(0, 1000, 1032);      // Efficient batch loading
```

### Mixed Precision Training

```rust
use rustorch::training::{MixedPrecisionTrainer, GradScaler};

// Mixed precision with automatic scaling
let mut scaler = GradScaler::new();
let mut trainer = MixedPrecisionTrainer::new(model, optimizer, scaler);

// Training step with mixed precision
let loss = trainer.train_step(&input, &target, &mut scaler)?;
```

## 🔍 Debugging and Profiling

### Computation Graph Visualization

```rust
use rustorch::debug::{GraphVisualizer, ProfilerConfig};

// Visualize computation graph
let visualizer = GraphVisualizer::new();
let graph_svg = visualizer.visualize(&output_variable)?;
std::fs::write("computation_graph.svg", graph_svg)?;

// Profile execution
let profiler = Profiler::new(ProfilerConfig::default());
profiler.start();
let result = model.forward(&input);
let profile_report = profiler.stop();
println!("{}", profile_report);
```

### Performance Analysis

```rust
use rustorch::profiling::{PerformanceProfiler, MemoryProfiler};

// Performance profiling
let perf_profiler = PerformanceProfiler::new();
perf_profiler.start("forward_pass");
let output = model.forward(&input);
let timing = perf_profiler.stop("forward_pass");

// Memory profiling
let mem_profiler = MemoryProfiler::new();
let (peak_memory, current_memory) = mem_profiler.get_memory_stats();
```

## 🔧 Error Handling

### Unified Error System

RusTorch uses a unified error handling system with `RusTorchError` and `RusTorchResult<T>` for consistent error management across all modules.

```rust
use rustorch::error::{RusTorchError, RusTorchResult};

// All operations return RusTorchResult<T>
fn tensor_operation() -> RusTorchResult<Tensor<f32>> {
    let tensor = Tensor::randn(vec![3, 3]);
    Ok(tensor)
}

// Pattern matching on specific error types
match tensor_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(RusTorchError::InvalidShape { expected, got }) => {
        eprintln!("Shape error: expected {:?}, got {:?}", expected, got);
    },
    Err(RusTorchError::InvalidDimension { dimension, reason }) => {
        eprintln!("Dimension error: {} (dimension: {})", reason, dimension);
    },
    Err(RusTorchError::ComputationError { operation, details }) => {
        eprintln!("Computation error in {}: {}", operation, details);
    },
    Err(RusTorchError::GpuError { device, operation, message }) => {
        eprintln!("GPU error on {}: {} - {}", device, operation, message);
    },
    Err(RusTorchError::MemoryError { size, device, operation }) => {
        eprintln!("Memory error: {} bytes on {} during {}", size, device, operation);
    },
    Err(e) => eprintln!("Other error: {}", e),
}

// Convenient constructor methods
let shape_error = RusTorchError::shape_mismatch(&[3, 3], &[2, 2]);
let param_error = RusTorchError::invalid_parameter("learning rate must be positive");
let memory_error = RusTorchError::memory_alloc(1024, "cuda:0");
let import_error = RusTorchError::import_error("missing required parameter");

// Automatic conversion from common error types
let json_result: RusTorchResult<()> = Ok(serde_json::to_string(&data)?); // serde_json::Error auto-converts
let string_result: RusTorchResult<()> = Err("Custom error message".into()); // String auto-converts
```

### Error Categories

The unified `RusTorchError` enum includes comprehensive error variants:

- **Shape & Dimension Errors**: `InvalidShape`, `InvalidDimension`, `BroadcastError`
- **Computation Errors**: `ComputationError`, `NumericalInstability`, `OverflowError`
- **GPU & Device Errors**: `GpuError`, `DeviceError`, `CudaError`, `MetalError`
- **Memory Management**: `MemoryError`, `AllocationFailed`, `OutOfMemory`
- **Data & IO Errors**: `DataError`, `FileNotFound`, `SerializationError`
- **Import & Export**: `ImportError`, `ExportError`, `FormatError`
- **Neural Network**: `ModelError`, `LayerError`, `OptimizerError`
- **Training & Validation**: `TrainingError`, `ValidationError`, `ConvergenceError`

## 📖 Usage Examples

### Complete Training Example

```rust
use rustorch::prelude::*;

fn main() -> Result<()> {
    // Create model
    let mut model = Sequential::<f32>::new()
        .add_layer(Box::new(Linear::<f32>::new(784, 128)))
        .add_activation(Box::new(ReLU::<f32>::new()))
        .add_layer(Box::new(Linear::<f32>::new(128, 10)));

    // Setup optimizer and loss
    let mut optimizer = Adam::<f32>::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::<f32>::new();

    // Training loop
    for epoch in 0..100 {
        let mut total_loss = 0.0;
        
        for (input, target) in &train_loader {
            // Forward pass
            let output = model.forward(input);
            let loss = loss_fn.forward(&output, target);
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item();
        }
        
        println!("Epoch {}: Loss = {:.4}", epoch, total_loss / train_loader.len() as f32);
    }
    
    Ok(())
}
```

### GPU Acceleration Example

```rust
use rustorch::prelude::*;

fn gpu_training() -> Result<()> {
    // Setup GPU device
    let device = Device::cuda(0);
    
    // Create model on GPU
    let mut model = Sequential::<f32>::new()
        .add_layer(Box::new(Linear::<f32>::new(1024, 512)))
        .add_activation(Box::new(ReLU::<f32>::new()))
        .add_layer(Box::new(Linear::<f32>::new(512, 10)))
        .to_device(&device);

    // Move data to GPU
    let input = Tensor::<f32>::randn(vec![32, 1024]).to_device(&device);
    let target = Tensor::<f32>::randint(0, 10, vec![32]).to_device(&device);

    // GPU-accelerated forward pass
    let output = model.forward(&input);
    let loss = cross_entropy(&output, &target);
    
    println!("GPU Loss: {:.4}", loss.item());
    Ok(())
}
```

## 📚 API Reference Summary

### Core Modules Available

| Module | Description | Key Features |
|--------|-------------|--------------|
| `tensor` | Core tensor operations | Creation, arithmetic, mathematical functions, **advanced shape operations with builder pattern** |
| `nn` | Neural network layers | Linear, Conv, RNN, LSTM, GRU, activations, normalization, loss functions, **Phase 6 Transformer components** |
| `autograd` | Automatic differentiation | Variables, gradients, custom functions, computation graphs, **Phase 4 higher-order derivatives** |
| `optim` | Optimizers and schedulers | SGD, Adam, AdamW, RMSprop, learning rate scheduling, **Phase 2 advanced optimizers** |
| `data` | Data loading and processing | **Phase 5 Dataset/DataLoader API**, sampling strategies, transforms |
| `special` | Special mathematical functions | Gamma, Bessel, error functions, hypergeometric, elliptic integrals |
| `distributions` | Statistical distributions | Normal, uniform, gamma, beta, categorical, multivariate |
| `vision` | Computer vision utilities | Image transforms, data augmentation, preprocessing |
| `linalg` | Linear algebra operations | SVD, QR, eigenvalues, matrix decomposition, solving |
| `gpu` | GPU acceleration | CUDA, Metal, OpenCL, WebGPU, memory management |
| `fft` | Fourier transforms | FFT, RFFT, 2D FFT, N-dimensional FFT, window functions |
| `wasm` | WebAssembly bindings | Browser support, JavaScript integration ([WASM docs](WASM_API_DOCUMENTATION.md)) |

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | Core tensor operations | None |
| `linalg` | Linear algebra with BLAS/LAPACK | OpenBLAS, LAPACK |
| `cuda` | NVIDIA GPU acceleration | CUDA Runtime |
| `metal` | Apple GPU acceleration | Metal framework (macOS) |
| `opencl` | OpenCL GPU acceleration | OpenCL drivers |
| `webgpu` | WebGPU browser acceleration | WebGPU API |
| `wasm` | WebAssembly compilation | wasm-bindgen ([WASM API docs](WASM_API_DOCUMENTATION.md)) |
| `model-hub` | Model downloading and caching | HTTP client, crypto |
| `safetensors` | SafeTensors format support | Memory mapping |
| `onnx` | ONNX model import/export | ONNX Runtime |
| `python` | Python bindings | PyO3 |

## 🆕 New Features in v0.5.14

### Phase 6: Transformer Components - PyTorch Compatible (NEW)

RusTorch v0.5.14 introduces **Phase 6** transformer architecture, bringing the library to **95% completion** with production-ready PyTorch-compatible transformer components.

#### 🚀 **Multi-head Attention Implementation**

```rust
use rustorch::nn::transformer_phase6::MultiheadAttention;

// Standard multi-head attention (PyTorch compatible)
let attention = MultiheadAttention::<f32>::new(
    512,        // embed_dim
    8,          // num_heads
    Some(0.1),  // dropout
    Some(true), // bias
    None,       // kdim (defaults to embed_dim)
    None,       // vdim (defaults to embed_dim)
    Some(true), // batch_first
)?;

// Forward pass with query, key, value
let (output, attention_weights) = attention.forward(
    &query,     // [batch_size, seq_len, embed_dim]
    &key,       // [batch_size, seq_len, embed_dim]
    &value,     // [batch_size, seq_len, embed_dim]
    None,       // key_padding_mask
    true,       // need_weights
    None,       // attn_mask
    false,      // average_attn_weights
)?;
```

#### 🏗️ **Transformer Encoder/Decoder Layers**

```rust
use rustorch::nn::transformer_phase6::{TransformerEncoderLayer, TransformerDecoderLayer};

// Encoder layer with self-attention
let encoder_layer = TransformerEncoderLayer::<f32>::new(
    512,        // d_model
    8,          // nhead
    2048,       // dim_feedforward
    0.1,        // dropout
    "relu",     // activation
    None,       // layer_norm_eps
    true,       // batch_first
    None,       // norm_first
)?;

// Decoder layer with self-attention and cross-attention
let decoder_layer = TransformerDecoderLayer::<f32>::new(
    512,        // d_model
    8,          // nhead
    2048,       // dim_feedforward
    0.1,        // dropout
    "relu",     // activation
    None,       // layer_norm_eps
    true,       // batch_first
    None,       // norm_first
)?;
```

#### 🎯 **Complete Transformer Model**

```rust
use rustorch::nn::transformer_phase6::Transformer;

// Full transformer model (encoder-decoder architecture)
let transformer = Transformer::<f32>::new(
    512,        // d_model
    8,          // nhead
    Some(6),    // num_encoder_layers
    Some(6),    // num_decoder_layers
    Some(2048), // dim_feedforward
    Some(0.1),  // dropout
    None,       // activation
    None,       // custom_encoder
    None,       // custom_decoder
    None,       // layer_norm_eps
    Some(true), // batch_first
    None,       // norm_first
)?;

// Training forward pass
let output = transformer.forward(
    &src,       // Source sequence [batch_size, src_len, d_model]
    &tgt,       // Target sequence [batch_size, tgt_len, d_model]
    None,       // src_mask
    None,       // tgt_mask
    None,       // memory_mask
    None,       // src_key_padding_mask
    None,       // tgt_key_padding_mask
    None,       // memory_key_padding_mask
)?;
```

#### ✨ **Key Features of Phase 6**

- **Full PyTorch Compatibility**: API matches `torch.nn.Transformer` exactly
- **Production Ready**: Comprehensive error handling and validation
- **Memory Efficient**: Optimized attention computation with proper scaling
- **Flexible Architecture**: Support for custom encoder/decoder layers
- **Batch Processing**: Optimized for modern training workflows
- **Mixed Precision**: Compatible with FP16/BF16 training modes

### Phase 5: Modern Dataset API (UPDATED)

#### 🗂️ **Core Dataset Trait**

```rust
use rustorch::data::{Dataset, DataLoader, TensorDataset};
use rustorch::data::sampler::{SequentialSampler, RandomSampler, BatchSampler};

// Create tensor dataset
let features = vec![
    Tensor::randn(vec![100, 784]), // 100 samples, 784 features
    Tensor::randn(vec![100, 784]),
];
let targets = vec![
    Tensor::randint(0, 10, vec![100, 1]), // 100 labels
    Tensor::randint(0, 10, vec![100, 1]),
];

let dataset = TensorDataset::from_features_targets(features, targets)?;

// Modern DataLoader with sampling strategies
let sequential_sampler = SequentialSampler::new(dataset.len());
let random_sampler = RandomSampler::new(dataset.len(), None);
let batch_sampler = BatchSampler::new(
    Box::new(random_sampler),
    32,    // batch_size
    false  // drop_last
);

let dataloader = DataLoader::new(
    dataset,
    Some(Box::new(batch_sampler)),
    None,     // batch_size (handled by sampler)
    false,    // shuffle (handled by sampler)
    None,     // sampler
    None,     // batch_sampler
    2,        // num_workers
    None,     // collate_fn
    false,    // pin_memory
    false,    // drop_last
    None,     // timeout
)?;

// Iterate through batches
for batch in dataloader {
    let features = &batch[0]; // First tensor (features)
    let targets = &batch[1];  // Second tensor (targets)
    // Training logic...
}
```

#### 📊 **Advanced Sampling Strategies**

```rust
use rustorch::data::sampler::{
    WeightedRandomSampler, SubsetRandomSampler, 
    DistributedSampler, StratifiedSampler
};

// Weighted sampling for imbalanced datasets
let weights = vec![0.1, 0.9, 0.5, 0.3]; // Sample weights
let weighted_sampler = WeightedRandomSampler::new(weights, 1000, true)?;

// Subset sampling for validation splits
let indices = vec![0, 2, 4, 6, 8]; // Validation indices
let subset_sampler = SubsetRandomSampler::new(indices);

// Distributed sampling for multi-GPU training
let distributed_sampler = DistributedSampler::new(
    1000,   // dataset_size
    4,      // num_replicas
    1,      // rank
    true,   // shuffle
    None,   // seed
)?;

// Stratified sampling for balanced batches
let labels = vec![0, 1, 0, 1, 2, 2]; // Class labels
let stratified_sampler = StratifiedSampler::new(labels, 0.5)?; // sampling_rate
```

### Phase 4: Higher-Order Derivatives (NEW)

#### 🧮 **Jacobian and Hessian Computation**

```rust
use rustorch::autograd::{jacobian, hessian, hvp};

// Jacobian matrix computation for vector-valued functions
let jacobian_matrix = jacobian(
    |x| {
        // Vector-valued function: R^n -> R^m
        let squared = x.pow(2.0);
        let doubled = &squared * &Variable::new(Tensor::from_scalar(2.0), false);
        doubled
    },
    &input_variable,
    false, // create_graph
)?;

// Hessian matrix computation for scalar-valued functions
let hessian_matrix = hessian(
    |x| {
        // Scalar-valued function: R^n -> R
        let squared = x.pow(2.0);
        squared.sum() // Sum to make it scalar
    },
    &input_variable,
)?;

// Hessian-vector product (efficient for large problems)
let direction = Variable::new(Tensor::randn(vec![10]), false);
let hvp_result = hvp(
    |x| {
        let squared = x.pow(2.0);
        squared.sum()
    },
    &input_variable,
    &direction,
    false, // create_graph
)?;
```

#### 🔍 **Gradient Utilities and Validation**

```rust
use rustorch::autograd::{
    grad, gradient, validate_grad_setup, is_variable_in_graph,
    GradCheckConfig, gradcheck, gradcheck_simple
};

// Advanced gradient computation
let gradients = grad(
    &[output],           // outputs
    &[input],            // inputs
    Some(&[grad_output]), // grad_outputs
    true,                // retain_graph
    true,                // create_graph
)?;

// Gradient validation for debugging
let is_valid = validate_grad_setup(&outputs, &inputs)?;
let in_graph = is_variable_in_graph(&variable, &computation_graph)?;

// Numerical gradient checking
let config = GradCheckConfig {
    eps: 1e-6,
    atol: 1e-5,
    rtol: 1e-3,
    raise_exception: false,
};

let grad_check_result = gradcheck(
    |inputs| my_function(inputs),
    &[input_variable],
    &config,
)?;

// Simple gradient checking for development
let is_correct = gradcheck_simple(
    |inputs| my_function(inputs),
    &[input_variable],
    1e-6, // eps
)?;
```

#### 🛡️ **Gradient Context Management**

```rust
use rustorch::autograd::{
    no_grad, enable_grad, detect_anomaly,
    NoGradGuard, EnableGradGuard, AnomalyDetectionGuard,
    is_grad_enabled, set_grad_enabled
};

// Context managers for gradient control
{
    let _guard = no_grad(); // Disable gradient computation
    let prediction = model.forward(&input); // No gradients computed
}

{
    let _guard = enable_grad(); // Ensure gradients enabled
    let output = model.forward(&input); // Gradients computed
}

{
    let _guard = detect_anomaly(); // Enable anomaly detection
    let loss = compute_loss(&output, &target); // Detects NaN/Inf
    loss.backward(); // Will raise error if anomalies found
}

// Manual gradient state management
let was_enabled = is_grad_enabled();
set_grad_enabled(false);
// ... operations without gradients
set_grad_enabled(was_enabled); // Restore previous state
```

### Phase 3: Advanced Neural Network Layers (UPDATED)

#### 🎯 **Enhanced Attention Mechanisms**

The Phase 3 implementation now serves as the foundation for Phase 6 transformer components:

```rust
use rustorch::nn::attention::{AttentionLayer, ScaledDotProductAttention};

// Scaled dot-product attention (building block for multi-head)
let attention = ScaledDotProductAttention::<f32>::new(0.1); // dropout
let output = attention.forward(&query, &key, &value, None)?; // mask optional

// Legacy attention layer (use Phase 6 MultiheadAttention instead)
let legacy_attention = AttentionLayer::<f32>::new(512, 8, 0.1)?;
```

#### 🏗️ **Advanced Pooling Operations**

```rust
use rustorch::nn::adaptive_pool::{AdaptiveAvgPool1d, AdaptiveMaxPool1d, AdaptiveAvgPool3d};

// 1D adaptive pooling
let adaptive_avg_1d = AdaptiveAvgPool1d::<f32>::new(10); // output_size
let adaptive_max_1d = AdaptiveMaxPool1d::<f32>::new(5);

// 3D adaptive pooling  
let adaptive_avg_3d = AdaptiveAvgPool3d::<f32>::new((4, 4, 4)); // output_size
```

#### 🔄 **Enhanced RNN Architectures**

```rust
use rustorch::nn::{lstm_layer::LSTMLayer, gru_layer::GRULayer};

// Advanced LSTM with layer-wise configuration
let lstm_layer = LSTMLayer::<f32>::new(
    256,    // input_size
    512,    // hidden_size
    true,   // bias
    0.2,    // dropout
    true,   // batch_first
)?;

// Advanced GRU with optimized cell operations
let gru_layer = GRULayer::<f32>::new(
    256,    // input_size
    512,    // hidden_size
    true,   // bias
    0.2,    // dropout
    true,   // batch_first
)?;
```

### Backward Compatibility and Migration

#### 🔄 **API Evolution Path**

```rust
// Phase 3 → Phase 6 Migration Example
// Old (Phase 3): Basic attention
use rustorch::nn::attention::AttentionLayer;
let old_attention = AttentionLayer::<f32>::new(512, 8, 0.1)?;

// New (Phase 6): Full transformer attention
use rustorch::nn::transformer_phase6::MultiheadAttention;
let new_attention = MultiheadAttention::<f32>::new(
    512, 8, Some(0.1), Some(true), None, None, Some(true)
)?;
```

#### 📚 **Legacy Support**

All Phase 3-5 APIs remain fully functional with deprecation warnings guiding users to modern alternatives:

```rust
// Legacy data API (still works, but deprecated)
#[allow(deprecated)]
use rustorch::data::LegacyDataset;

// Modern Phase 5 API (recommended)
use rustorch::data::Dataset;
```

### Phase 2: Advanced Optimization Framework (COMPLETED)

The v0.5.13 release **completes Phase 2** of the RusTorch development roadmap with a comprehensive advanced optimization framework that significantly improves training efficiency, numerical stability, and code quality. This update raises PyTorch compatibility from 55% to **65%** and introduces **world-class performance optimizations**.

#### 🏆 **Major Achievements**

- **🚀 Performance Revolution**: Up to 580% performance improvement (Adamax: 33,632 steps/sec)
- **🧹 Code Quality**: 50%+ code deduplication through unified architecture
- **🔧 Architectural Excellence**: GenericAdamOptimizer foundation for all Adam variants
- **📊 Comprehensive Testing**: 159/159 tests passing with full integration coverage
- **🎯 Production Ready**: Clippy-clean code with rustfmt formatting

#### 🔧 **Technical Innovations**

##### Unified GenericAdamOptimizer Architecture

The centerpiece of Phase 2 is the **GenericAdamOptimizer<V: AdamVariant>** architecture that:

```rust
use rustorch::optim::common::{GenericAdamOptimizer, AdamVariant, AdamState, AdamConfig};

// All Adam variants now use this unified foundation:
pub type NAdam = GenericAdamOptimizer<NAdamVariant>;
pub type RAdam = GenericAdamOptimizer<RAdamVariant>;  
pub type Adamax = GenericAdamOptimizer<AdamaxVariant>;

// Shared infrastructure provides:
// - Consistent state management (AdamState)
// - Unified configuration (AdamConfig)
// - Common utilities (AdamUtils)
// - Robust error handling (RusTorchResult<T>)
```

**Architecture Benefits:**
- **50%+ Code Reduction**: Eliminated duplicate Adam implementations
- **Consistent API**: Uniform interface across all optimizers  
- **Improved Performance**: Shared optimizations benefit all variants
- **Enhanced Maintainability**: Single source of truth for Adam logic

##### Advanced Optimizer Implementations

**NAdam (Nesterov-accelerated Adam)** - 18,976+ steps/sec
```rust
let nadam = NAdam::default_params(0.002)?;  // Excellent for most tasks
// Key features: Time-dependent beta1 scheduling, superior NLP performance
```

**RAdam (Rectified Adam)** - 21,939+ steps/sec  
```rust
let radam = RAdam::default_params(0.001)?;  // Recommended for stable training
// Key features: Automatic warmup, exceptional transformer stability
```

**Adamax (Infinity Norm Adam)** - 33,632+ steps/sec ⚡ **FASTEST**
```rust
let adamax = Adamax::default_params(0.002)?;  // Ideal for sparse data
// Key features: Infinity norm, no bias correction needed, outlier handling
```

**Enhanced L-BFGS with Modular Memory System**
```rust
let lbfgs = LBFGS::with_params(
    1.0, 20, 20, 1e-5, 1e-9, 10,
    LineSearchMethod::StrongWolfe { c1: 1e-4, c2: 0.9 }
)?;
// Key features: Modular LBFGSMemory, advanced line search, convergence detection
```

#### 🛠️ **Advanced Utilities and Stability System**

##### Comprehensive Stability Framework

```rust
use rustorch::optim::utils::{StabilityConfig, OptimizerUtils, OptimizerMetrics};

// Automatic gradient stabilization
let config = StabilityConfig::default();
let stabilized_grad = config.stabilize_gradient(&grad);

// Performance monitoring  
let mut metrics = OptimizerMetrics::new(1000);
metrics.update_gradient_norm(OptimizerUtils::l2_norm(&grad));

// Issue detection
if let Some(issue) = metrics.detect_issues() {
    println!("Training issue detected: {:?}", issue);
}
```

##### Parameter Suggestion System

```rust
use rustorch::optim::utils::{OptimizerFactory, OptimizerType, ProblemType};

let suggestions = OptimizerFactory::suggest_parameters(
    OptimizerType::RAdam,
    ProblemType::ImageClassification,
    DatasetSize::Large,
    Some(0.001)
);
// Automatically suggests optimal hyperparameters
```

#### 📊 **Performance Benchmarking Framework**

```rust
use rustorch::optim::benchmarks::{OptimizerBenchmark, quick_performance_test};

// Comprehensive performance analysis
let mut benchmark = OptimizerBenchmark::new();
let results = benchmark.run_adam_comparison();

// Quick development testing
quick_performance_test(); // Shows: Adamax 33,632 steps/sec ⚡
```

#### 🎯 **Migration Guide and Best Practices**

##### Recommended Optimizer Selection

| Use Case | Recommended | Alternative | Performance |
|----------|-------------|-------------|-------------|
| **General Deep Learning** | RAdam | Adamax | 21,939 steps/sec |
| **NLP & Fine-tuning** | NAdam | RAdam | 18,976 steps/sec |
| **Sparse Features** | Adamax | NAdam | **33,632 steps/sec** |
| **Second-order** | Enhanced L-BFGS | RAdam | Variable |

##### Simple Migration

```rust
// Old (v0.5.12 and earlier)
use rustorch::optim::Adam;
let optimizer = Adam::default_params(0.001);

// New (v0.5.13 - Choose based on use case)
use rustorch::optim::RAdam;  // Recommended general upgrade
let optimizer = RAdam::default_params(0.001)?;

// With stability features
use rustorch::optim::utils::StabilityConfig;
let config = StabilityConfig::default();
optimizer.step(&param, &config.stabilize_gradient(&grad));
```

#### 🏆 **Quality Achievements**

- **Code Quality**: 
  - ✅ 0 Clippy warnings
  - ✅ Full rustfmt formatting
  - ✅ Comprehensive error handling
  
- **Testing Coverage**:
  - ✅ 159/159 tests passing (100%)
  - ✅ Integration test coverage
  - ✅ Performance benchmark validation
  
- **Performance Verified**:
  - ✅ Up to 580% speed improvement
  - ✅ 58 examples all working perfectly
  - ✅ Production-ready reliability

### Enhanced Shape Operations with Builder Pattern

The latest release introduces a comprehensive refactoring of shape operations with significant improvements:

#### Key Enhancements

- **Builder Pattern**: Chainable operations for complex tensor transformations
- **Fluent Interface**: Ergonomic API for intuitive operation sequencing  
- **Macro Support**: Concise syntax for common operation patterns
- **Enhanced Error Handling**: Detailed error messages with proper context
- **Performance Optimizations**: Generic recursive processing and helper functions
- **Zero-Copy Views**: Optimized memory usage where possible

#### Complete Shape Operations API

```rust
use rustorch::tensor::ops::shape_operations::{ShapeOps, ShapeMode, shape_ops};

// All available shape operations:
let tensor = tensor
    .shape_builder()
    .squeeze()                              // Remove all size-1 dimensions
    .squeeze_dim(1)?                        // Remove specific size-1 dimension
    .unsqueeze(0)?                          // Add dimension at position
    .flatten()?                             // Flatten to 1D
    .flatten_range(1, Some(3))?             // Flatten specific range
    .unflatten(0, &[2, 3])?                 // Reverse flatten
    .expand(&[10, 6])?                      // Expand to target shape
    .expand_as(&other_tensor)?              // Expand to match tensor
    .repeat(&[2, 1])?                       // Repeat along dimensions
    .repeat_interleave(3, Some(0))?         // Interleave elements
    .roll(2, Some(1))?                      // Roll elements
    .rot90(1, &[0, 1])?                     // 90-degree rotation
    .flip(&[0])?                            // Flip along dimensions
    .fliplr()?                              // Flip left-right
    .flipud()?                              // Flip up-down
    .view_shape(&[4, 15])?                  // Create view
    .build();
```

#### Advanced Usage Patterns

```rust
// Ownership control with ShapeMode
let result = tensor.squeeze_with_mode(ShapeMode::ViewOnly)?;  // Zero-copy guarantee
let result = tensor.expand_lazy(&[10, 10])?;                 // Lazy evaluation

// Intermediate inspection
let builder = tensor.shape_builder().squeeze().unwrap();
println!("Shape after squeeze: {:?}", builder.current_shape());
let final_tensor = builder.flatten().unwrap().build();

// Error handling with context
match tensor.unsqueeze(10) {
    Ok(result) => println!("Success"),
    Err(RusTorchError::InvalidDimension(msg)) => {
        eprintln!("Dimension error: {}", msg);  // Detailed context provided
    }
}
```

#### Migration Guide

**Before (v0.5.11 and earlier):**
```rust
let tensor = tensor.squeeze();
let tensor = tensor.unsqueeze(1).unwrap();
let tensor = tensor.flatten_owned();
```

**After (v0.5.12 - Recommended):**
```rust
// More readable and maintainable
let result = tensor
    .shape_builder()
    .squeeze().unwrap()
    .unsqueeze(1).unwrap()
    .flatten().unwrap()
    .build();

// Or use the macro for brevity
let result = shape_ops!(tensor, squeeze, unsqueeze(1), flatten).unwrap();
```

## 🔧 Sparse Tensor Module (Phase 12)

### Sparse Tensor Creation and Formats

```rust
use rustorch::sparse::{SparseTensor, SparseFormat};
use ndarray::Array1;

// COO (Coordinate) format creation
let indices = vec![
    Array1::from_vec(vec![0, 1, 2]),  // Row indices
    Array1::from_vec(vec![1, 2, 0]),  // Column indices
];
let values = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let shape = vec![3, 3];
let sparse_tensor = SparseTensor::from_coo(indices, values, shape)?;

// Convert between formats
let csr_tensor = sparse_tensor.to_csr()?;     // COO → CSR
let coo_tensor = csr_tensor.to_coo()?;        // CSR → COO
let dense_tensor = sparse_tensor.to_dense()?; // Sparse → Dense

// CSC format support (future implementation)
// CSC format provides column-major compressed storage for column-wise operations
let csc_format = SparseFormat::CSC;  // Available format type
```

### Sparse Operations

```rust
use rustorch::sparse::SparseOps;

// Sparse matrix-vector multiplication
let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let result = sparse_tensor.spmv(&vector)?;

// Sparse matrix-matrix multiplication
let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let result = sparse_tensor.spmm(&matrix)?;

// Sparse arithmetic operations
let sum = sparse_a.sparse_add(&sparse_b)?;
let product = sparse_a.sparse_mul(&sparse_b)?;
let transposed = sparse_tensor.transpose()?;
```

### Sparse Neural Network Layers

```rust
use rustorch::sparse::sparse_layers::{SparseLinear, SparseConv2d, SparseEmbedding};
use rustorch::sparse::pruning::PruningConfig;

// Sparse linear layer
let sparse_linear = SparseLinear::new(
    input_size: 784,
    output_size: 10,
    sparsity: 0.9,      // 90% sparsity
    Some(PruningConfig::magnitude(0.1))
)?;

// Sparse convolution layer
let sparse_conv = SparseConv2d::new(
    in_channels: 3,
    out_channels: 64,
    kernel_size: (3, 3),
    sparsity: 0.8,
    Some(PruningConfig::structured(0.5))
)?;

// Sparse embedding layer
let sparse_embedding = SparseEmbedding::new(
    num_embeddings: 10000,
    embedding_dim: 256,
    sparsity: 0.95
)?;
```

### Model Pruning Algorithms

```rust
use rustorch::sparse::pruning::{ModelPruner, PruningAlgorithm, PruningSchedule};

// Magnitude-based pruning
let magnitude_pruner = ModelPruner::new(PruningConfig {
    algorithm: PruningAlgorithm::Magnitude,
    sparsity_ratio: 0.8,
    schedule: PruningSchedule::Gradual { steps: 1000 },
});

// Structured pruning
let structured_pruner = ModelPruner::new(PruningConfig {
    algorithm: PruningAlgorithm::Structured,
    sparsity_ratio: 0.5,
    schedule: PruningSchedule::OneShot,
});

// Fisher information pruning
let fisher_pruner = ModelPruner::new(PruningConfig {
    algorithm: PruningAlgorithm::Fisher,
    sparsity_ratio: 0.9,
    schedule: PruningSchedule::Gradual { steps: 500 },
});

// Apply pruning to model
let pruned_weights = magnitude_pruner.prune_weights(&weights)?;
let mask = magnitude_pruner.create_mask(&weights)?;
```

### GPU-Accelerated Sparse Operations

```rust
use rustorch::sparse::gpu_ops::{CudaSparseOps, MetalSparseOps, SparseBatchProcessor};

// CUDA acceleration (if available)
#[cfg(feature = "cuda")]
{
    let cuda_ops = CudaSparseOps::init()?;
    let result = cuda_ops.spmv(&sparse_matrix, &vector)?;
    let batch_result = cuda_ops.sparse_add(&sparse_a, &sparse_b)?;
}

// Metal acceleration for Apple Silicon
#[cfg(feature = "metal")]
{
    let metal_ops = MetalSparseOps::new()?;
    let result = metal_ops.spmv(&sparse_matrix, &vector)?;
}

// Batch processing for efficiency
let mut batch_processor = SparseBatchProcessor::new(max_batch_size: 32);
batch_processor.add_to_batch(sparse_tensor1)?;
batch_processor.add_to_batch(sparse_tensor2)?;
let results = batch_processor.process_batch()?;
```

### Sparse Utilities and Analysis

```rust
use rustorch::sparse::utils::{SparseAnalyzer, SparseValidator, SparseConverter};

// Analyze sparse tensor patterns
let analyzer = SparseAnalyzer::new();
let sparsity_ratio = analyzer.sparsity_ratio(&sparse_tensor);
let pattern_analysis = analyzer.analyze_pattern(&sparse_tensor)?;
let hotspots = analyzer.identify_hotspots(&sparse_tensor, threshold: 0.1)?;

// Validate sparse tensor integrity
let validator = SparseValidator::new();
let is_valid = validator.validate_coo(&sparse_tensor)?;
let integrity_check = validator.check_integrity(&sparse_tensor)?;

// Convert and optimize formats
let converter = SparseConverter::new();
let optimized = converter.optimize_for_operations(&sparse_tensor, operation_type)?;
let converted = converter.convert_with_validation(&sparse_tensor, SparseFormat::CSR)?;

// Batch format conversion for multiple tensors
let tensors = vec![tensor1, tensor2, tensor3];
let converted_tensors = converter.batch_convert(tensors, SparseFormat::CSR)?;
```

### Advanced Sparse Layers

```rust
use rustorch::sparse::sparse_layers::{SparseAttention, SparseTransformerBlock};

// Sparse attention mechanism
let mut sparse_attention = SparseAttention::new(
    embed_dim: 512,
    num_heads: 8, 
    dropout: 0.1
);

// Set attention sparsity pattern
let attention_mask = SparseTensor::from_coo(/* attention pattern */);
sparse_attention.set_attention_mask(attention_mask)?;

// Forward pass with sparse attention
let output = sparse_attention.forward(&query, &key, &value)?;
let (avg_sparsity, active_heads) = sparse_attention.attention_stats().unwrap();

// Complete sparse transformer block
let transformer_block = SparseTransformerBlock::new(
    embed_dim: 512,
    num_heads: 8,
    ff_dim: 2048,
    dropout: 0.1,
    attention_sparsity: 0.9,
    ff_sparsity: 0.8
);

let output = transformer_block.forward(&input)?;
let sparsity_report = transformer_block.sparsity_report();
```

### Pruning Schedules and Advanced Algorithms

```rust
use rustorch::sparse::pruning::{PruningSchedule, StructuredPruner, FisherPruner, StructuredGranularity};

// Gradual pruning schedule
let mut schedule = PruningSchedule::new(
    initial_sparsity: 0.1,
    final_sparsity: 0.9,
    num_steps: 1000
);

// Schedule management
for epoch in 0..100 {
    let current_sparsity = schedule.current_sparsity();
    // Apply pruning with current sparsity
    schedule.step();
}

// Structured pruning with different granularities
let channel_pruner = StructuredPruner::new(StructuredGranularity::Channel, ratio: 0.5);
let filter_pruner = StructuredPruner::new(StructuredGranularity::Filter, ratio: 0.3);
let block_pruner = StructuredPruner::new(StructuredGranularity::Block(4), ratio: 0.7);

let pruned_weights = channel_pruner.prune_linear_weights(&weights)?;

// Fisher information pruning
let mut fisher_pruner = FisherPruner::new();
fisher_pruner.update_fisher("layer1.weight", &gradients);
let fisher_pruned = fisher_pruner.prune_with_fisher("layer1.weight", sparsity: 0.8)?;
```

### Sparse I/O and Serialization

```rust
use rustorch::sparse::utils::SparseIO;

// Save sparse tensor to binary format
SparseIO::save_binary(&sparse_tensor, "model_weights.sparse")?;

// Load sparse tensor from binary format
let loaded_tensor: SparseTensor<f32> = SparseIO::load_binary("model_weights.sparse")?;

// Efficient storage for large sparse models
let memory_saved = original_size - loaded_tensor.memory_usage();
```

### Sparse Pattern Analysis and Optimization

```rust
use rustorch::sparse::utils::{SparseAnalyzer, AccessPattern};

// Advanced pattern analysis
let analyzer = SparseAnalyzer::new();
let pattern_analysis = analyzer.analyze_pattern(&sparse_tensor);

// Access pattern optimization
let optimal_format = analyzer.suggest_optimal_format(&sparse_tensor, AccessPattern::RandomAccess);
let recommendations = pattern_analysis.optimization_recommendations();

// Performance analysis report
println!("{}", pattern_analysis.report());

// Check for optimization opportunities
if pattern_analysis.clustering_coefficient > 0.8 {
    println!("High clustering detected - consider block sparse format");
}
```

### Performance Benchmarking

```rust
use rustorch::sparse::utils::SparseBenchmark;

// Comprehensive benchmarking
let mut benchmark = SparseBenchmark::new();
benchmark.benchmark_spmv(&sparse_tensor, &vector, iterations: 1000)?;

// Compare sparse vs dense performance
let comparison = benchmark.compare_with_dense(&sparse_tensor, &dense_tensor)?;
println!("Speedup: {:.2}x", comparison.speedup_factor);

// Detailed performance report
println!("{}", benchmark.report());
```

## 🌐 WebAssembly (WASM) Module

> 📋 **Complete WASM API Reference**: [WASM API Documentation](WASM_API_DOCUMENTATION.md)

The WASM module provides comprehensive browser-based machine learning capabilities. For detailed API reference, examples, and integration guides, see the dedicated [WASM API Documentation](WASM_API_DOCUMENTATION.md).

For complete documentation and examples, visit the [examples directory](../examples/) or generate local docs:

```bash
cargo doc --open --no-deps --features "linalg,cuda,metal,wasm,webgpu"
```