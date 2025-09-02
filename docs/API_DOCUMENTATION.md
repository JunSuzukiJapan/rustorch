# RusTorch API Documentation

## 📚 Complete API Reference

This document provides comprehensive API documentation for RusTorch v0.5.12, organized by module and functionality.

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
└── wasm/                # WebAssembly bindings
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

## 🌐 WebAssembly Support

### WASM Bindings

```rust
use rustorch::wasm::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<f32>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor {
            inner: Tensor::from_vec(data.to_vec(), shape.to_vec()),
        }
    }

    #[wasm_bindgen]
    pub fn add(&self, other: &WasmTensor) -> WasmTensor {
        WasmTensor {
            inner: self.inner.add(&other.inner),
        }
    }

    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f32> {
        self.inner.to_vec()
    }
}

// Neural network for WASM
#[wasm_bindgen]
pub struct WasmModel {
    model: Sequential<f32>,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmModel {
        let model = Sequential::<f32>::new()
            .add_layer(Box::new(Linear::<f32>::new(2, 10)))
            .add_activation(Box::new(ReLU::<f32>::new()))
            .add_layer(Box::new(Linear::<f32>::new(10, 1)));
        
        WasmModel { model }
    }

    #[wasm_bindgen]
    pub fn predict(&self, input: &[f32]) -> Vec<f32> {
        let input_tensor = Tensor::from_vec(input.to_vec(), vec![1, input.len()]);
        let output = self.model.forward(&input_tensor);
        output.to_vec()
    }
}
```

### WebGPU Integration

```rust
use rustorch::gpu::webgpu::*;

// WebGPU context
let webgpu_context = WebGpuContext::new().await?;
let device = webgpu_context.device();
let queue = webgpu_context.queue();

// WebGPU tensors
let webgpu_tensor = WebGpuTensor::<f32>::new(&device, &[1024, 1024]);
let result = webgpu_tensor.matmul(&other_webgpu_tensor).await?;

// Shader-based operations
let custom_shader = webgpu_context.create_compute_shader(shader_source)?;
let result = custom_shader.execute(&[&input_tensor], &output_shape).await?;
```

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

### Error Types

```rust
use rustorch::{RusTorchError, Result};

// Main error types
match tensor_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(RusTorchError::InvalidShape(msg)) => eprintln!("Shape error: {}", msg),
    Err(RusTorchError::InvalidDimension(msg)) => eprintln!("Dimension error: {}", msg),
    Err(RusTorchError::ComputationError(msg)) => eprintln!("Computation error: {}", msg),
    Err(RusTorchError::GpuError(msg)) => eprintln!("GPU error: {}", msg),
    Err(RusTorchError::MemoryError(msg)) => eprintln!("Memory error: {}", msg),
    Err(RusTorchError::DeviceError(msg)) => eprintln!("Device error: {}", msg),
    Err(RusTorchError::SerializationError(msg)) => eprintln!("Serialization error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

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
| `nn` | Neural network layers | Linear, Conv, RNN, LSTM, GRU, activations, normalization, loss functions |
| `autograd` | Automatic differentiation | Variables, gradients, custom functions, computation graphs |
| `optim` | Optimizers and schedulers | SGD, Adam, AdamW, RMSprop, learning rate scheduling |
| `special` | Special mathematical functions | Gamma, Bessel, error functions, hypergeometric, elliptic integrals |
| `distributions` | Statistical distributions | Normal, uniform, gamma, beta, categorical, multivariate |
| `vision` | Computer vision utilities | Image transforms, data augmentation, preprocessing |
| `linalg` | Linear algebra operations | SVD, QR, eigenvalues, matrix decomposition, solving |
| `gpu` | GPU acceleration | CUDA, Metal, OpenCL, WebGPU, memory management |
| `fft` | Fourier transforms | FFT, RFFT, 2D FFT, N-dimensional FFT, window functions |
| `wasm` | WebAssembly bindings | Browser support, JavaScript integration |

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | Core tensor operations | None |
| `linalg` | Linear algebra with BLAS/LAPACK | OpenBLAS, LAPACK |
| `cuda` | NVIDIA GPU acceleration | CUDA Runtime |
| `metal` | Apple GPU acceleration | Metal framework (macOS) |
| `opencl` | OpenCL GPU acceleration | OpenCL drivers |
| `webgpu` | WebGPU browser acceleration | WebGPU API |
| `wasm` | WebAssembly compilation | wasm-bindgen |
| `model-hub` | Model downloading and caching | HTTP client, crypto |
| `safetensors` | SafeTensors format support | Memory mapping |
| `onnx` | ONNX model import/export | ONNX Runtime |
| `python` | Python bindings | PyO3 |

## 🆕 New Features in v0.5.12

### Phase 2: Advanced Optimizers

The v0.5.12 release introduces Phase 2 of the RusTorch development roadmap with advanced optimization algorithms that significantly improve training efficiency and convergence behavior. This update raises PyTorch compatibility from 55% to 65%.

#### New Advanced Optimizers

##### NAdam (Nesterov-accelerated Adam)
```rust
use rustorch::optim::NAdam;

// Basic usage - recommended for most applications
let mut optimizer = NAdam::default_params(0.002);

// With weight decay for regularization
let mut optimizer = NAdam::with_weight_decay(0.001, 0.01);

// Fine-tuned parameters for advanced users
let mut optimizer = NAdam::new(
    0.002,    // learning_rate
    0.9,      // beta1 (momentum coefficient)
    0.999,    // beta2 (RMSprop coefficient)
    1e-8,     // eps (numerical stability)
    0.01,     // weight_decay
    0.004,    // momentum_decay (NAdam-specific)
    0.004,    // schedule_decay (NAdam-specific)
);

// Training step
optimizer.step(&param, &grad);
```

**Key Benefits:**
- Combines Adam with Nesterov momentum for faster convergence
- Superior performance on NLP and fine-tuning tasks
- Time-dependent beta1 scheduling improves late-stage training

##### RAdam (Rectified Adam)
```rust
use rustorch::optim::RAdam;

// Basic usage - excellent default choice
let mut optimizer = RAdam::default_params(0.001);

// With weight decay
let mut optimizer = RAdam::with_weight_decay(0.001, 0.01);

// Custom configuration
let mut optimizer = RAdam::new(
    0.001,    // learning_rate
    0.9,      // beta1
    0.999,    // beta2
    1e-8,     // eps
    0.01,     // weight_decay
);
```

**Key Benefits:**
- Automatically handles Adam's variance issue in early training
- No manual warmup scheduling required
- Exceptional stability for transformer architectures
- Falls back to momentum-only updates when variance is not rectifiable

##### Adamax (Adam with Infinity Norm)
```rust
use rustorch::optim::Adamax;

// Basic usage - ideal for sparse data
let mut optimizer = Adamax::default_params(0.002);

// With weight decay
let mut optimizer = Adamax::with_weight_decay(0.001, 0.01);

// Custom parameters
let mut optimizer = Adamax::new(
    0.002,    // learning_rate  
    0.9,      // beta1
    0.999,    // beta2
    1e-7,     // eps (smaller default for stability)
    0.01,     // weight_decay
);
```

**Key Benefits:**
- Uses infinity norm instead of L2 norm for second moment
- More stable than Adam for embeddings and sparse features
- No bias correction needed for second moment estimation
- Better handling of outlier gradients

##### Enhanced L-BFGS with Advanced Line Search
```rust
use rustorch::optim::{LBFGS, LineSearchMethod};

// Enhanced second-order optimization
let mut optimizer = LBFGS::with_params(
    1.0,                              // learning_rate
    20,                               // max_iter
    20,                               // max_eval  
    1e-5,                             // tolerance_grad
    1e-9,                             // tolerance_change
    10,                               // history_size
    LineSearchMethod::StrongWolfe,    // line_search_fn
);

// Available line search strategies
let backtrack_lbfgs = LBFGS::with_params(
    1.0, 20, 20, 1e-5, 1e-9, 10,
    LineSearchMethod::Backtracking
);

let fixed_step_lbfgs = LBFGS::with_params(
    0.1, 20, 20, 1e-5, 1e-9, 10,
    LineSearchMethod::None
);
```

**Enhanced Features:**
- Improved Strong Wolfe line search with better convergence
- Enhanced Armijo backtracking with numerical safeguards
- Better memory management for large-scale problems
- Robust convergence detection

#### Memory Optimization and Numerical Stability

New utility system for robust and memory-efficient training:

```rust
use rustorch::optim::utils::{OptimizerUtils, StabilityConfig, OptimizerState};

// Gradient stabilization
let clipped_grad = OptimizerUtils::clip_gradient_norm(&grad, 1.0);
let stable_grad = OptimizerUtils::sanitize_tensor(&grad, 0.0, 1e6);

// Comprehensive stability configuration
let config = StabilityConfig {
    min_eps: 1e-8,                    // Numerical stability threshold
    max_grad_norm: 10.0,              // Gradient clipping threshold
    max_param_value: 1e6,             // Parameter value limit
    auto_nan_correction: true,        // Automatic NaN/Inf handling
    gradient_clipping: true,          // Enable gradient clipping
};

// Apply stability measures
let stabilized_grad = config.stabilize_gradient(&grad);
let stabilized_param = config.stabilize_parameter(&param);

// Memory-efficient state management
let mut opt_state = OptimizerState::new(500); // 500MB memory limit
opt_state.init_momentum(param_id, param.shape());
opt_state.init_velocity(param_id, param.shape());

// Automatic cleanup of stale states
if opt_state.get_step() % 1000 == 0 {
    opt_state.cleanup_stale_states(10000); // Remove states older than 10k steps
    println!("Memory usage: {}MB", opt_state.estimate_memory_mb());
}
```

#### Advanced Usage Patterns

##### Combining Optimizers with Stability Features
```rust
use rustorch::optim::{RAdam, utils::StabilityConfig};

let mut optimizer = RAdam::with_weight_decay(0.001, 0.01);
let stability_config = StabilityConfig::default();

// In your training loop
for (param, grad) in params_and_grads {
    let stable_grad = stability_config.stabilize_gradient(&grad);
    optimizer.step(&param, &stable_grad);
}
```

##### Memory-Efficient Large Model Training
```rust
use rustorch::optim::{NAdam, utils::{OptimizerState, OptimizerUtils}};

let mut optimizer = NAdam::default_params(0.002);
let mut state_manager = OptimizerState::new(1000); // 1GB limit

for step in 0..num_steps {
    // Your forward/backward pass here
    
    // Efficient gradient processing
    let processed_grad = OptimizerUtils::clip_gradient_norm(&grad, 5.0);
    optimizer.step(&param, &processed_grad);
    
    // Periodic maintenance
    state_manager.step();
    
    if step % 5000 == 0 {
        println!("Step {}: Memory usage {}MB", step, state_manager.estimate_memory_mb());
    }
}
```

#### Performance Comparison

| Optimizer | Convergence Speed | Memory Usage | Best Use Cases |
|-----------|------------------|--------------|----------------|
| **NAdam** | ⚡ Fast initial | 🔵 Medium | NLP, Fine-tuning, Quick prototyping |
| **RAdam** | 🔄 Stable warmup | 🔵 Medium | Transformers, Large models, Stable training |
| **Adamax** | 📈 Consistent | 🟢 Low | Sparse features, Embeddings, Outlier-prone data |
| **Enhanced L-BFGS** | 🎯 Variable | 🔴 High | Small datasets, Second-order optimization |

#### Migration Guide

**Upgrading from v0.5.11:**

```rust
// Old: Basic Adam optimization
use rustorch::optim::Adam;
let mut optimizer = Adam::default_params(0.001);

// New: Choose the best optimizer for your use case

// For general deep learning (recommended upgrade)
use rustorch::optim::RAdam;
let mut optimizer = RAdam::default_params(0.001);

// For NLP and fine-tuning tasks
use rustorch::optim::NAdam;
let mut optimizer = NAdam::default_params(0.002);

// For sparse data and embeddings
use rustorch::optim::Adamax;
let mut optimizer = Adamax::default_params(0.002);
```

**Adding stability features:**

```rust
// Old: Basic optimization
optimizer.step(&param, &grad);

// New: With stability features
use rustorch::optim::utils::StabilityConfig;
let config = StabilityConfig::default();
optimizer.step(&param, &config.stabilize_gradient(&grad));
```

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

### Backward Compatibility

All existing APIs remain fully functional. The new builder pattern and enhanced operations are additive features that don't break existing code.

For complete documentation and examples, visit the [examples directory](../examples/) or generate local docs:

```bash
cargo doc --open --no-deps --features "linalg,cuda,metal"
```