# RusTorch

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)

A PyTorch-compatible deep learning library in Rust, providing tensor computation with strong GPU acceleration and deep neural networks built on a tape-based autograd system.

## Features

- **Tensor Computation**: N-dimensional tensors with GPU support (coming soon)
- **Automatic Differentiation**: Build computational graphs for automatic differentiation
- **Neural Network Building Blocks**: Pre-defined layers, loss functions, and optimization algorithms
- **PyTorch-like API**: Familiar interface for PyTorch users
- **Safe and Fast**: Leveraging Rust's safety guarantees and performance

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.1.0"
```

## Quick Start

Here's a simple example of training a linear regression model:

```rust
use rustorch::prelude::*;
use rustorch::nn::{Module, Linear};

fn main() {
    // Create a simple linear model
    let model = Linear::new(1, 1, true);
    
    // Example input and target
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
    let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]);
    
    // Convert to variables
    let x_var = Variable::new(x, true);
    let y_var = Variable::new(y, false);
    
    // Forward pass
    let output = model.forward(&x_var);
    
    // Compute loss
    let loss = (output - y_var).pow(2.0).mean();
    
    println!("Loss: {}", loss.data());
}
```

## Examples

See the [examples](examples/) directory for more complete examples:

- [Linear Regression](examples/linear_regression.rs): Simple linear regression example
- [MNIST Classifier](examples/mnist.rs): Handwritten digit classification (coming soon)

## Documentation

For detailed documentation, please refer to the [API documentation](https://docs.rs/rustorch).

## License

Licensed under either of:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
