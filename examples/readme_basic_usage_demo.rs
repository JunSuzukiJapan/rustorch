use rustorch::error::RusTorchResult;
use rustorch::optim::{AnnealStrategy, OneCycleLR, WarmupScheduler, SGD};
use rustorch::{tensor, tensor::Tensor};

fn main() -> RusTorchResult<()> {
    // Create tensors with convenient macro syntax
    let a = tensor!([[1, 2], [3, 4]]);
    let b = tensor!([[5, 6], [7, 8]]);

    // Basic operations with operator overloads
    let c = &a + &b; // Element-wise addition
    let d = &a - &b; // Element-wise subtraction
    let e = &a * &b; // Element-wise multiplication
    let f = &a / &b; // Element-wise division

    // Scalar operations
    let g = &a + 10.0; // Add scalar to all elements
    let h = &a * 2.0; // Multiply by scalar

    // Mathematical functions
    let exp_result = a.exp(); // Exponential function
    let ln_result = a.ln(); // Natural logarithm
    let sin_result = a.sin(); // Sine function
    let sqrt_result = a.sqrt(); // Square root

    // Matrix operations
    let matmul_result = a.matmul(&b); // Matrix multiplication

    // Linear algebra operations (requires linalg feature)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd(); // SVD decomposition
        let qr_result = a.qr(); // QR decomposition
        let eig_result = a.eigh(); // Eigenvalue decomposition
    }

    // Advanced optimizers with learning rate scheduling
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Warmup to 0.1 over 5 epochs

    // One-cycle learning rate policy
    let optimizer2 = SGD::new(0.01);
    let mut one_cycle = OneCycleLR::new(optimizer2, 1.0, 100, 0.3, AnnealStrategy::Cos);

    println!("Shape: {:?}", c.shape());
    println!("Result: {:?}", c.as_slice());

    Ok(())
}
