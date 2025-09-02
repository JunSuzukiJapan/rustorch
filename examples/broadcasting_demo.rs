/// Demonstration of tensor broadcasting operations
/// テンソルブロードキャスティング演算のデモンストレーション
use rustorch::tensor::Tensor;

fn main() {
    println!("=== RusTorch Broadcasting Demo ===\n");

    // Test basic broadcasting compatibility
    let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    let tensor2 = Tensor::from_vec(vec![4.0f32], vec![1]);

    println!(
        "Tensor1 shape: {:?}, data: {:?}",
        tensor1.shape(),
        tensor1.as_slice().unwrap()
    );
    println!(
        "Tensor2 shape: {:?}, data: {:?}",
        tensor2.shape(),
        tensor2.as_slice().unwrap()
    );

    // Check if they can be broadcasted
    if let Ok((broadcasted1, broadcasted2)) = tensor1.broadcast_with(&tensor2) {
        println!("✓ Tensors were broadcasted together successfully");
        println!(
            "Broadcasted tensor1: {:?}",
            broadcasted1.as_slice().unwrap()
        );
        println!(
            "Broadcasted tensor2: {:?}",
            broadcasted2.as_slice().unwrap()
        );
    } else {
        println!("✗ Tensors could not be broadcasted together");
    }

    // Test unsqueeze and squeeze operations
    println!("\n--- Unsqueeze and Squeeze ---");
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    println!("Original tensor shape: {:?}", tensor.shape());

    if let Ok(unsqueezed) = tensor.unsqueeze(0) {
        println!("After unsqueeze(0): {:?}", unsqueezed.shape());
        let squeezed = unsqueezed.squeeze();
        println!("After squeeze: {:?}", squeezed.shape());
    }

    println!("\n=== Broadcasting demo completed ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_broadcasting() {
        let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::from_vec(vec![4.0f32], vec![1]);

        if let Ok((broadcasted1, broadcasted2)) = tensor1.broadcast_with(&tensor2) {
            assert_eq!(broadcasted1.shape(), &[3]);
            assert_eq!(broadcasted2.shape(), &[3]);
            assert_eq!(broadcasted2.as_slice().unwrap(), &[4.0, 4.0, 4.0]);
        } else {
            panic!("Broadcasting failed");
        }
    }
}
