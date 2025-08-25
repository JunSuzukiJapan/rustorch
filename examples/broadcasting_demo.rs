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
    if tensor1.can_broadcast_with(&tensor2) {
        println!("✓ Tensors can be broadcasted together");

        // Perform broadcasting
        match tensor1.broadcast_with(&tensor2) {
            Ok((broadcasted1, broadcasted2)) => {
                println!(
                    "Broadcasted tensor1: {:?}",
                    broadcasted1.as_slice().unwrap()
                );
                println!(
                    "Broadcasted tensor2: {:?}",
                    broadcasted2.as_slice().unwrap()
                );
            }
            Err(e) => println!("Broadcasting failed: {}", e),
        }
    } else {
        println!("✗ Tensors cannot be broadcasted together");
    }

    // Test broadcast_to specific shape
    println!("\n--- Broadcast to specific shape ---");
    let small_tensor = Tensor::from_vec(vec![5.0f32], vec![1]);
    println!(
        "Small tensor shape: {:?}, data: {:?}",
        small_tensor.shape(),
        small_tensor.as_slice().unwrap()
    );

    match small_tensor.broadcast_to(&[4]) {
        Ok(broadcasted) => {
            println!("Broadcasted to [4]: {:?}", broadcasted.as_slice().unwrap());
        }
        Err(e) => println!("Broadcast failed: {}", e),
    }

    // Test 2D broadcasting
    println!("\n--- 2D Broadcasting ---");
    let matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let row_vector = Tensor::from_vec(vec![10.0f32, 20.0], vec![1, 2]);

    println!("Matrix shape: {:?}", matrix.shape());
    println!("Row vector shape: {:?}", row_vector.shape());

    if matrix.can_broadcast_with(&row_vector) {
        println!("✓ Matrix and row vector can be broadcasted");
        match matrix.broadcast_with(&row_vector) {
            Ok((broadcasted_matrix, broadcasted_row)) => {
                println!("Broadcasted matrix shape: {:?}", broadcasted_matrix.shape());
                println!("Broadcasted row shape: {:?}", broadcasted_row.shape());
            }
            Err(e) => println!("Broadcasting failed: {}", e),
        }
    }

    // Test unsqueeze and squeeze operations
    println!("\n--- Unsqueeze and Squeeze ---");
    let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
    println!("Original tensor shape: {:?}", tensor.shape());
    match tensor.unsqueeze(0) {
        Ok(unsqueezed) => {
            println!("After unsqueeze(0): {:?}", unsqueezed.shape());

            let squeezed = unsqueezed.squeeze();
            println!("After squeeze: {:?}", squeezed.shape());
        }
        Err(e) => println!("Unsqueeze failed: {}", e),
    }

    // Test repeat operation
    println!("\n--- Repeat Operation ---");
    let small = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
    println!("Small tensor: {:?}", small.as_slice().unwrap());

    match small.repeat(&[3]) {
        Ok(repeated) => {
            println!("Repeated 3 times: {:?}", repeated.as_slice().unwrap());
        }
        Err(e) => println!("Repeat failed: {}", e),
    }

    println!("\n=== Broadcasting demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_broadcasting() {
        let tensor1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::from_vec(vec![4.0f32], vec![1]);

        assert!(tensor1.can_broadcast_with(&tensor2));

        let (broadcasted1, broadcasted2) = tensor1.broadcast_with(&tensor2).unwrap();
        assert_eq!(broadcasted1.shape(), &[3]);
        assert_eq!(broadcasted2.shape(), &[3]);
        assert_eq!(broadcasted2.as_slice().unwrap(), &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_broadcast_to() {
        let tensor = Tensor::from_vec(vec![5.0f32], vec![1]);
        let broadcasted = tensor.broadcast_to(&[4]).unwrap();

        assert_eq!(broadcasted.shape(), &[4]);
        assert_eq!(broadcasted.as_slice().unwrap(), &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);

        let unsqueezed = tensor.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);

        let squeezed = unsqueezed.squeeze();
        assert_eq!(squeezed.shape(), &[3]);
    }
}
