/// Demonstration of tensor indexing and slicing operations
/// テンソルインデックス・スライシング演算のデモンストレーション

use rustorch::tensor::{Tensor, TensorIndex};

fn main() {
    println!("=== RusTorch Indexing Demo ===\n");
    
    // Create a 2D tensor for testing
    let matrix = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4]
    );
    println!("Original matrix (3x4):");
    println!("Shape: {:?}", matrix.shape());
    if let Some(data) = matrix.as_slice() {
        for i in 0..3 {
            let row: Vec<f32> = data[i*4..(i+1)*4].to_vec();
            println!("  {:?}", row);
        }
    }
    
    // Test basic indexing (using direct array access)
    println!("\n--- Basic Indexing ---");
    if let Some(data) = matrix.as_slice() {
        let element = data[1 * 4 + 2]; // Row 1, Column 2
        println!("Element at [1, 2]: {}", element);
    }
    
    // Test select operation
    println!("\n--- Select Operation ---");
    match matrix.select(0, &[1]) {
        Ok(selected) => {
            println!("Selected row 1: {:?}", selected.as_slice().unwrap());
        }
        Err(e) => println!("Select failed: {}", e),
    }
    
    // Test multiple row selection
    println!("\n--- Multiple Row Selection ---");
    match matrix.select(0, &[0, 2]) {
        Ok(selected) => {
            println!("Selected rows [0, 2]:");
            println!("Shape: {:?}", selected.shape());
            if let Some(data) = selected.as_slice() {
                for i in 0..2 {
                    let row: Vec<f32> = data[i*4..(i+1)*4].to_vec();
                    println!("  {:?}", row);
                }
            }
        }
        Err(e) => println!("Select failed: {}", e),
    }
    
    // Test column selection
    println!("\n--- Column Selection ---");
    match matrix.select(1, &[0, 2, 1]) {
        Ok(selected) => {
            println!("Selected columns [0, 2, 1]:");
            println!("Shape: {:?}", selected.shape());
            if let Some(data) = selected.as_slice() {
                for i in 0..3 {
                    let row: Vec<f32> = data[i*3..(i+1)*3].to_vec();
                    println!("  {:?}", row);
                }
            }
        }
        Err(e) => println!("Column select failed: {}", e),
    }
    
    println!("\n=== Indexing demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_indexing() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        let element = tensor.get_item(&[TensorIndex::Single(1), TensorIndex::Single(0)]).unwrap();
        assert_eq!(element.as_slice().unwrap()[0], 3.0);
    }
    
    #[test]
    fn test_range_slicing() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        let sliced = tensor.get_item(&[
            TensorIndex::Range(Some(0), Some(1), Some(1)),
            TensorIndex::Range(Some(1), Some(3), Some(1))
        ]).unwrap();
        
        assert_eq!(sliced.shape(), &[1, 2]);
        assert_eq!(sliced.as_slice().unwrap(), &[2.0, 3.0]);
    }
    
    #[test]
    fn test_select() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        
        let selected = tensor.select(0, 1).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }
}
