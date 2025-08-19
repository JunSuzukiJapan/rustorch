/// Demonstration of tensor statistical operations
/// テンソル統計演算のデモンストレーション

use rustorch::tensor::Tensor;

fn main() {
    println!("=== RusTorch Statistics Demo ===\n");
    
    // Create a 2D tensor for testing
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4]
    );
    println!("Original matrix (3x4):");
    println!("Shape: {:?}", data.shape());
    if let Some(slice) = data.as_slice() {
        for i in 0..3 {
            let row: Vec<f32> = slice[i*4..(i+1)*4].to_vec();
            println!("  {:?}", row);
        }
    }
    
    // Test mean calculation
    println!("\n--- Mean Calculation ---");
    let mean_all = data.mean(None);
    println!("Overall mean: {:?}", mean_all.as_slice().unwrap());
    
    let mean_axis0 = data.mean(Some(0));
    println!("Mean along axis 0 (columns): {:?}", mean_axis0.as_slice().unwrap());
    
    let mean_axis1 = data.mean(Some(1));
    println!("Mean along axis 1 (rows): {:?}", mean_axis1.as_slice().unwrap());
    
    // Test variance calculation
    println!("\n--- Variance Calculation ---");
    let var_all = data.var(None, false);
    println!("Overall variance (biased): {:?}", var_all.as_slice().unwrap());
    
    let var_unbiased = data.var(None, true);
    println!("Overall variance (unbiased): {:?}", var_unbiased.as_slice().unwrap());
    
    let var_axis0 = data.var(Some(0), false);
    println!("Variance along axis 0: {:?}", var_axis0.as_slice().unwrap());
    
    // Test standard deviation
    println!("\n--- Standard Deviation ---");
    let std_all = data.std(None, false);
    println!("Overall std (biased): {:?}", std_all.as_slice().unwrap());
    
    let std_axis1 = data.std(Some(1), true);
    println!("Std along axis 1 (unbiased): {:?}", std_axis1.as_slice().unwrap());
    
    // Test median calculation
    println!("\n--- Median Calculation ---");
    let median_all = data.median(None);
    println!("Overall median: {:?}", median_all.as_slice().unwrap());
    
    let median_axis0 = data.median(Some(0));
    println!("Median along axis 0: {:?}", median_axis0.as_slice().unwrap());
    
    // Test quantile calculation
    println!("\n--- Quantile Calculation ---");
    let q25 = data.quantile(0.25, None);
    println!("25th percentile: {:?}", q25.as_slice().unwrap());
    
    let q75 = data.quantile(0.75, None);
    println!("75th percentile: {:?}", q75.as_slice().unwrap());
    
    // Test cumulative sum
    println!("\n--- Cumulative Sum ---");
    match data.cumsum(1) {
        Ok(cumsum) => {
            println!("Cumulative sum along axis 1:");
            println!("Shape: {:?}", cumsum.shape());
            if let Some(slice) = cumsum.as_slice() {
                for i in 0..3 {
                    let row: Vec<f32> = slice[i*4..(i+1)*4].to_vec();
                    println!("  {:?}", row);
                }
            }
        }
        Err(e) => println!("Cumsum failed: {}", e),
    }
    
    // Test covariance matrix (for 2D data)
    println!("\n--- Covariance Matrix ---");
    let cov_matrix = data.cov();
    println!("Covariance matrix shape: {:?}", cov_matrix.shape());
    if let Some(slice) = cov_matrix.as_slice() {
        let n = cov_matrix.shape()[0];
        for i in 0..n {
            let row: Vec<f32> = slice[i*n..(i+1)*n].to_vec();
            println!("  {:?}", row);
        }
    }
    
    // Test correlation matrix
    println!("\n--- Correlation Matrix ---");
    let corr_matrix = data.corrcoef();
    println!("Correlation matrix shape: {:?}", corr_matrix.shape());
    if let Some(slice) = corr_matrix.as_slice() {
        let n = corr_matrix.shape()[0];
        for i in 0..n {
            let row: Vec<f32> = slice[i*n..(i+1)*n].to_vec();
            println!("  {:?}", row);
        }
    }
    
    println!("\n=== Statistics demo completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        
        let mean_all = tensor.mean(None);
        assert!((mean_all.as_slice().unwrap()[0] - 2.5).abs() < 1e-6);
        
        let mean_axis0 = tensor.mean(Some(0));
        assert_eq!(mean_axis0.shape(), &[2]);
        assert!((mean_axis0.as_slice().unwrap()[0] - 2.0).abs() < 1e-6);
        assert!((mean_axis0.as_slice().unwrap()[1] - 3.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_variance() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let var_biased = tensor.var(None, false);
        assert!((var_biased.as_slice().unwrap()[0] - 1.25).abs() < 1e-6);
        
        let var_unbiased = tensor.var(None, true);
        assert!((var_unbiased.as_slice().unwrap()[0] - 1.666667).abs() < 1e-5);
    }
    
    #[test]
    fn test_std() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        
        let std_val = tensor.std(None, false);
        assert!((std_val.as_slice().unwrap()[0] - 1.118034).abs() < 1e-5);
    }
}
