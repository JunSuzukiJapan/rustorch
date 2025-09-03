//! Phase 8 Tensor Utilities Demo
//! フェーズ8テンソルユーティリティデモ
//!
//! Demonstrates the new tensor utility operations implemented in Phase 8.

use ndarray::ArrayD;
use rustorch::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phase 8 Tensor Utilities Demo");
    println!("フェーズ8テンソルユーティリティデモ");
    println!("================================\n");

    // Create sample tensors
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3]);
    println!("Original tensor (2x3): {:?}", tensor.data.as_slice());

    // Test conditional operations
    println!("\n1. Conditional Operations:");

    // Test masked_select
    let mask = ArrayD::from_shape_vec(vec![2, 3], vec![true, false, true, true, false, false])?;
    let selected = tensor.masked_select(&mask)?;
    println!("   masked_select result: {:?}", selected.data.as_slice());

    // Test masked_fill
    let filled = tensor.masked_fill(&mask, 999.0)?;
    println!("   masked_fill result: {:?}", filled.data.as_slice());

    // Test index operations
    println!("\n2. Index Operations:");

    let index = ArrayD::from_shape_vec(vec![2], vec![0i64, 2])?;
    let gathered = tensor.gather(1, &index)?;
    println!("   gather result: {:?}", gathered.data.as_slice());

    let selected_idx = tensor.index_select(1, &index)?;
    println!("   index_select result: {:?}", selected_idx.data.as_slice());

    // Test statistical operations
    println!("\n3. Statistical Operations:");

    let (top_values, _top_indices) = tensor.topk_util(2, 1, true, true)?;
    println!("   topk values: {:?}", top_values.data.as_slice());

    let (kth_val, _kth_idx) = tensor.kthvalue(1, 1, false)?;
    println!("   kthvalue: {:?}", kth_val.data.as_slice());

    // Test advanced operations
    println!("\n4. Advanced Operations:");

    let data2 = vec![1.0f32, 2.0, 1.0, 3.0, 2.0];
    let tensor2 = Tensor::from_vec(data2, vec![5]);
    let (unique_vals, _inv, _counts) = tensor2.unique(true, false, false)?;
    println!("   unique values: {:?}", unique_vals.data.as_slice());

    let (hist_counts, hist_edges) = tensor.histogram(3, Some((1.0, 6.0)))?;
    println!("   histogram counts: {:?}", hist_counts.as_slice());
    let _hist_counts = &hist_counts; // Use variable to avoid warning
    println!("   histogram edges: {:?}", hist_edges.data.as_slice());

    println!("\nPhase 8 implementation completed successfully!");
    Ok(())
}
