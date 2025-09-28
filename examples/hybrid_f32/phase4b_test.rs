//! フェーズ4B条件操作・フィルタリングテスト例
//! Phase 4B Conditional Operations & Filtering Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::F32Tensor;
    use std::time::Instant;

    rustorch::hybrid_f32_experimental!();

    println!("🔍 フェーズ4B条件操作・フィルタリングテスト");
    println!("🔍 Phase 4B Conditional Operations & Filtering Test");
    println!("==============================================\n");

    // ===== 条件演算デモ / Conditional Operations Demo =====
    println!("⚖️  1. 条件演算デモ / Conditional Operations Demo");
    println!("-------------------------------------------");

    let data_a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    let data_b = F32Tensor::from_vec(vec![2.5, 1.5, 3.5, 2.0, 6.0], vec![5])?;
    let condition = F32Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], vec![5])?;

    println!("  Data A: {:?}", data_a.as_slice());
    println!("  Data B: {:?}", data_b.as_slice());
    println!("  Condition: {:?}", condition.as_slice());

    let where_result = data_a.where_condition(&condition, &data_b)?;
    println!("  where_condition: {:?}", where_result.as_slice());

    let mask = F32Tensor::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0], vec![5])?;
    let selected = data_a.masked_select(&mask)?;
    println!("  masked_select: {:?}", selected.as_slice());

    let filled = data_a.masked_fill(&mask, 99.0)?;
    println!("  masked_fill (99.0): {:?}", filled.as_slice());

    let clamped = data_a.clamp(Some(2.0), Some(4.0))?;
    println!("  clamp [2.0, 4.0]: {:?}", clamped.as_slice());

    // 論理演算
    let bool_a = F32Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4])?;
    let bool_b = F32Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], vec![4])?;

    let and_result = bool_a.logical_and(&bool_b)?;
    let or_result = bool_a.logical_or(&bool_b)?;
    let not_result = bool_a.logical_not()?;

    println!("  Logical AND: {:?}", and_result.as_slice());
    println!("  Logical OR: {:?}", or_result.as_slice());
    println!("  Logical NOT: {:?}", not_result.as_slice());

    // 比較演算
    let greater = data_a.greater(&data_b)?;
    let equal = data_a.equal(&data_a)?;
    println!("  Greater than: {:?}", greater.as_slice());
    println!("  Equal: {:?}", equal.as_slice());

    // ===== フィルタリング・マスク操作デモ / Filtering & Masking Demo =====
    println!("\n🔍 2. フィルタリング・マスク操作デモ / Filtering & Masking Demo");
    println!("-------------------------------------------------------");

    let test_data = F32Tensor::from_vec(
        vec![
            1.0,
            0.0,
            3.0,
            f32::NAN,
            5.0,
            f32::INFINITY,
            -2.0,
            f32::NEG_INFINITY,
        ],
        vec![8],
    )?;
    println!("  Test data: {:?}", test_data.as_slice());

    let filtered = test_data.filter(|x| x > 2.0)?;
    println!("  Filtered (>2.0): {:?}", filtered.as_slice());

    let nonzero = test_data.nonzero()?;
    println!("  Non-zero indices: {:?}", nonzero.as_slice());

    let nan_mask = test_data.isnan()?;
    let inf_mask = test_data.isinf()?;
    let finite_mask = test_data.isfinite()?;

    println!("  NaN mask: {:?}", nan_mask.as_slice());
    println!("  Inf mask: {:?}", inf_mask.as_slice());
    println!("  Finite mask: {:?}", finite_mask.as_slice());

    let clean_data = test_data.nan_to_num(Some(0.0), Some(f32::MAX), Some(f32::MIN))?;
    println!("  NaN to num: {:?}", clean_data.as_slice());

    let dropped_nan = test_data.drop_nan()?;
    println!("  Drop NaN: {:?}", dropped_nan.as_slice());

    let threshold_data = F32Tensor::from_vec(vec![-1.0, 0.5, 1.5, 2.5], vec![4])?;
    let thresholded = threshold_data.threshold(1.0, 0.0)?;
    println!("  Threshold (1.0): {:?}", thresholded.as_slice());

    let relu_mask = threshold_data.relu_mask()?;
    println!("  ReLU mask: {:?}", relu_mask.as_slice());

    // ===== 検索・インデックス操作デモ / Search & Indexing Demo =====
    println!("\n🔎 3. 検索・インデックス操作デモ / Search & Indexing Demo");
    println!("----------------------------------------------------");

    let search_data = F32Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], vec![7])?;
    println!("  Search data: {:?}", search_data.as_slice());

    let max_idx = search_data.argmax(None)?;
    let min_idx = search_data.argmin(None)?;
    println!("  Max index: {:?}", max_idx.as_slice());
    println!("  Min index: {:?}", min_idx.as_slice());

    let value_indices = search_data.argwhere(|x| x > 3.0)?;
    println!("  Indices where >3.0: {:?}", value_indices.as_slice());

    // ソート済み配列での検索
    let sorted_data = F32Tensor::from_vec(vec![1.0, 2.0, 4.0, 7.0, 8.0], vec![5])?;
    let search_vals = F32Tensor::from_vec(vec![3.0, 5.0, 9.0], vec![3])?;
    let search_result = sorted_data.searchsorted(&search_vals, "left")?;
    println!("  Searchsorted: {:?}", search_result.as_slice());

    // ヒストグラム
    let hist_data =
        F32Tensor::from_vec(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0], vec![9])?;
    let (hist, bins) = hist_data.histogram(5, None)?;
    println!("  Histogram: {:?}", hist.as_slice());
    println!("  Bin edges: {:?}", bins.as_slice());

    // ピーク・谷検出
    let signal = F32Tensor::from_vec(vec![1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 1.0, 2.5, 0.5], vec![9])?;
    let peaks = signal.find_peaks(Some(2.5), Some(2))?;
    let valleys = signal.find_valleys(Some(2.0), Some(2))?;
    println!("  Signal: {:?}", signal.as_slice());
    println!("  Peaks: {:?}", peaks.as_slice());
    println!("  Valleys: {:?}", valleys.as_slice());

    // ゼロ交差
    let wave = F32Tensor::from_vec(vec![1.0, 0.5, -0.3, -1.0, 0.2, 0.8], vec![6])?;
    let zeros = wave.find_zeros(None)?;
    println!("  Wave: {:?}", wave.as_slice());
    println!("  Zero crossings: {:?}", zeros.as_slice());

    // ===== 選択・置換操作デモ / Selection & Replacement Demo =====
    println!("\n📝 4. 選択・置換操作デモ / Selection & Replacement Demo");
    println!("---------------------------------------------");

    let base_data = F32Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5])?;
    let indices = F32Tensor::from_vec(vec![0.0, 2.0, 4.0], vec![3])?;

    let taken = base_data.take(&indices)?;
    println!("  Original: {:?}", base_data.as_slice());
    println!("  Take indices [0,2,4]: {:?}", taken.as_slice());

    let replaced = base_data.where_replace(|x| x > 25.0, 99.0)?;
    println!("  Replace >25 with 99: {:?}", replaced.as_slice());

    // 範囲置換
    let range_replaced = base_data.replace_range(20.0, 40.0, 77.0)?;
    println!("  Replace [20,40] with 77: {:?}", range_replaced.as_slice());

    // 条件付き交換
    let swap_data = F32Tensor::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0], vec![5])?;
    let swapped = swap_data.conditional_swap(|x| x <= 2.0, 1.0, 2.0)?;
    println!(
        "  Swap 1↔2: {:?} → {:?}",
        swap_data.as_slice(),
        swapped.as_slice()
    );

    // 重複除去
    let dup_data = F32Tensor::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 4.0], vec![6])?;
    let (unique, unique_indices) = dup_data.unique_select(true)?;
    println!("  Unique values: {:?}", unique.as_slice());
    if let Some(idx) = unique_indices {
        println!("  Unique indices: {:?}", idx.as_slice());
    }

    // ===== パフォーマンステスト / Performance Test =====
    println!("\n🚀 5. パフォーマンステスト / Performance Test");
    println!("------------------------------------------");

    let large_data: Vec<f32> = (0..1000).map(|i| (i % 100) as f32).collect();
    let large_tensor = F32Tensor::from_vec(large_data, vec![1000])?;
    let large_condition = F32Tensor::from_vec(
        (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect(),
        vec![1000],
    )?;

    let start = Instant::now();
    let _filtered = large_tensor.filter(|x| x > 50.0)?;
    let filter_time = start.elapsed();

    let start = Instant::now();
    let _selected = large_tensor.masked_select(&large_condition)?;
    let mask_time = start.elapsed();

    let start = Instant::now();
    let _argmax = large_tensor.argmax(None)?;
    let argmax_time = start.elapsed();

    let start = Instant::now();
    let (_, _) = large_tensor.histogram(20, None)?;
    let hist_time = start.elapsed();

    println!("  Performance results (size: 1000):");
    println!("    Filter operation: {:?}", filter_time);
    println!("    Masked select: {:?}", mask_time);
    println!("    ArgMax: {:?}", argmax_time);
    println!("    Histogram: {:?}", hist_time);

    println!("\n✅ フェーズ4Bテスト完了！");
    println!("✅ Phase 4B tests completed!");
    println!("\n📊 Phase 4B実装済みメソッド数: 60メソッド（累計: 218メソッド）");
    println!("📊 Phase 4B implemented methods: 60 methods (Total: 218 methods)");
    println!("   - 条件演算: 15メソッド (Conditional operations: 15 methods)");
    println!("     * where_condition, masked_select, masked_fill, masked_scatter");
    println!("     * clamp, clamp_min, clamp_max, clip");
    println!("     * logical_and, logical_or, logical_not, logical_xor");
    println!("     * greater, less, equal");
    println!("   - フィルタリング・マスク: 15メソッド (Filtering & masking: 15 methods)");
    println!("     * filter, nonzero, nonzero_indices, zero_indices");
    println!("     * isnan, isinf, isfinite, isneginf, isposinf");
    println!("     * nan_to_num, replace_nan, drop_nan, fill_nan");
    println!("     * threshold, relu_mask, dropout_mask");
    println!("   - 検索・インデックス: 15メソッド (Search & indexing: 15 methods)");
    println!("     * argmax, argmin, argwhere, searchsorted, bucketize");
    println!("     * histogram, bincount, digitize, find_indices");
    println!("     * first_occurrence, last_occurrence, closest_value");
    println!("     * find_peaks, find_valleys, find_zeros");
    println!("   - 選択・置換: 15メソッド (Selection & replacement: 15 methods)");
    println!("     * take, take_along_axis, where_replace, put, put_along_axis");
    println!("     * select, slice, index_select, conditional_select");
    println!("     * advanced_mask_select, unique_select, replace_range");
    println!("     * pattern_replace, conditional_swap");

    println!("\n🎯 フェーズ4Bの特徴:");
    println!("🎯 Phase 4B Features:");
    println!("   ✓ 完全f32専用条件処理実装（変換コスト0）");
    println!("   ✓ Complete f32-specific conditional processing (zero conversion cost)");
    println!("   ✓ 高性能マスク・フィルタリング操作");
    println!("   ✓ High-performance masking and filtering operations");
    println!("   ✓ 効率的検索・インデックス操作（バイナリサーチ等）");
    println!("   ✓ Efficient search and indexing (binary search, etc.)");
    println!("   ✓ 信号処理向けピーク・谷・ゼロ交差検出");
    println!("   ✓ Signal processing peak, valley, and zero-crossing detection");
    println!("   ✓ 柔軟な選択・置換・パターンマッチング");
    println!("   ✓ Flexible selection, replacement, and pattern matching");
    println!("   ✓ PyTorch互換条件操作API設計");
    println!("   ✓ PyTorch-compatible conditional operations API design");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example hybrid_f32_phase4b_test --features hybrid-f32");
}
