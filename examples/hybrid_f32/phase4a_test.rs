//! フェーズ4A高度統計操作テスト例
//! Phase 4A Advanced Statistical Operations Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::F32Tensor;
    use std::time::Instant;

    rustorch::hybrid_f32_experimental!();

    println!("📊 フェーズ4A高度統計操作テスト");
    println!("📊 Phase 4A Advanced Statistical Operations Test");
    println!("============================================\n");

    // ===== 分位数・順序統計デモ / Quantile & Order Statistics Demo =====
    println!("🎯 1. 分位数・順序統計デモ / Quantile & Order Statistics Demo");
    println!("--------------------------------------------------------");

    let sample_data = F32Tensor::from_vec(
        vec![
            1.2, 3.5, 2.1, 5.8, 4.2, 7.9, 6.1, 8.5, 9.2, 0.5, 3.8, 4.7, 6.3, 7.1, 2.9,
        ],
        vec![15],
    )?;
    println!("  Sample data: {:?}", sample_data.as_slice());

    let q25 = sample_data.quantile(0.25)?;
    let median = sample_data.median()?;
    let q75 = sample_data.quantile(0.75)?;
    let p90 = sample_data.percentile(90.0)?;

    println!("  25th percentile (Q1): {:.3}", q25);
    println!("  Median (Q2): {:.3}", median);
    println!("  75th percentile (Q3): {:.3}", q75);
    println!("  90th percentile: {:.3}", p90);

    let iqr = sample_data.iqr()?;
    let range = sample_data.range()?;
    println!("  IQR (Q3-Q1): {:.3}", iqr);
    println!("  Range (max-min): {:.3}", range);

    let (unique_vals, counts) = sample_data.unique_counts()?;
    println!("  Unique values: {:?}", unique_vals.as_slice());
    println!("  Counts: {:?}", counts);

    let (top3_values, top3_indices) = sample_data.topk(3)?;
    println!("  Top 3 values: {:?}", top3_values.as_slice());
    println!("  Top 3 indices: {:?}", top3_indices.as_slice());

    // ===== 累積統計デモ / Cumulative Statistics Demo =====
    println!("\n🔢 2. 累積統計デモ / Cumulative Statistics Demo");
    println!("-------------------------------------------");

    let time_series = F32Tensor::from_vec(
        vec![10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0, 14.0, 7.0, 16.0],
        vec![10],
    )?;
    println!("  Time series: {:?}", time_series.as_slice());

    let cumsum = time_series.cumsum()?;
    let cumprod = time_series.cumprod()?;
    let cummax = time_series.cummax()?;
    let cummin = time_series.cummin()?;

    println!("  Cumulative sum: {:?}", cumsum.as_slice());
    println!(
        "  Cumulative product: {:?}",
        cumprod.as_slice().iter().take(5).collect::<Vec<_>>()
    );
    println!("  Cumulative max: {:?}", cummax.as_slice());
    println!("  Cumulative min: {:?}", cummin.as_slice());

    let diff = time_series.diff()?;
    println!("  First differences: {:?}", diff.as_slice());

    let moving_avg = time_series.moving_average(3)?;
    let moving_std = time_series.moving_std(3)?;
    println!("  Moving average (window=3): {:?}", moving_avg.as_slice());
    println!(
        "  Moving std (window=3): {:?}",
        moving_std
            .as_slice()
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // ===== 相関・共分散デモ / Correlation & Covariance Demo =====
    println!("\n🔗 3. 相関・共分散デモ / Correlation & Covariance Demo");
    println!("-----------------------------------------------");

    let x_data = F32Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![10],
    )?;
    let y_data = F32Tensor::from_vec(
        vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.8, 16.2, 18.1, 20.0],
        vec![10],
    )?;
    println!("  X data: {:?}", x_data.as_slice());
    println!("  Y data: {:?}", y_data.as_slice());

    let correlation = x_data.corrcoef(&y_data)?;
    let covariance = x_data.cov(&y_data)?;
    println!("  Correlation coefficient: {:.6}", correlation);
    println!("  Covariance: {:.3}", covariance);

    let cross_corr = x_data.cross_correlation(&y_data, 0)?;
    let auto_corr = x_data.autocorrelation(2)?;
    println!("  Cross-correlation: {:.6}", cross_corr);
    println!("  Autocorrelation (lag=2): {:.6}", auto_corr);

    // スピアマン相関（ランク相関）
    let spearman_corr = x_data.spearman_correlation(&y_data)?;
    println!("  Spearman correlation: {:.6}", spearman_corr);

    // ===== 高度な分布統計デモ / Advanced Distribution Statistics Demo =====
    println!("\n📈 4. 高度な分布統計デモ / Advanced Distribution Statistics Demo");
    println!("------------------------------------------------------------");

    let normal_like = F32Tensor::from_vec(
        vec![
            2.1, 2.5, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0,
        ],
        vec![15],
    )?;
    println!("  Normal-like data: {:?}", normal_like.as_slice());

    let skewness = normal_like.skewness()?;
    let kurtosis = normal_like.kurtosis()?;
    let jarque_bera = normal_like.jarque_bera()?;
    println!("  Skewness: {:.6}", skewness);
    println!("  Kurtosis: {:.6}", kurtosis);
    println!("  Jarque-Bera statistic: {:.6}", jarque_bera);

    let cv = normal_like.coefficient_of_variation()?;
    let mad = normal_like.mad()?;
    let mean_abs_dev = normal_like.mean_absolute_deviation()?;
    println!("  Coefficient of variation: {:.6}", cv);
    println!("  Median absolute deviation: {:.6}", mad);
    println!("  Mean absolute deviation: {:.6}", mean_abs_dev);

    let entropy = normal_like.entropy()?;
    let gini = normal_like.gini_coefficient()?;
    println!("  Entropy: {:.6}", entropy);
    println!("  Gini coefficient: {:.6}", gini);

    // 異常値検出
    let outliers = normal_like.outliers_iqr(1.5)?;
    println!("  Outliers (IQR method): {:?}", outliers);

    // Z-score標準化
    let zscore = normal_like.zscore()?;
    println!(
        "  Z-scores: {:?}",
        zscore
            .as_slice()
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // ===== 正規性検定デモ / Normality Testing Demo =====
    println!("\n🔬 5. 正規性検定デモ / Normality Testing Demo");
    println!("------------------------------------------");

    let shapiro_stat = normal_like.shapiro_wilk()?;
    let anderson_stat = normal_like.anderson_darling()?;
    let ks_stat = normal_like.kolmogorov_smirnov()?;

    println!("  Shapiro-Wilk statistic: {:.6}", shapiro_stat);
    println!("  Anderson-Darling statistic: {:.6}", anderson_stat);
    println!("  Kolmogorov-Smirnov statistic: {:.6}", ks_stat);

    // 高次モーメント
    let second_moment = normal_like.moment(2)?;
    let third_moment = normal_like.moment(3)?;
    let fourth_moment = normal_like.fourth_moment()?;
    println!("  2nd moment: {:.6}", second_moment);
    println!("  3rd moment: {:.6}", third_moment);
    println!("  4th moment: {:.6}", fourth_moment);

    // ===== 統計的比較デモ / Statistical Comparison Demo =====
    println!("\n⚖️  6. 統計的比較デモ / Statistical Comparison Demo");
    println!("---------------------------------------------");

    let dataset_a = F32Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![5])?;
    let dataset_b = F32Tensor::from_vec(vec![2.8, 4.2, 5.1, 5.9, 7.2], vec![5])?;

    let mse = dataset_a.mse(&dataset_b)?;
    println!("  Mean squared error between datasets: {:.6}", mse);

    // 各データセットの統計
    println!("\n  Dataset A statistics:");
    println!("    Mean: {:.3}", dataset_a.mean()?);
    println!("    Std: {:.3}", dataset_a.std()?);
    println!("    Skewness: {:.6}", dataset_a.skewness()?);
    println!("    Kurtosis: {:.6}", dataset_a.kurtosis()?);

    println!("  Dataset B statistics:");
    println!("    Mean: {:.3}", dataset_b.mean()?);
    println!("    Std: {:.3}", dataset_b.std()?);
    println!("    Skewness: {:.6}", dataset_b.skewness()?);
    println!("    Kurtosis: {:.6}", dataset_b.kurtosis()?);

    // ===== パフォーマンステスト / Performance Test =====
    println!("\n🚀 7. パフォーマンステスト / Performance Test");
    println!("------------------------------------------");

    let large_data: Vec<f32> = (0..1000)
        .map(|i| (i as f32) * 0.01 + (i % 10) as f32)
        .collect();
    let large_tensor = F32Tensor::from_vec(large_data, vec![1000])?;

    let start = Instant::now();
    let _quantile = large_tensor.quantile(0.95)?;
    let quantile_time = start.elapsed();

    let start = Instant::now();
    let _correlation_matrix = large_tensor.corrcoef(&large_tensor)?;
    let correlation_time = start.elapsed();

    let start = Instant::now();
    let _skewness = large_tensor.skewness()?;
    let skewness_time = start.elapsed();

    let start = Instant::now();
    let _zscore = large_tensor.zscore()?;
    let zscore_time = start.elapsed();

    println!("  Performance results (size: 1000):");
    println!("    95th percentile: {:?}", quantile_time);
    println!("    Correlation: {:?}", correlation_time);
    println!("    Skewness: {:?}", skewness_time);
    println!("    Z-score: {:?}", zscore_time);

    println!("\n✅ フェーズ4Aテスト完了！");
    println!("✅ Phase 4A tests completed!");
    println!("\n📊 フェーズ4A実装済みメソッド数: 60メソッド（累計: 158メソッド）");
    println!("📊 Phase 4A implemented methods: 60 methods (Total: 158 methods)");
    println!("   - 分位数・順序統計: 15メソッド (Quantile & order statistics: 15 methods)");
    println!("     * quantile, median, percentile, quartiles, mode, unique, unique_counts");
    println!("     * topk, kthvalue, argsort, sort, msort, nanquantile, nanmedian, nanpercentile");
    println!("   - 累積統計: 15メソッド (Cumulative statistics: 15 methods)");
    println!("     * cumsum, cumprod, cummax, cummin, diff, gradient, moving_average");
    println!("     * moving_std, rolling_mean, rolling_std, rolling_max, rolling_min");
    println!("     * exponential_moving_average, weighted_average, running_statistics");
    println!("   - 相関・共分散: 12メソッド (Correlation & covariance: 12 methods)");
    println!("     * corrcoef, cov, cross_correlation, autocorrelation, partial_correlation");
    println!("     * spearman_correlation, kendall_tau, mutual_information, covariance_matrix");
    println!("     * correlation_matrix, distance_correlation, canonical_correlation");
    println!("   - 高度分布統計: 18メソッド (Advanced distribution statistics: 18 methods)");
    println!("     * skewness, kurtosis, jarque_bera, coefficient_of_variation, range, iqr");
    println!("     * mad, mean_absolute_deviation, entropy, gini_coefficient, outliers_iqr");
    println!("     * zscore, fourth_moment, moment, shapiro_wilk, anderson_darling");
    println!("     * kolmogorov_smirnov, mse");

    println!("\n🎯 フェーズ4Aの特徴:");
    println!("🎯 Phase 4A Features:");
    println!("   ✓ 完全f32専用高度統計実装（変換コスト0）");
    println!(
        "   ✓ Complete f32-specific advanced statistics implementation (zero conversion cost)"
    );
    println!("   ✓ 分位数・順序統計による分布解析");
    println!("   ✓ Distribution analysis with quantiles and order statistics");
    println!("   ✓ 時系列データ向け累積統計・移動窓統計");
    println!("   ✓ Time series cumulative and moving window statistics");
    println!("   ✓ 多変量相関解析（ピアソン・スピアマン・ケンドール）");
    println!("   ✓ Multivariate correlation analysis (Pearson, Spearman, Kendall)");
    println!("   ✓ 正規性検定・分布適合度検定");
    println!("   ✓ Normality and goodness-of-fit testing");
    println!("   ✓ 異常値検出・データ品質評価");
    println!("   ✓ Outlier detection and data quality assessment");
    println!("   ✓ PyTorch互換統計API設計");
    println!("   ✓ PyTorch-compatible statistical API design");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example hybrid_f32_phase4a_test --features hybrid-f32");
}
