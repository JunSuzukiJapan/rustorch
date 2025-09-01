use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustorch::tensor::Tensor;

fn create_optimized_benchmark_config() -> Criterion {
    Criterion::default()
        .sample_size(10) // Criterionの最小値
        .measurement_time(std::time::Duration::from_secs(2)) // 短い測定時間
        .warm_up_time(std::time::Duration::from_secs(1)) // 短いウォームアップ
        .significance_level(0.1) // より緩い有意水準
        .noise_threshold(0.05) // ノイズ閾値を緩く
}

fn bench_matrix_decomposition_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fast Matrix Decomposition");

    // 設定を最適化（Criterionの最小値を守る）
    group.sample_size(10); // 最小値
    group.measurement_time(std::time::Duration::from_secs(1));
    group.warm_up_time(std::time::Duration::from_millis(500));

    // 小さな行列サイズのみでテスト
    let sizes = vec![4, 8, 12]; // 16以上は除外

    for size in sizes {
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 + 1.0) % 8.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        // SVDのみベンチマーク（最も重要）
        group.bench_with_input(
            BenchmarkId::new("SVD", size),
            &matrix,
            |b, m| b.iter(|| black_box(m.svd())), // eigenvectors無しで高速化
        );
    }

    group.finish();
}

fn bench_comparison_lightweight(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decomposition Comparison Light");

    // 非常に軽い設定
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));

    let size = 8; // 小さなサイズ固定

    // テスト行列作成
    let matrix_data: Vec<f32> = (0..size * size)
        .map(|i| (i as f32 + 1.0) % 6.0 + 1.0)
        .collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    // 各分解を個別にベンチマーク
    group.bench_function("SVD_8x8", |b| b.iter(|| black_box(&matrix).svd()));

    group.bench_function("QR_8x8", |b| b.iter(|| black_box(&matrix).qr()));

    group.bench_function("QR_8x8_alt", |b| b.iter(|| black_box(&matrix).qr()));

    // 対称行列を作成
    let mut sym_data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            if i == j {
                sym_data[i * size + j] = (i + 1) as f32;
            } else if i < j {
                sym_data[i * size + j] = 0.5;
                sym_data[j * size + i] = 0.5;
            }
        }
    }
    let sym_matrix = Tensor::from_vec(sym_data, vec![size, size]);

    group.bench_function("Symeig_8x8", |b| {
        b.iter(|| black_box(&sym_matrix).eigh()) // eigenvectors無し
    });

    group.bench_function("Eig_8x8", |b| {
        b.iter(|| black_box(&matrix).eigh()) // eigenvectors無し
    });

    group.finish();
}

fn bench_rectangular_matrices_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rectangular Matrices Fast");

    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));

    // 小さな長方行列のみ
    let test_cases = vec![(6, 3, "6x3"), (8, 4, "8x4"), (10, 5, "10x5")];

    for (rows, cols, label) in test_cases {
        let matrix_data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 + 1.0) % 7.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![rows, cols]);

        // SVDとQRのみ（LUは正方行列向け）
        group.bench_with_input(BenchmarkId::new("SVD", label), &matrix, |b, m| {
            b.iter(|| black_box(m.svd()))
        });

        group.bench_with_input(BenchmarkId::new("QR", label), &matrix, |b, m| {
            b.iter(|| black_box(m.qr()))
        });
    }

    group.finish();
}

fn bench_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling Analysis");

    group.sample_size(10); // 最小サンプル
    group.measurement_time(std::time::Duration::from_millis(500));

    // サイズ vs 時間の関係を測定
    let sizes = vec![4, 6, 8, 10]; // 小刻みでスケーリング確認

    for size in sizes {
        let matrix_data: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 1.3 + 2.1) % 5.0 + 1.0)
            .collect();
        let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

        // 最も使用頻度の高いSVDのみでスケーリング測定
        group.bench_with_input(BenchmarkId::new("SVD_scaling", size), &matrix, |b, m| {
            b.iter(|| black_box(m.svd()))
        });
    }

    group.finish();
}

// タイムアウト回避のためのカスタムベンチマーク関数
fn bench_with_timeout_protection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Timeout Protected Benchmark");

    // 非常に保守的な設定
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_millis(800));
    group.warm_up_time(std::time::Duration::from_millis(200));

    let size = 6; // 非常に小さなサイズ

    let matrix_data: Vec<f32> = (0..size * size).map(|i| (i + 1) as f32).collect();
    let matrix = Tensor::from_vec(matrix_data, vec![size, size]);

    // 1つずつ安全にテスト
    group.bench_function("safe_svd", |b| {
        b.iter(|| {
            // タイムアウト回避のため結果をすぐにdrop
            let result = black_box(&matrix).svd();
            drop(result);
        })
    });

    group.bench_function("safe_qr", |b| {
        b.iter(|| {
            let result = black_box(&matrix).qr();
            drop(result);
        })
    });

    group.finish();
}

// Criterionの設定をグローバルに適用
criterion_group! {
    name = benches;
    config = create_optimized_benchmark_config();
    targets =
        bench_matrix_decomposition_fast,
        bench_comparison_lightweight,
        bench_rectangular_matrices_fast,
        bench_scaling_analysis,
        bench_with_timeout_protection
}

criterion_main!(benches);
