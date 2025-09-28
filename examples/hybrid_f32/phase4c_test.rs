//! フェーズ4Cユーティリティ・システム操作テスト例
//! Phase 4C Utility & System Operations Test Example

#[cfg(feature = "hybrid-f32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rustorch::hybrid_f32::tensor::F32Tensor;
    use std::time::Instant;

    rustorch::hybrid_f32_experimental!();

    println!("🔧 フェーズ4Cユーティリティ・システム操作テスト");
    println!("🔧 Phase 4C Utility & System Operations Test");
    println!("===========================================\n");

    // ===== メモリ・ストレージ操作デモ / Memory & Storage Operations Demo =====
    println!("💾 1. メモリ・ストレージ操作デモ / Memory & Storage Operations Demo");
    println!("----------------------------------------------------------");

    let data = F32Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])?;
    println!("  Original tensor: {:?}", data.as_slice());

    // クローンとデタッチ
    let cloned = data.clone()?;
    let detached = data.detach()?;
    println!("  Cloned data: {:?}", cloned.as_slice());
    println!("  Detached data: {:?}", detached.as_slice());

    // メモリ操作
    println!("  Memory usage: {} bytes", data.memory_usage());
    println!("  Is contiguous: {}", data.is_contiguous());
    println!("  Stride: {:?}", data.stride());

    // デバイス情報
    println!("  Device info: CPU tensor");
    #[cfg(feature = "cuda")]
    {
        match data.cuda() {
            Ok(cuda_tensor) => println!("  CUDA tensor: {:?}", cuda_tensor.as_slice()),
            Err(_) => println!("  CUDA not available"),
        }
    }

    // ===== 型変換・キャスト操作デモ / Type Conversion & Casting Demo =====
    println!("\n🔄 2. 型変換・キャスト操作デモ / Type Conversion & Casting Demo");
    println!("------------------------------------------------------------");

    let test_data = F32Tensor::from_vec(vec![1.5, 2.7, -1.2, 4.8, 0.0], vec![5])?;
    println!("  Source f32 data: {:?}", test_data.as_slice());

    // 各種型への変換
    let f64_data = test_data.to_f64()?;
    let i32_data = test_data.to_i32()?;
    let u8_data = test_data.to_u8()?;
    let bool_data = test_data.bool()?;

    println!("  To f64: {:?}", f64_data.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    println!("  To i32: {:?}", i32_data);
    println!("  To u8: {:?}", u8_data);
    println!("  To bool: {:?}", bool_data);

    // PyTorch互換変換
    let float_tensor = test_data.float()?;
    let double_data = test_data.double()?;
    println!("  Float tensor: {:?}", float_tensor.as_slice());
    println!("  Double data: {:?}", double_data.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());

    // データ型情報
    println!("  Data type: {}", test_data.dtype());

    // ===== デバッグ・情報取得操作デモ / Debug & Information Operations Demo =====
    println!("\n🔍 3. デバッグ・情報取得操作デモ / Debug & Information Operations Demo");
    println!("-------------------------------------------------------------");

    let debug_data = F32Tensor::from_vec(
        vec![1.0, 2.5, f32::NAN, 4.0, f32::INFINITY, -2.0],
        vec![6]
    )?;
    println!("  Debug tensor: {:?}", debug_data.as_slice());

    // 基本情報
    println!("  Number of elements: {}", debug_data.numel());
    println!("  Number of dimensions: {}", debug_data.ndim());
    println!("  Is empty: {}", debug_data.is_empty());
    println!("  Is scalar: {}", debug_data.is_scalar());
    println!("  Data hash: {:x}", debug_data.data_hash());

    // 状態チェック
    match debug_data.check_state() {
        Ok(state) => println!("  State check: {}", state),
        Err(e) => println!("  State check error: {}", e),
    }

    // 健全性チェック
    match debug_data.sanity_check() {
        Ok(is_sane) => println!("  Sanity check: {}", if is_sane { "PASS" } else { "FAIL" }),
        Err(e) => println!("  Sanity check error: {}", e),
    }

    // 情報表示
    println!("\n  Tensor Info:");
    println!("{}", debug_data.info());

    println!("\n  Summary:");
    println!("{}", debug_data.summary());

    // ===== システム・ハードウェア操作デモ / System & Hardware Operations Demo =====
    println!("\n🖥️  4. システム・ハードウェア操作デモ / System & Hardware Operations Demo");
    println!("----------------------------------------------------------------");

    let perf_data = F32Tensor::from_vec((0..1000).map(|i| i as f32 * 0.1).collect(), vec![1000])?;

    // システム情報
    println!("  System Information:");
    println!("{}", perf_data.system_info());

    println!("\n  Device Information:");
    println!("{}", perf_data.device_info());

    // ハードウェア機能
    println!("\n  Hardware Capabilities:");
    println!("{}", perf_data.hardware_caps());

    // SIMD情報
    println!("\n  SIMD Information:");
    println!("{}", perf_data.simd_info());

    // 並列処理設定
    println!("\n  Parallel Configuration:");
    println!("{}", perf_data.parallel_config());

    // ===== パフォーマンス測定デモ / Performance Measurement Demo =====
    println!("\n⚡ 5. パフォーマンス測定デモ / Performance Measurement Demo");
    println!("-------------------------------------------------------");

    let bench_data = F32Tensor::from_vec((0..5000).map(|i| (i as f32).sin()).collect(), vec![5000])?;

    // CPU使用率
    println!("  CPU Usage:");
    println!("{}", bench_data.cpu_usage());

    // メモリ帯域幅
    println!("\n  Memory Bandwidth:");
    println!("{}", bench_data.memory_bandwidth());

    // 電力効率
    println!("\n  Power Efficiency:");
    println!("{}", bench_data.power_efficiency());

    // 温度監視
    println!("\n  Thermal Status:");
    println!("{}", bench_data.thermal_status());

    // ベンチマーク実行
    println!("\n  Benchmark Results:");
    println!("{}", bench_data.benchmark());

    // ===== 最適化デモ / Optimization Demo =====
    println!("\n🚀 6. 最適化デモ / Optimization Demo");
    println!("--------------------------------");

    let mut opt_data = F32Tensor::from_vec((0..2000).map(|i| i as f32).collect(), vec![2000])?;

    // パフォーマンス最適化
    let start = Instant::now();
    opt_data.optimize_performance()?;
    let opt_time = start.elapsed();
    println!("  Performance optimization completed in: {:?}", opt_time);

    // キャッシュ最適化
    match opt_data.cache_optimize() {
        Ok(cache_info) => {
            println!("\n  Cache Optimization:");
            println!("{}", cache_info);
        }
        Err(e) => println!("  Cache optimization error: {}", e),
    }

    // 最適化提案
    println!("\n  Optimization Recommendations:");
    println!("{}", opt_data.optimization_hints());

    // リソース使用率
    println!("\n  Resource Usage:");
    println!("{}", opt_data.resource_usage());

    // プロファイル情報
    println!("\n  Profile Information:");
    println!("{}", opt_data.profile());

    // ===== パフォーマンステスト / Performance Test =====
    println!("\n🏁 7. パフォーマンステスト / Performance Test");
    println!("-------------------------------------------");

    let large_data = F32Tensor::from_vec((0..10000).map(|i| (i as f32) * 0.001).collect(), vec![10000])?;

    let start = Instant::now();
    let _info = large_data.info();
    let info_time = start.elapsed();

    let start = Instant::now();
    let _state = large_data.check_state()?;
    let state_time = start.elapsed();

    let start = Instant::now();
    let _converted = large_data.to_f64()?;
    let convert_time = start.elapsed();

    let start = Instant::now();
    let _optimized = large_data.optimization_hints();
    let hints_time = start.elapsed();

    println!("  Performance results (size: 10000):");
    println!("    Info generation: {:?}", info_time);
    println!("    State check: {:?}", state_time);
    println!("    Type conversion: {:?}", convert_time);
    println!("    Optimization hints: {:?}", hints_time);

    println!("\n✅ フェーズ4Cテスト完了！");
    println!("✅ Phase 4C tests completed!");
    println!("\n📊 フェーズ4C実装済みメソッド数: 60メソッド（累計: 278メソッド）");
    println!("📊 Phase 4C implemented methods: 60 methods (Total: 278 methods)");
    println!("   - メモリ・ストレージ操作: 15メソッド (Memory & storage operations: 15 methods)");
    println!("     * clone, copy_, detach, share_memory_, is_shared");
    println!("     * storage, storage_offset, stride, contiguous, is_contiguous");
    println!("     * pin_memory, cpu, cuda, to_device, memory_format");
    println!("   - 型変換・キャスト操作: 15メソッド (Type conversion & casting: 15 methods)");
    println!("     * to_f64, to_f32, to_i64, to_i32, to_u8, half");
    println!("     * float, double, long, int, bool, byte, char");
    println!("     * type_as, dtype");
    println!("   - デバッグ・情報取得操作: 15メソッド (Debug & information operations: 15 methods)");
    println!("     * info, check_state, memory_usage, numel, ndim");
    println!("     * is_empty, is_scalar, data_hash, debug_info, perf_stats");
    println!("     * summary, sanity_check, trace_info, backtrace, profile");
    println!("   - システム・ハードウェア操作: 15メソッド (System & hardware operations: 15 methods)");
    println!("     * system_info, device_info, optimize_performance, cpu_usage, memory_bandwidth");
    println!("     * parallel_config, cache_optimize, simd_info, power_efficiency, thermal_status");
    println!("     * resource_usage, hardware_caps, optimization_hints, benchmark");

    println!("\n🎯 フェーズ4Cの特徴:");
    println!("🎯 Phase 4C Features:");
    println!("   ✓ 完全f32専用ユーティリティ実装（変換コスト0）");
    println!("   ✓ Complete f32-specific utility implementation (zero conversion cost)");
    println!("   ✓ 包括的メモリ管理・デバイス制御");
    println!("   ✓ Comprehensive memory management and device control");
    println!("   ✓ 高精度型変換・キャスト操作");
    println!("   ✓ High-precision type conversion and casting operations");
    println!("   ✓ 詳細デバッグ・プロファイリング情報");
    println!("   ✓ Detailed debugging and profiling information");
    println!("   ✓ システム最適化・パフォーマンス監視");
    println!("   ✓ System optimization and performance monitoring");
    println!("   ✓ ハードウェア特性検出・活用");
    println!("   ✓ Hardware capability detection and utilization");
    println!("   ✓ PyTorch互換システムユーティリティAPI設計");
    println!("   ✓ PyTorch-compatible system utility API design");

    println!("\n🏆 Phase 4 全体完了！ (4A: 60 + 4B: 60 + 4C: 60 = 180メソッド)");
    println!("🏆 Phase 4 Complete! (4A: 60 + 4B: 60 + 4C: 60 = 180 methods)");
    println!("総実装メソッド数: 278メソッド");
    println!("Total implemented methods: 278 methods");

    Ok(())
}

#[cfg(not(feature = "hybrid-f32"))]
fn main() {
    println!("❌ hybrid-f32 フィーチャーが必要です。");
    println!("❌ hybrid-f32 feature required.");
    println!("実行方法: cargo run --example hybrid_f32_phase4c_test --features hybrid-f32");
}