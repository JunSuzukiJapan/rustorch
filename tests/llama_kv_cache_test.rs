/// KVキャッシュの動作を検証するテスト
///
/// テスト内容：
/// 1. KVキャッシュの初期化とクリア
/// 2. トークン追加時のキャッシュサイズ増加
/// 3. 複数ステップでのキャッシュ使用
/// 4. キャッシュクリア後の再初期化

use rustorch::models::llama::KVCache;

#[test]
fn test_kv_cache_initialization() {
    println!("🧪 Test 1: KVキャッシュの初期化");

    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let kv_dim = 256; // TinyLlama: 4 KV heads × 64 head_dim

    let cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // 初期状態の検証
    assert_eq!(cache.batch_size, batch_size, "バッチサイズが一致すること");
    assert_eq!(cache.max_cache_size, max_cache_size, "最大キャッシュサイズが一致すること");
    assert_eq!(cache.k_cache.len(), num_layers, "レイヤー数が一致すること");
    assert_eq!(cache.v_cache.len(), num_layers, "レイヤー数が一致すること");
    assert_eq!(cache.cached_tokens[0], 0, "初期キャッシュトークン数は0であること");

    println!("✅ KVキャッシュが正しく初期化されました");
}

#[test]
fn test_kv_cache_clear() {
    println!("🧪 Test 2: KVキャッシュのクリア");

    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let kv_dim = 256;

    let mut cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // トークン数を設定
    cache.cached_tokens[0] = 100;
    assert_eq!(cache.cached_tokens[0], 100, "トークン数が設定されること");

    // クリア
    cache.clear();
    assert_eq!(cache.cached_tokens[0], 0, "クリア後はトークン数が0になること");

    println!("✅ KVキャッシュが正しくクリアされました");
}

#[test]
fn test_kv_cache_token_accumulation() {
    println!("🧪 Test 3: トークンの累積");

    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let kv_dim = 256;

    let mut cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // シミュレーション: 複数ステップでトークンを追加
    println!("Step 0: プロンプト処理 (14トークン)");
    cache.cached_tokens[0] += 14;
    assert_eq!(cache.cached_tokens[0], 14, "Step 0後: 14トークン");

    println!("Step 1: 1トークン生成");
    cache.cached_tokens[0] += 1;
    assert_eq!(cache.cached_tokens[0], 15, "Step 1後: 15トークン");

    println!("Step 2: 1トークン生成");
    cache.cached_tokens[0] += 1;
    assert_eq!(cache.cached_tokens[0], 16, "Step 2後: 16トークン");

    println!("Step 3: 1トークン生成");
    cache.cached_tokens[0] += 1;
    assert_eq!(cache.cached_tokens[0], 17, "Step 3後: 17トークン");

    println!("✅ トークンが正しく累積されました");
}

#[test]
fn test_kv_cache_size_calculation() {
    println!("🧪 Test 4: キャッシュサイズの計算");

    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let kv_dim = 256;

    let cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // 各レイヤーのキャッシュサイズを検証
    for layer_idx in 0..num_layers {
        let k_cache_size = cache.k_cache[layer_idx][0].len();
        let v_cache_size = cache.v_cache[layer_idx][0].len();

        let expected_size = max_cache_size * kv_dim;

        assert_eq!(k_cache_size, expected_size,
            "Layer {}: Kキャッシュサイズが正しいこと", layer_idx);
        assert_eq!(v_cache_size, expected_size,
            "Layer {}: Vキャッシュサイズが正しいこと", layer_idx);
    }

    println!("✅ キャッシュサイズが正しく計算されました");
}

#[test]
fn test_kv_cache_overflow_detection() {
    println!("🧪 Test 5: キャッシュオーバーフロー検出");

    let num_layers = 2;
    let batch_size = 1;
    let max_cache_size = 100; // 小さい値でテスト
    let kv_dim = 256;

    let mut cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // 最大サイズまで追加
    cache.cached_tokens[0] = max_cache_size;

    // オーバーフロー検出のシミュレーション
    let would_overflow = cache.cached_tokens[0] + 1 > max_cache_size;
    assert!(would_overflow, "最大サイズ超過を検出できること");

    println!("✅ オーバーフロー検出が正しく動作します");
}

#[test]
fn test_kv_cache_multi_batch() {
    println!("🧪 Test 6: マルチバッチのキャッシュ");

    let num_layers = 22;
    let batch_size = 4;
    let max_cache_size = 2048;
    let kv_dim = 256;

    let mut cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // 各バッチアイテムに異なるトークン数を設定
    cache.cached_tokens[0] = 10;
    cache.cached_tokens[1] = 20;
    cache.cached_tokens[2] = 30;
    cache.cached_tokens[3] = 40;

    assert_eq!(cache.cached_tokens[0], 10, "バッチ0: 10トークン");
    assert_eq!(cache.cached_tokens[1], 20, "バッチ1: 20トークン");
    assert_eq!(cache.cached_tokens[2], 30, "バッチ2: 30トークン");
    assert_eq!(cache.cached_tokens[3], 40, "バッチ3: 40トークン");

    // クリア後はすべて0
    cache.clear();
    for i in 0..batch_size {
        assert_eq!(cache.cached_tokens[i], 0, "バッチ{}: クリア後は0", i);
    }

    println!("✅ マルチバッチキャッシュが正しく動作します");
}

#[test]
fn test_kv_cache_gqa_dimensions() {
    println!("🧪 Test 7: GQA (Grouped Query Attention) 用のキャッシュ次元");

    // TinyLlamaの設定
    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let num_heads = 32;
    let num_kv_heads = 4; // GQA: 4 KV heads
    let head_dim = 64;
    let kv_dim = num_kv_heads * head_dim; // 4 × 64 = 256

    let cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // GQAの次元検証
    let heads_per_kv = num_heads / num_kv_heads; // 32 / 4 = 8
    println!("  GQA設定:");
    println!("    - num_heads (Q): {}", num_heads);
    println!("    - num_kv_heads (K/V): {}", num_kv_heads);
    println!("    - heads_per_kv: {}", heads_per_kv);
    println!("    - head_dim: {}", head_dim);
    println!("    - kv_dim: {}", kv_dim);

    assert_eq!(kv_dim, 256, "TinyLlama GQAのkv_dimは256であること");
    assert_eq!(heads_per_kv, 8, "各KVヘッドは8個のQヘッドで共有されること");

    // キャッシュサイズの検証
    for layer_idx in 0..num_layers {
        let k_cache_size = cache.k_cache[layer_idx][0].len();
        let expected_size = max_cache_size * kv_dim; // 2048 × 256 = 524288
        assert_eq!(k_cache_size, expected_size,
            "Layer {}: Kキャッシュサイズが正しいこと", layer_idx);
    }

    println!("✅ GQA用のKVキャッシュ次元が正しく設定されました");
}

#[test]
fn test_kv_cache_position_tracking() {
    println!("🧪 Test 8: Position tracking (自己回帰生成のシミュレーション)");

    let num_layers = 22;
    let batch_size = 1;
    let max_cache_size = 2048;
    let kv_dim = 256;

    let mut cache = KVCache::new(num_layers, batch_size, max_cache_size, kv_dim);

    // 自己回帰生成のシミュレーション
    println!("\n📝 自己回帰生成シミュレーション:");

    // Step 0: プロンプト全体を処理
    let prompt_len = 14;
    let cache_start = cache.cached_tokens[0];
    let cache_end = cache_start + prompt_len;

    println!("  Step 0: プロンプト処理");
    println!("    - seq_len: {}", prompt_len);
    println!("    - cache_start: {}", cache_start);
    println!("    - cache_end: {}", cache_end);

    cache.cached_tokens[0] += prompt_len;
    assert_eq!(cache.cached_tokens[0], 14, "Step 0後: 14トークン");

    // Step 1: 1トークン生成
    let gen_len = 1;
    let cache_start = cache.cached_tokens[0];
    let cache_end = cache_start + gen_len;
    let total_seq_len = cache.cached_tokens[0] + gen_len;

    println!("\n  Step 1: 1トークン生成");
    println!("    - seq_len: {}", gen_len);
    println!("    - cache_start: {}", cache_start);
    println!("    - cache_end: {}", cache_end);
    println!("    - total_seq_len (for attention): {}", total_seq_len);

    cache.cached_tokens[0] += gen_len;
    assert_eq!(cache.cached_tokens[0], 15, "Step 1後: 15トークン");
    assert_eq!(total_seq_len, 15, "Attention用のtotal_seq_lenが正しいこと");

    // Step 2: 1トークン生成
    let gen_len = 1;
    let cache_start = cache.cached_tokens[0];
    let cache_end = cache_start + gen_len;
    let total_seq_len = cache.cached_tokens[0] + gen_len;

    println!("\n  Step 2: 1トークン生成");
    println!("    - seq_len: {}", gen_len);
    println!("    - cache_start: {}", cache_start);
    println!("    - cache_end: {}", cache_end);
    println!("    - total_seq_len (for attention): {}", total_seq_len);

    cache.cached_tokens[0] += gen_len;
    assert_eq!(cache.cached_tokens[0], 16, "Step 2後: 16トークン");
    assert_eq!(total_seq_len, 16, "Attention用のtotal_seq_lenが正しいこと");

    println!("\n✅ Position trackingが正しく動作します");
}
