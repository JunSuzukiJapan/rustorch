# RusTorch Testing Guide
# RusTorchテストガイド

Comprehensive guide for running and maintaining tests in RusTorch.

RusTorchのテスト実行と保守のための包括的なガイド。

## Test Categories / テストカテゴリ

### 1. Unit Tests / ユニットテスト

Run all library unit tests:
```bash
cargo test --lib
```

Run specific module tests:
```bash
cargo test tensor::
cargo test nn::
cargo test gpu::
```

### 2. Documentation Tests / ドキュメンテーションテスト

Run all doctests:
```bash
cargo test --doc
```

**Current Status**: ✅ **19 doctests passing** (最終確認: 2025年8月31日)

### 3. Integration Tests / 統合テスト

Run integration tests:
```bash
cargo test --test '*'
```

### 4. Example Tests / サンプルテスト

Test examples compile and run:
```bash
cargo test --examples
```

## Test Coverage / テストカバレッジ

### Current Test Statistics / 現在のテスト統計

| Component | Tests | Status |
|-----------|-------|--------|
| **Tensor Operations** | 173 | ✅ Passing |
| **Neural Networks** | 150+ | ✅ Passing |
| **GPU Operations** | 100+ | ✅ Passing |
| **Vision** | 50+ | ✅ Passing |
| **Doctests** | 19 | ✅ Passing |
| **Total** | 950+ | ✅ 99%+ Pass Rate |

## Running Tests by Feature / 機能別テスト実行

### Default Features / デフォルト機能
```bash
cargo test
```

### WASM Tests / WASMテスト
```bash
cargo test --features wasm
wasm-pack test --node --features wasm
```

### GPU Tests (Individual) / GPU個別テスト
```bash
cargo test --features cuda
cargo test --features metal
cargo test --features opencl
```

### Performance Tests / パフォーマンステスト
```bash
cargo test --release
cargo bench
```

## Common Test Commands / よく使うテストコマンド

### Quick Test Suite / クイックテストスイート
```bash
# Fast unit tests only
cargo test --lib --release

# Documentation tests
cargo test --doc

# Specific module
cargo test tensor::ops::
```

### Comprehensive Testing / 包括的テスト
```bash
# All tests with default features
cargo test

# With specific output
cargo test -- --nocapture

# With specific test threads
RUST_TEST_THREADS=1 cargo test
```

### Continuous Integration / 継続的インテグレーション
```bash
# Check code compiles
cargo check --all-targets

# Run clippy
cargo clippy --all-targets

# Format check
cargo fmt --check

# Full CI suite
cargo fmt --check && cargo clippy && cargo test
```

## Troubleshooting / トラブルシューティング

### Test Timeouts / テストタイムアウト

Some tests may take longer to run. Use timeout:
```bash
timeout 60 cargo test --lib
```

### Memory Issues / メモリ問題

For memory-intensive tests:
```bash
RUST_TEST_THREADS=1 cargo test
```

### GPU Test Issues / GPUテスト問題

GPU tests require hardware availability:
```bash
# Check CUDA availability
nvidia-smi

# Check Metal availability (macOS)
system_profiler SPDisplaysDataType

# Skip GPU tests
cargo test --lib --no-default-features
```

## Writing Tests / テストの書き方

### Unit Test Template / ユニットテストテンプレート
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Arrange
        let input = create_test_data();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_value);
    }
}
```

### Documentation Test Template / ドキュメンテーションテストテンプレート
```rust
/// Example usage of the function
///
/// ```
/// use rustorch::tensor::Tensor;
/// 
/// let tensor = Tensor::zeros(vec![2, 3]);
/// assert_eq!(tensor.shape(), &[2, 3]);
/// ```
pub fn example_function() {
    // Implementation
}
```

### Integration Test Template / 統合テストテンプレート
```rust
// tests/integration_test.rs
use rustorch::prelude::*;

#[test]
fn test_end_to_end_workflow() {
    // Complex multi-component test
    let model = create_model();
    let data = load_data();
    let result = train_model(model, data);
    assert!(result.is_ok());
}
```

## Test Best Practices / テストベストプラクティス

### 1. Test Naming / テスト命名
- Use descriptive names: `test_tensor_addition_with_broadcasting`
- Group related tests in modules
- Prefix with `test_` for clarity

### 2. Test Organization / テスト組織
- Keep tests close to implementation
- Use `#[cfg(test)]` for test modules
- Separate unit and integration tests

### 3. Test Data / テストデータ
- Use small, deterministic data
- Avoid random values without seeds
- Create helper functions for common data

### 4. Assertions / アサーション
- Use appropriate assertion macros
- Include helpful error messages
- Test both success and failure cases

### 5. Performance / パフォーマンス
- Mark slow tests with `#[ignore]`
- Use `--release` for performance tests
- Profile tests when needed

## CI/CD Integration / CI/CD統合

### GitHub Actions Workflow
```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - uses: actions-rs/toolchain@v1
    - run: cargo fmt --check
    - run: cargo clippy -- -D warnings
    - run: cargo test --lib
    - run: cargo test --doc
```

### Pre-commit Hooks
```bash
#!/bin/sh
# .git/hooks/pre-commit

cargo fmt --check
cargo clippy -- -D warnings
cargo test --lib --quiet
```

## Test Maintenance / テストメンテナンス

### Regular Tasks / 定期的なタスク
1. **Weekly**: Run full test suite
2. **Monthly**: Review and update slow tests
3. **Quarterly**: Analyze test coverage
4. **Release**: Comprehensive testing across all features

### Test Health Metrics / テスト健全性メトリクス
- Test execution time
- Flaky test identification
- Coverage percentage
- Test/code ratio

---

**📝 Keep tests fast, reliable, and maintainable**  
**テストを高速、信頼性が高く、保守しやすく保つ**