# PyTorch-RusTorch Compatibility Limitations
# PyTorch-RusTorch互換性制限事項

## Critical Incompatibilities / 重大な非互換性

### 1. Dynamic vs Static Typing / 動的型付け vs 静的型付け

**PyTorch (Python):**
```python
# 実行時に型が決定される
tensor = torch.randn(3, 3)  # 型推論
if some_condition:
    tensor = tensor.float()  # 実行時型変更
else:
    tensor = tensor.double()
```

**RusTorch (Rust):**
```rust
// コンパイル時に型が固定される
let tensor = Tensor::<f32>::randn(&[3, 3]);  // 明示的型指定必須
// tensor = tensor.to_f64();  // ❌ 異なる型への変更は不可
```

**影響:** RusTorchでは型変更時に新しいテンソルを作成する必要があり、PyTorchのような柔軟な型操作ができない。

### 2. Memory Management Model / メモリ管理モデル

**PyTorch:**
```python
# ガベージコレクション
tensor1 = torch.randn(1000, 1000)
tensor2 = tensor1.view(-1)  # ビュー共有
del tensor1  # 参照カウントで管理
```

**RusTorch:**
```rust
// 所有権システム
let tensor1 = Tensor::<f32>::randn(&[1000, 1000]);
let tensor2 = tensor1.reshape(&[1000000]);  // tensor1は移動される
// println!("{:?}", tensor1);  // ❌ 使用後アクセス不可
```

**影響:** Rustの所有権により、PyTorchのような複数参照やビューの共有が制限される。

### 3. Gradient Computation API / 勾配計算API

**PyTorch:**
```python
# シンプルな勾配計算
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(4.0)
```

**RusTorch:**
```rust
// より複雑な勾配アクセス
let x = Variable::new(Tensor::from_vec(vec![2.0], vec![1]), true);
let y = &x * &x;
y.backward();

// 複雑な勾配アクセス
let grad_binding = x.grad();
let grad_guard = grad_binding.read().unwrap();
if let Some(ref grad_tensor) = *grad_guard {
    println!("{:?}", grad_tensor.as_array()[0]);
}
```

**影響:** 勾配アクセスがPyTorchより冗長で、borrow checkerのため複雑になる。

## API Design Differences / API設計の違い

### 4. Method Chaining / メソッドチェーン

**PyTorch:**
```python
# 流暢なメソッドチェーン
result = tensor.relu().dropout(0.5).normalize().softmax(dim=1)
```

**RusTorch:**
```rust
// 所有権のため制限されたチェーン
let tensor = tensor_var.clone();
let relu_result = relu.forward(&tensor);
let dropout_result = dropout.forward(&relu_result);
// チェーンが制限される
```

### 5. In-place Operations / インプレース操作

**PyTorch:**
```python
# インプレース操作が簡単
tensor.add_(1.0)  # tensor自体を変更
tensor.relu_()    # tensor自体を変更
```

**RusTorch:**
```rust
// インプレース操作が制限される
// tensor.add_inplace(1.0);  // ❌ 所有権の問題で困難
let tensor = &tensor + &Tensor::from_scalar(1.0);  // 新しいテンソル作成
```

### 6. Dynamic Neural Networks / 動的ニューラルネットワーク

**PyTorch:**
```python
class DynamicNet(nn.Module):
    def forward(self, x):
        if x.size(1) > 100:
            return self.large_path(x)
        else:
            return self.small_path(x)  # 実行時分岐
```

**RusTorch:**
```rust
// 静的な構造のみ
impl Module<f32> for MyNet {
    fn forward(&self, x: &Variable<f32>) -> Variable<f32> {
        // 動的分岐はコンパイル時に決定される必要がある
        self.layer.forward(x)
    }
}
```

## Missing Features / 未実装機能

### 7. Advanced Tensor Operations / 高度なテンソル操作

**PyTorch で利用可能だがRusTorchで未実装:**
```python
# PyTorchにあるがRusTorchにない操作
tensor.scatter_(dim=1, index=indices, src=values)
tensor.gather(dim=1, index=indices)
torch.einsum('ij,jk->ik', a, b)
torch.fft.fft(tensor)
torch.linalg.svd(tensor)
```

### 8. Complex Data Types / 複素数データ型

**PyTorch:**
```python
# 完全な複素数サポート
complex_tensor = torch.complex(real, imag)
result = torch.fft.fft(complex_tensor)
```

**RusTorch:**
```rust
// 複素数型は定義されているが操作が限定的
// Complex64, Complex128は基本的な定義のみ
```

### 9. GPU Memory Management / GPUメモリ管理

**PyTorch:**
```python
# 簡単なGPU操作
tensor = tensor.cuda()
torch.cuda.empty_cache()
torch.cuda.memory_summary()
```

**RusTorch:**
```rust
// GPU操作はより低レベル
// let gpu_tensor = tensor.to_device(device);  // 実装はモック段階
```

## Performance Limitations / パフォーマンス制限

### 10. Compilation Overhead / コンパイルオーバーヘッド

**PyTorch:**
- インタープリター実行で即座に実行開始
- JIT コンパイルで最適化

**RusTorch:**
- 全てコンパイル時に型チェック
- 大きなプロジェクトでコンパイル時間増加
- 型変更時の再コンパイル必要

### 11. Memory Usage Patterns / メモリ使用パターン

**PyTorch:**
```python
# 参照共有で効率的
a = torch.randn(1000, 1000)
b = a[::2, ::2]  # ビューで共有、追加メモリ不要
```

**RusTorch:**
```rust
// 所有権により時にコピーが必要
let a = Tensor::<f32>::randn(&[1000, 1000]);
// スライス操作が制限される場合がある
```

## Ecosystem Differences / エコシステムの違い

### 12. Library Ecosystem / ライブラリエコシステム

**PyTorch:**
- 豊富なサードパーティライブラリ
- Hugging Face Transformers
- torchvision, torchaudio
- 多数の事前学習済みモデル

**RusTorch:**
- 限定的なエコシステム
- サードパーティ統合が少ない
- 事前学習済みモデルの選択肢が限定

### 13. Research Workflow / 研究ワークフロー

**PyTorch:**
```python
# Jupyter Notebookでの対話的開発
tensor = torch.randn(3, 3)
# セルごとに実行、即座に結果確認
```

**RusTorch:**
```rust
// コンパイル-実行サイクル
// 対話的開発が困難
// プロトタイピングに時間がかかる
```

## Model Import/Export Limitations / モデルインポート・エクスポート制限

### 14. Model Format Support / モデル形式サポート

**制限事項:**
- ONNX読み込みは基本実装のみ（フル機能ではない）
- PyTorch state_dict の完全互換性なし
- 一部のオペレーターが未サポート
- 動的グラフのシリアライゼーション不可

### 15. Distributed Training / 分散学習

**PyTorch:**
```python
# 簡単な分散学習
torch.distributed.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

**RusTorch:**
```rust
// 分散学習は基本実装のみ
// NCCLバインディングが限定的
// 完全な分散学習サポートなし
```

## Workaround Solutions / 回避策

### 16. 互換性問題への対処法

1. **型変換問題:**
```rust
// 明示的な型変換関数を作成
fn convert_tensor_type<T, U>(tensor: &Tensor<T>) -> Tensor<U>
where T: Float, U: Float {
    // 型変換実装
}
```

2. **所有権問題:**
```rust
// Rc/Arc を使用した共有
use std::sync::Arc;
let shared_tensor = Arc::new(tensor);
```

3. **勾配アクセス簡略化:**
```rust
// ヘルパー関数の作成
fn get_gradient(var: &Variable<f32>) -> Option<f32> {
    let grad_data = var.grad();
    let grad_guard = grad_data.read().unwrap();
    grad_guard.as_ref().map(|g| g.as_array()[0])
}
```

## Migration Strategy / 移行戦略

### 17. 段階的移行アプローチ

1. **評価フェーズ:** PyTorchでプロトタイプ開発
2. **実装フェーズ:** 確定した設計をRusTorchで実装
3. **最適化フェーズ:** Rustの利点を活用した最適化

### 18. 適用場面の選択

**RusTorchが適している:**
- プロダクション環境での推論
- システム統合が必要な場合
- メモリ安全性が重要
- WebAssembly展開

**PyTorchが適している:**
- 研究開発・プロトタイピング
- 複雑な動的モデル
- 豊富なライブラリが必要
- 対話的開発

## Conclusion / 結論

RusTorchとPyTorchの非互換性は主に**言語レベルの設計思想の違い**に起因します：

- **PyTorch:** 柔軟性、動的性、研究指向
- **RusTorch:** 安全性、静的性、プロダクション指向

これらの制限を理解して適切に使い分けることで、両方の利点を最大化できます。

**互換性スコア詳細:**
- **コア機能:** 95% ✅
- **API設計:** 70% ⚠️
- **エコシステム:** 30% ❌
- **研究ワークフロー:** 40% ❌
- **プロダクション:** 90% ✅