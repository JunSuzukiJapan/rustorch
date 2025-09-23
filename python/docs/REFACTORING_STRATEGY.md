# リファクタリング戦略 - 段階的アプローチ
# Refactoring Strategy - Incremental Approach

## 問題分析 / Problem Analysis

現在のモジュラー構造での問題:
1. マクロ定義の欠如 (invalid_param!, nn_error!, optim_error!)
2. API の不一致 (Variable.grad() の戻り値型)
3. モジュール間の循環依存
4. PyO3 マクロの適用方法の違い

## 段階的リファクタリング計画 / Incremental Refactoring Plan

### フェーズ1: 現在のlib.rsをクリーンアップ
- [x] Phase 4実装完了と動作確認
- [x] バグ修正完了確認
- [ ] lib.rsのコード整理（コメント、構造）
- [ ] 重複コード除去

### フェーズ2: インラインモジュール化
- [ ] 同じファイル内でmodを使って論理分割
- [ ] テスト実行して動作確認
- [ ] 段階的にコードを移動

### フェーズ3: ファイル分割
- [ ] 一つずつモジュールをファイルに分割
- [ ] 各段階でテスト実行
- [ ] マクロ定義を各モジュールに配置

### フェーズ4: 最終統合
- [ ] 全体テスト
- [ ] パフォーマンス確認
- [ ] ドキュメント更新

## 現在の成果 / Current Achievements

✅ **Phase 4 CNN実装完了**
- Conv2d, MaxPool2d, BatchNorm1d/2d, Dropout, Flatten
- 全てのニューラルネットワーク層が動作

✅ **重要なバグ修正完了**
- Flatten層のtensor API修正
- SGDパラメータ不足修正
- BatchNorm1d表示問題修正
- Tensor作成エラーハンドリング修正

✅ **モジュール設計完了**
- core/ (tensor, variable, errors)
- nn/layers/ (linear, conv, norm, dropout, flatten)
- nn/ (activation, loss)
- optim/ (sgd, adam)

## 次のステップ / Next Steps

1. 現在のlib.rsで段階的なコード整理
2. インラインモジュール化によるテスト
3. 段階的ファイル分割