//! ニューラルネットワークモジュールの定義
//! Neural network module definitions.

pub mod linear;
pub mod conv2d;
pub mod pool2d;
pub mod batchnorm;
pub mod activation;
pub mod loss;

use crate::autograd::Variable;
use num_traits::Float;
use std::any::Any;
use std::fmt::Debug;
use std::marker::{Send, Sync};

/// ニューラルネットワークモジュールのトレイト
/// A trait for neural network modules.
pub trait Module<T>: Send + Sync + Debug 
where
    T: Float + 'static + Send + Sync,
{
    /// モジュールの順伝搬を実行します。
    /// Performs the forward pass of the module.
    fn forward(&self, input: &Variable<T>) -> Variable<T>;
    
    /// モジュールのパラメータへの参照を返します。
    /// Returns a reference to the module's parameters.
    fn parameters(&self) -> Vec<Variable<T>>;
    
    /// ダウンキャストのための`&dyn Any`としてのモジュールへの参照を返します。
    /// Returns a reference to the module as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// モジュールを順番に適用するコンテナ
/// A sequential container for modules.
#[derive(Debug, Default)]
pub struct Sequential<T> {
    modules: Vec<Box<dyn Module<T>>>,
}

impl<T> Sequential<T> 
where
    T: Float + 'static + Send + Sync + Debug,
{
    /// 空のシーケンシャルコンテナを作成します。
    /// Creates a new empty sequential container.
    pub fn new() -> Self {
        Sequential { modules: Vec::new() }
    }
    
    /// モジュールをコンテナに追加します。
    /// Adds a module to the container.
    pub fn add_module<M: Module<T> + 'static>(&mut self, module: M) -> &mut Self {
        self.modules.push(Box::new(module));
        self
    }
    
    /// 指定されたインデックスのモジュールへの参照を返します。
    /// Returns a reference to the module at the given index.
    pub fn get_module(&self, index: usize) -> Option<&dyn Module<T>> {
        self.modules.get(index).map(|m| &**m)
    }
    
    /// コンテナ内のモジュールの数を返します。
    /// Returns the number of modules in the container.
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    
    /// コンテナが空の場合は`true`を返します。
    /// Returns `true` if the container is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl<T> Module<T> for Sequential<T> 
where
    T: Float + 'static + Send + Sync + Debug,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x);
        }
        x
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Re-export neural network modules
// ニューラルネットワークモジュールを再エクスポート
/// 線形（全結合）レイヤー
/// Linear (fully connected) layer
pub use linear::Linear;
/// 2次元畳み込みレイヤー
/// 2D convolution layer
pub use conv2d::Conv2d;
/// 2次元プーリングレイヤー
/// 2D pooling layers
pub use pool2d::{MaxPool2d, AvgPool2d};
/// バッチ正規化レイヤー
/// Batch normalization layers
pub use batchnorm::{BatchNorm1d, BatchNorm2d};
