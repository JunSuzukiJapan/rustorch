//! Common functionality for recurrent neural networks
//! リカレントニューラルネットワーク用共通機能

use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;
use rand_distr::{Distribution, Normal};
use std::fmt::Debug;

/// Common configuration for recurrent cells
/// リカレントセル用共通設定
#[derive(Debug, Clone)]
pub struct RecurrentConfig {
    /// Input size
    /// 入力サイズ
    pub input_size: usize,

    /// Hidden size
    /// 隠れ状態サイズ
    pub hidden_size: usize,

    /// Number of gates (RNN: 1, GRU: 3, LSTM: 4)
    /// ゲート数（RNN: 1, GRU: 3, LSTM: 4）
    pub num_gates: usize,

    /// Whether to use bias
    /// バイアスを使用するか
    pub bias: bool,

    /// Training mode
    /// 学習モード
    pub training: bool,
}

impl RecurrentConfig {
    /// Create new RNN configuration
    /// 新しいRNN設定を作成
    pub fn rnn(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        Self {
            input_size,
            hidden_size,
            num_gates: 1,
            bias,
            training: true,
        }
    }

    /// Create new GRU configuration
    /// 新しいGRU設定を作成
    pub fn gru(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        Self {
            input_size,
            hidden_size,
            num_gates: 3,
            bias,
            training: true,
        }
    }

    /// Create new LSTM configuration
    /// 新しいLSTM設定を作成
    pub fn lstm(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        Self {
            input_size,
            hidden_size,
            num_gates: 4,
            bias,
            training: true,
        }
    }
}

/// Common trait for recurrent cells
/// リカレントセル用共通トレイト
pub trait RecurrentCell<T: Float + Send + Sync + Debug + 'static> {
    /// Get input size
    /// 入力サイズを取得
    fn input_size(&self) -> usize;

    /// Get hidden size
    /// 隠れ状態サイズを取得
    fn hidden_size(&self) -> usize;

    /// Set training mode
    /// 学習モードを設定
    fn set_training(&mut self, training: bool);

    /// Check if in training mode
    /// 学習モードかどうかをチェック
    fn is_training(&self) -> bool;

    /// Get configuration
    /// 設定を取得
    fn config(&self) -> &RecurrentConfig;
}

/// Common operations for recurrent cells
/// リカレントセル用共通操作
pub struct RecurrentOps;

impl RecurrentOps {
    /// Initialize weights using Xavier/Glorot initialization
    /// Xavier/Glorot初期化で重みを初期化
    pub fn init_weights<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        input_size: usize,
        hidden_size: usize,
        num_gates: usize,
    ) -> (Variable<T>, Variable<T>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Input-to-hidden weights
        let weight_ih_data: Vec<T> = (0..num_gates * hidden_size * input_size)
            .map(|_| num_traits::cast(normal.sample(&mut rng) as f64).unwrap_or(T::zero()))
            .collect();
        let weight_ih = Variable::new(
            Tensor::from_vec(weight_ih_data, vec![num_gates * hidden_size, input_size]),
            true,
        );

        // Hidden-to-hidden weights
        let weight_hh_data: Vec<T> = (0..num_gates * hidden_size * hidden_size)
            .map(|_| num_traits::cast(normal.sample(&mut rng) as f64).unwrap_or(T::zero()))
            .collect();
        let weight_hh = Variable::new(
            Tensor::from_vec(weight_hh_data, vec![num_gates * hidden_size, hidden_size]),
            true,
        );

        (weight_ih, weight_hh)
    }

    /// Initialize bias
    /// バイアスを初期化
    pub fn init_bias<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        hidden_size: usize,
        num_gates: usize,
    ) -> (Option<Variable<T>>, Option<Variable<T>>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let bias_ih_data: Vec<T> = (0..num_gates * hidden_size)
            .map(|_| num_traits::cast(normal.sample(&mut rng) as f64).unwrap_or(T::zero()))
            .collect();
        let bias_ih = Some(Variable::new(
            Tensor::from_vec(bias_ih_data, vec![num_gates * hidden_size]),
            true,
        ));

        let bias_hh_data: Vec<T> = (0..num_gates * hidden_size)
            .map(|_| num_traits::cast(normal.sample(&mut rng) as f64).unwrap_or(T::zero()))
            .collect();
        let bias_hh = Some(Variable::new(
            Tensor::from_vec(bias_hh_data, vec![num_gates * hidden_size]),
            true,
        ));

        (bias_ih, bias_hh)
    }

    /// Linear transformation: input @ weight^T + bias
    /// 線形変換: input @ weight^T + bias
    pub fn linear_transform<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        input: &Variable<T>,
        weight: &Variable<T>,
        bias: Option<&Variable<T>>,
    ) -> Variable<T> {
        let output = Self::matmul_variables(input, &Self::transpose_variable(weight));

        match bias {
            Some(b) => Self::add_variables(&output, b),
            None => output,
        }
    }

    /// Matrix multiplication for variables
    /// Variable用の行列乗算
    pub fn matmul_variables<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        a: &Variable<T>,
        b: &Variable<T>,
    ) -> Variable<T> {
        // Use Variable's matmul method directly
        a.matmul(b)
    }

    /// Addition for variables
    /// Variable用の加算
    pub fn add_variables<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        a: &Variable<T>,
        b: &Variable<T>,
    ) -> Variable<T> {
        // Use Variable's add operator directly
        a + b
    }

    /// Multiplication for variables
    /// Variable用の乗算
    pub fn multiply_variables<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        a: &Variable<T>,
        b: &Variable<T>,
    ) -> Variable<T> {
        // Use Variable's multiplication operator directly
        a * b
    }

    /// Subtract variable from scalar
    /// スカラーから変数を減算
    pub fn subtract_from_scalar<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        var: &Variable<T>,
        scalar: T,
    ) -> Variable<T> {
        let var_binding = var.data();
        let var_data = var_binding.read().unwrap();
        let result_data = var_data.map(|x| scalar - x);
        Variable::new(result_data, var.requires_grad())
    }

    /// Transpose for variables
    /// Variable用の転置
    pub fn transpose_variable<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        var: &Variable<T>,
    ) -> Variable<T> {
        let var_binding = var.data();
        let var_data = var_binding.read().unwrap();
        let transposed_data = var_data.transpose().unwrap();
        Variable::new(transposed_data, var.requires_grad())
    }

    /// Sigmoid activation for variables
    /// Variable用のシグモイド活性化
    pub fn sigmoid<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        var: &Variable<T>,
    ) -> Variable<T> {
        let var_binding = var.data();
        let var_data = var_binding.read().unwrap();
        let sigmoid_data = var_data.map(|x| T::one() / (T::one() + (-x).exp()));
        Variable::new(sigmoid_data, var.requires_grad())
    }

    /// Tanh activation for variables
    /// Variable用のtanh活性化
    pub fn tanh<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        var: &Variable<T>,
    ) -> Variable<T> {
        let var_binding = var.data();
        let var_data = var_binding.read().unwrap();
        let tanh_data = var_data.map(|x| x.tanh());
        Variable::new(tanh_data, var.requires_grad())
    }

    /// Slice gates from concatenated tensor
    /// 連結されたテンソルからゲートをスライス
    pub fn slice_gates<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        gates: &Variable<T>,
        gate_idx: usize,
        hidden_size: usize,
    ) -> Variable<T> {
        let start_idx = gate_idx * hidden_size;
        let end_idx = (gate_idx + 1) * hidden_size;

        // Simplified slicing - in practice would need proper tensor slicing
        let gates_binding = gates.data();
        let gates_data = gates_binding.read().unwrap();
        let gate_data: Vec<T> = gates_data.as_slice().unwrap()[start_idx..end_idx].to_vec();
        Variable::new(
            Tensor::from_vec(gate_data, vec![gates_data.shape()[0], hidden_size]),
            gates.requires_grad(),
        )
    }

    /// Create zero hidden state
    /// ゼロ隠れ状態を作成
    pub fn zero_hidden_state<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        batch_size: usize,
        hidden_size: usize,
    ) -> Variable<T> {
        Variable::new(Tensor::zeros(&[batch_size, hidden_size]), false)
    }
}

/// Training mode enumeration
/// 学習モード列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMode {
    /// Training mode
    /// 学習モード
    Train,
    /// Evaluation mode
    /// 評価モード
    Eval,
}

impl From<bool> for TrainingMode {
    fn from(training: bool) -> Self {
        if training {
            TrainingMode::Train
        } else {
            TrainingMode::Eval
        }
    }
}

impl From<TrainingMode> for bool {
    fn from(mode: TrainingMode) -> Self {
        matches!(mode, TrainingMode::Train)
    }
}

/// Common parameter collection for recurrent cells
/// リカレントセル用共通パラメータ収集
pub fn collect_recurrent_parameters<
    T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
>(
    weight_ih: &Variable<T>,
    weight_hh: &Variable<T>,
    bias_ih: &Option<Variable<T>>,
    bias_hh: &Option<Variable<T>>,
) -> Vec<Variable<T>> {
    let mut params = vec![weight_ih.clone(), weight_hh.clone()];

    if let Some(ref bias) = bias_ih {
        params.push(bias.clone());
    }

    if let Some(ref bias) = bias_hh {
        params.push(bias.clone());
    }

    params
}

/// Common forward pass utilities for multi-layer recurrent networks
/// 多層リカレントネットワーク用共通順伝播ユーティリティ
pub struct MultiLayerUtils;

impl MultiLayerUtils {
    /// Get input for a specific timestep
    /// 特定のタイムステップの入力を取得
    pub fn get_timestep_input<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        input: &Variable<T>,
        timestep: usize,
    ) -> Variable<T> {
        // Simplified implementation - would need proper tensor slicing
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let batch_size = input_data.shape()[0];
        let feature_size = input_data.shape()[2];

        // Extract data for this timestep
        let timestep_data: Vec<T> = (0..batch_size * feature_size)
            .map(|i| {
                let batch_idx = i / feature_size;
                let feat_idx = i % feature_size;
                input_data.as_slice().unwrap()[batch_idx * input_data.shape()[1] * feature_size
                    + timestep * feature_size
                    + feat_idx]
            })
            .collect();

        Variable::new(
            Tensor::from_vec(timestep_data, vec![batch_size, feature_size]),
            input.requires_grad(),
        )
    }

    /// Stack outputs along sequence dimension
    /// シーケンス次元に沿って出力をスタック
    pub fn stack_outputs<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        outputs: &[Variable<T>],
    ) -> Variable<T> {
        let output_binding = outputs[0].data();
        let output_data = output_binding.read().unwrap();
        let batch_size = output_data.shape()[0];
        let hidden_size = output_data.shape()[1];
        let seq_len = outputs.len();

        let mut stacked_data = Vec::new();

        for batch_idx in 0..batch_size {
            for t in 0..seq_len {
                let output_binding = outputs[t].data();
                let output_data = output_binding.read().unwrap();
                let output_slice = output_data.as_slice().unwrap();
                let start_idx = batch_idx * hidden_size;
                let end_idx = start_idx + hidden_size;
                stacked_data.extend_from_slice(&output_slice[start_idx..end_idx]);
            }
        }

        Variable::new(
            Tensor::from_vec(stacked_data, vec![batch_size, seq_len, hidden_size]),
            outputs[0].requires_grad(),
        )
    }

    /// Stack hidden states by layer
    /// レイヤーごとに隠れ状態をスタック
    pub fn stack_hidden_states<
        T: Float + Send + Sync + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    >(
        states: &[Variable<T>],
        num_layers: usize,
    ) -> Variable<T> {
        let state_binding = states[0].data();
        let state_data = state_binding.read().unwrap();
        let batch_size = state_data.shape()[0];
        let hidden_size = state_data.shape()[1];

        let mut stacked_data = Vec::new();

        for state in states {
            let state_binding = state.data();
            let state_data = state_binding.read().unwrap();
            stacked_data.extend_from_slice(state_data.as_slice().unwrap());
        }

        Variable::new(
            Tensor::from_vec(stacked_data, vec![num_layers, batch_size, hidden_size]),
            states[0].requires_grad(),
        )
    }
}
