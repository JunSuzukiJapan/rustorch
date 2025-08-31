/* tslint:disable */
/* eslint-disable */
/**
 * Initialize WASM module
 */
export function init_wasm(): void;
/**
 * Create tensor from Float32Array
 */
export function tensor_from_float32_array(data: Float32Array, shape: Array<any>): WasmTensor;
/**
 * Convert tensor to Float32Array
 */
export function tensor_to_float32_array(tensor: WasmTensor): Float32Array;
/**
 * Create tensor from nested JavaScript array
 */
export function tensor_from_nested_array(array: any): WasmTensor;
/**
 * Convert tensor to nested JavaScript array
 */
export function tensor_to_nested_array(tensor: WasmTensor): Array<any>;
/**
 * Memory-efficient tensor slicing
 */
export function tensor_slice(tensor: WasmTensor, start: number, end: number): WasmTensor;
/**
 * Simple benchmark for tensor operations
 */
export function benchmark_matmul(size: number, iterations: number): BenchmarkResult;
/**
 * Initialize WASM runtime
 */
export function initialize_wasm_runtime(): void;
/**
 * Detect WASM runtime features
 */
export function detect_wasm_features(): object;
/**
 * Version information
 */
export function wasm_advanced_math_version(): string;
/**
 * Version information
 */
export function wasm_anomaly_detection_version(): string;
/**
 * Create a simple anomaly detector for web applications
 */
export function create_simple_detector(threshold: number): WasmAnomalyDetector;
/**
 * Create a time series detector for streaming data
 */
export function create_streaming_detector(window_size: number): WasmTimeSeriesDetector;
/**
 * Batch anomaly detection for arrays
 */
export function detect_anomalies_batch(data: Float32Array, threshold: number): Array<any>;
/**
 * Version information
 */
export function wasm_transforms_version(): string;
/**
 * Create ImageNet preprocessing pipeline
 */
export function create_imagenet_preprocessing(): WasmNormalize;
/**
 * Create CIFAR preprocessing pipeline
 */
export function create_cifar_preprocessing(): WasmNormalize;
/**
 * Version information
 */
export function wasm_quality_metrics_version(): string;
/**
 * Create quality analyzer with default threshold
 */
export function create_quality_analyzer(): WasmQualityMetrics;
/**
 * Quick quality assessment for web applications
 */
export function quick_quality_assessment(tensor: WasmTensor): string;
export function gamma_wasm(x: number): number;
export function lgamma_wasm(x: number): number;
export function digamma_wasm(x: number): number;
export function beta_wasm(a: number, b: number): number;
export function lbeta_wasm(a: number, b: number): number;
export function bessel_j_wasm(n: number, x: number): number;
export function bessel_y_wasm(n: number, x: number): number;
export function bessel_i_wasm(n: number, x: number): number;
export function bessel_k_wasm(n: number, x: number): number;
export function erf_wasm(x: number): number;
export function erfc_wasm(x: number): number;
export function erfinv_wasm(x: number): number;
export function erfcinv_wasm(x: number): number;
export function gamma_array_wasm(values: Float64Array): Float64Array;
export function bessel_j_array_wasm(n: number, values: Float64Array): Float64Array;
export function erf_array_wasm(values: Float64Array): Float64Array;
export function factorial_wasm(n: number): number;
export function log_factorial_wasm(n: number): number;
export function euler_gamma(): number;
export function sqrt_2pi(): number;
export function log_sqrt_2pi(): number;
export function normal_cdf_wasm(x: number, mean: number, std: number): number;
export function normal_quantile_wasm(p: number, mean: number, std: number): number;
export function quick_stats_wasm(values: Float64Array): Float64Array;
export function benchmark_special_functions_wasm(iterations: number): Float64Array;
export function learning_rate_schedule_wasm(initial_lr: number, step: bigint, decay_rate: number, decay_steps: bigint): number;
export function cosine_annealing_wasm(initial_lr: number, current_step: bigint, total_steps: bigint): number;
export function relu_wasm(x: number): number;
export function relu_array_wasm(values: Float64Array): Float64Array;
export function sigmoid_wasm(x: number): number;
export function sigmoid_array_wasm(values: Float64Array): Float64Array;
export function tanh_wasm(x: number): number;
export function tanh_array_wasm(values: Float64Array): Float64Array;
export function softmax_wasm(values: Float64Array): Float64Array;
export function get_browser_webgpu_info(): string;
export function calculate_performance_estimate(operation: string, size: number): number;
export class AdaGradWasm {
  free(): void;
  constructor(learning_rate: number, epsilon: number, weight_decay: number);
  step(param_name: string, params: Float64Array, gradients: Float64Array): void;
  get_learning_rate(): number;
  set_learning_rate(lr: number): void;
  reset_state(): void;
}
export class AdamWasm {
  free(): void;
  constructor(learning_rate: number, beta1: number, beta2: number, epsilon: number, weight_decay: number);
  step(param_name: string, params: Float64Array, gradients: Float64Array): void;
  get_learning_rate(): number;
  set_learning_rate(lr: number): void;
  get_step_count(): bigint;
  reset_state(): void;
}
/**
 * Performance benchmarking utility
 */
export class BenchmarkResult {
  private constructor();
  free(): void;
  /**
   * Get operation name
   */
  readonly operation: string;
  /**
   * Get duration in milliseconds
   */
  readonly duration_ms: number;
  /**
   * Get throughput (operations per second)
   */
  readonly throughput: number;
}
export class BernoulliDistributionWasm {
  free(): void;
  constructor(p: number);
  sample(): boolean;
  sample_array(n: number): Uint8Array;
  sample_f64(): number;
  sample_f64_array(n: number): Float64Array;
  log_prob(x: boolean): number;
  mean(): number;
  variance(): number;
}
export class BetaDistributionWasm {
  free(): void;
  constructor(alpha: number, beta: number);
  sample(): number;
  sample_array(n: number): Float64Array;
  log_prob(x: number): number;
  mean(): number;
  variance(): number;
}
/**
 * Browser storage utilities
 */
export class BrowserStorage {
  free(): void;
  /**
   * Create new browser storage utility
   */
  constructor();
  /**
   * Save tensor to localStorage
   */
  save_tensor(key: string, tensor: WasmTensor): void;
  /**
   * Load tensor from localStorage
   */
  load_tensor(key: string): WasmTensor;
  /**
   * List all saved tensor keys
   */
  list_tensor_keys(): Array<any>;
  /**
   * Clear all saved tensors
   */
  clear_tensors(): void;
}
/**
 * Canvas utilities for tensor visualization
 */
export class CanvasRenderer {
  free(): void;
  /**
   * Create new canvas renderer for the specified canvas element
   */
  constructor(canvas_id: string);
  /**
   * Render 2D tensor as heatmap
   */
  render_heatmap(tensor: WasmTensor): void;
  /**
   * Clear canvas
   */
  clear(): void;
}
export class ComputationGraphWasm {
  free(): void;
  constructor();
  create_variable(data: Float64Array, shape: Uint32Array, requires_grad: boolean): string;
  get_variable_data(id: string): Float64Array | undefined;
  get_variable_grad(id: string): Float64Array | undefined;
  add_variables(id1: string, id2: string): string | undefined;
  mul_variables(id1: string, id2: string): string | undefined;
  backward(id: string): void;
  zero_grad_all(): void;
  clear_graph(): void;
  variable_count(): number;
}
export class ExponentialDistributionWasm {
  free(): void;
  constructor(rate: number);
  sample(): number;
  sample_array(n: number): Float64Array;
  log_prob(x: number): number;
  mean(): number;
  variance(): number;
}
/**
 * File API utilities for loading tensors from files
 */
export class FileLoader {
  free(): void;
  /**
   * Create new file loader utility
   */
  constructor();
  /**
   * Create file input element for tensor loading
   */
  create_file_input(): HTMLInputElement;
}
export class GammaDistributionWasm {
  free(): void;
  constructor(shape: number, scale: number);
  sample(): number;
  sample_array(n: number): Float64Array;
  log_prob(x: number): number;
  mean(): number;
  variance(): number;
}
/**
 * JavaScript interop utilities
 */
export class JsInterop {
  free(): void;
  /**
   * Create new JavaScript interop utility
   */
  constructor();
  /**
   * Create tensor filled with ones
   */
  ones(shape: Array<any>): WasmTensor;
  /**
   * Create tensor filled with zeros
   */
  zeros(shape: Array<any>): WasmTensor;
  /**
   * Create random tensor
   */
  random_tensor(shape: Array<any>, min: number, max: number): WasmTensor;
  /**
   * Log tensor information to console
   */
  log_tensor(tensor: WasmTensor, name: string): void;
}
export class LinearLayerWasm {
  free(): void;
  constructor(input_size: number, output_size: number);
  forward(input: Float64Array): Float64Array | undefined;
  get_weights(): Float64Array;
  get_bias(): Float64Array;
  update_weights(new_weights: Float64Array): boolean;
  update_bias(new_bias: Float64Array): boolean;
}
export class NormalDistributionWasm {
  free(): void;
  constructor(mean: number, std: number);
  sample(): number;
  sample_array(n: number): Float64Array;
  log_prob(x: number): number;
  log_prob_array(values: Float64Array): Float64Array;
  mean(): number;
  variance(): number;
  std_dev(): number;
}
/**
 * Optimized tensor operations for WASM
 */
export class OptimizedOps {
  free(): void;
  /**
   * Create new optimized operations utility
   */
  constructor();
  /**
   * Fast matrix multiplication using blocking for cache efficiency
   */
  fast_matmul(a: WasmTensor, b: WasmTensor): WasmTensor;
  /**
   * Vectorized element-wise operations
   */
  vectorized_add(a: WasmTensor, b: WasmTensor): WasmTensor;
  /**
   * Fast ReLU with fused operations
   */
  fused_relu_add(input: WasmTensor, bias: WasmTensor): WasmTensor;
  /**
   * Memory-efficient convolution-like operation (simplified 1D)
   */
  conv1d(input: WasmTensor, kernel: WasmTensor, stride: number): WasmTensor;
  /**
   * Batch normalization-like operation
   */
  batch_normalize(input: WasmTensor, epsilon: number): WasmTensor;
}
/**
 * Parallel execution utilities (simulated for WASM single-threaded environment)
 */
export class ParallelOps {
  private constructor();
  free(): void;
  /**
   * Parallel-style reduction (sequential in WASM)
   */
  static parallel_sum(data: Float32Array): number;
  /**
   * Parallel-style element-wise operation
   */
  static parallel_map_add(a: Float32Array, b: Float32Array): Float32Array;
}
/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
  private constructor();
  free(): void;
  /**
   * Get memory usage information
   */
  static get_memory_info(): object;
  /**
   * Measure function execution time
   */
  static time_function(name: string): void;
  /**
   * End timing measurement
   */
  static time_end(name: string): void;
}
export class RMSpropWasm {
  free(): void;
  constructor(learning_rate: number, alpha: number, epsilon: number, weight_decay: number, momentum: number);
  step(param_name: string, params: Float64Array, gradients: Float64Array): void;
  get_learning_rate(): number;
  set_learning_rate(lr: number): void;
  reset_state(): void;
}
export class SGDWasm {
  free(): void;
  constructor(learning_rate: number, momentum: number, dampening: number, weight_decay: number, nesterov: boolean);
  step(param_name: string, params: Float64Array, gradients: Float64Array): void;
  get_learning_rate(): number;
  set_learning_rate(lr: number): void;
  reset_state(): void;
}
export class SpecialFunctionsBatch {
  free(): void;
  constructor(cache_size: number);
  gamma_batch(values: Float64Array): Float64Array;
  bessel_j0_batch(values: Float64Array): Float64Array;
  erf_batch(values: Float64Array): Float64Array;
}
export class UniformDistributionWasm {
  free(): void;
  constructor(low: number, high: number);
  sample(): number;
  sample_array(n: number): Float64Array;
  log_prob(x: number): number;
  mean(): number;
  variance(): number;
}
export class VariableWasm {
  free(): void;
  constructor(data: Float64Array, shape: Uint32Array, requires_grad: boolean);
  data(): Float64Array;
  shape(): Uint32Array;
  grad(): Float64Array | undefined;
  requires_grad(): boolean;
  zero_grad(): void;
  backward(): void;
  sum(): VariableWasm;
  mean(): VariableWasm;
  pow(exponent: number): VariableWasm;
}
/**
 * WASM-compatible activation functions
 * WASM互換活性化関数
 */
export class WasmActivation {
  private constructor();
  free(): void;
  /**
   * ReLU (Rectified Linear Unit) activation function
   * ReLU(x) = max(0, x)
   */
  static relu(input: Float32Array): Float32Array;
  /**
   * ReLU derivative for backward pass
   * ReLUの微分（逆伝播用）
   */
  static relu_derivative(input: Float32Array): Float32Array;
  /**
   * Leaky ReLU activation function
   * Leaky ReLU(x) = max(alpha * x, x)
   */
  static leaky_relu(input: Float32Array, alpha: number): Float32Array;
  /**
   * Leaky ReLU derivative
   */
  static leaky_relu_derivative(input: Float32Array, alpha: number): Float32Array;
  /**
   * Sigmoid activation function
   * Sigmoid(x) = 1 / (1 + exp(-x))
   */
  static sigmoid(input: Float32Array): Float32Array;
  /**
   * Sigmoid derivative
   * σ'(x) = σ(x) * (1 - σ(x))
   */
  static sigmoid_derivative(input: Float32Array): Float32Array;
  /**
   * Tanh (Hyperbolic Tangent) activation function
   * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
   */
  static tanh(input: Float32Array): Float32Array;
  /**
   * Tanh derivative
   * tanh'(x) = 1 - tanh²(x)
   */
  static tanh_derivative(input: Float32Array): Float32Array;
  /**
   * Softmax activation function
   * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
   */
  static softmax(input: Float32Array): Float32Array;
  /**
   * Log Softmax activation function (numerically stable)
   * LogSoftmax(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
   */
  static log_softmax(input: Float32Array): Float32Array;
  /**
   * GELU (Gaussian Error Linear Unit) activation function
   * GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
   * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
   */
  static gelu(input: Float32Array): Float32Array;
  /**
   * GELU derivative (approximate)
   */
  static gelu_derivative(input: Float32Array): Float32Array;
  /**
   * Swish/SiLU activation function
   * Swish(x) = x * sigmoid(x)
   */
  static swish(input: Float32Array): Float32Array;
  /**
   * Mish activation function
   * Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
   */
  static mish(input: Float32Array): Float32Array;
  /**
   * ELU (Exponential Linear Unit) activation function
   * ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
   */
  static elu(input: Float32Array, alpha: number): Float32Array;
  /**
   * ELU derivative
   */
  static elu_derivative(input: Float32Array, alpha: number): Float32Array;
  /**
   * Softplus activation function
   * Softplus(x) = ln(1 + exp(x))
   */
  static softplus(input: Float32Array): Float32Array;
  /**
   * Softsign activation function
   * Softsign(x) = x / (1 + |x|)
   */
  static softsign(input: Float32Array): Float32Array;
  /**
   * Apply activation function to 2D data (batch processing)
   * 2Dデータに活性化関数を適用（バッチ処理）
   */
  static relu_2d(input: Float32Array, rows: number, cols: number): Float32Array;
  /**
   * Apply softmax along specified axis for 2D data
   * 2Dデータの指定軸に沿ってソフトマックスを適用
   */
  static softmax_2d(input: Float32Array, rows: number, cols: number, axis: number): Float32Array;
  /**
   * Combined activation function selector
   * 活性化関数セレクター
   */
  static apply_activation(input: Float32Array, activation_type: string): Float32Array;
}
/**
 * AdaGrad optimizer for WASM (simpler than Adam)
 * WASM用AdaGradオプティマイザ（Adamより簡単）
 */
export class WasmAdaGrad {
  free(): void;
  /**
   * Create new AdaGrad optimizer
   */
  constructor(learning_rate: number, epsilon: number);
  /**
   * Update parameters with AdaGrad algorithm
   */
  step(param_id: string, parameters: Float32Array, gradients: Float32Array): Float32Array;
}
/**
 * Adam optimizer for WASM
 * WASM用Adamオプティマイザ
 */
export class WasmAdam {
  free(): void;
  /**
   * Create new Adam optimizer
   */
  constructor(learning_rate: number);
  /**
   * Create Adam with custom parameters
   */
  static with_params(learning_rate: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdam;
  /**
   * Update parameters with gradients using Adam algorithm
   */
  step(param_id: string, parameters: Float32Array, gradients: Float32Array): Float32Array;
  /**
   * Get learning rate
   */
  get_learning_rate(): number;
  /**
   * Set learning rate
   */
  set_learning_rate(lr: number): void;
  /**
   * Get step count
   */
  get_step_count(): number;
  /**
   * Reset optimizer state
   */
  reset(): void;
}
/**
 * WASM wrapper for advanced mathematical operations
 */
export class WasmAdvancedMath {
  free(): void;
  /**
   * Create new advanced math instance
   */
  constructor();
  /**
   * Hyperbolic sine
   */
  sinh(tensor: WasmTensor): WasmTensor;
  /**
   * Hyperbolic cosine
   */
  cosh(tensor: WasmTensor): WasmTensor;
  /**
   * Hyperbolic tangent
   */
  tanh(tensor: WasmTensor): WasmTensor;
  /**
   * Inverse sine (arcsine)
   */
  asin(tensor: WasmTensor): WasmTensor;
  /**
   * Inverse cosine (arccosine)
   */
  acos(tensor: WasmTensor): WasmTensor;
  /**
   * Inverse tangent (arctangent)
   */
  atan(tensor: WasmTensor): WasmTensor;
  /**
   * Two-argument arctangent
   */
  atan2(y: WasmTensor, x: WasmTensor): WasmTensor;
  /**
   * Error function (approximate)
   */
  erf(tensor: WasmTensor): WasmTensor;
  /**
   * Complementary error function
   */
  erfc(tensor: WasmTensor): WasmTensor;
  /**
   * Gamma function (approximate)
   */
  gamma(tensor: WasmTensor): WasmTensor;
  /**
   * Log gamma function
   */
  lgamma(tensor: WasmTensor): WasmTensor;
  /**
   * Clamp values between min and max
   */
  clamp(tensor: WasmTensor, min_val: number, max_val: number): WasmTensor;
  /**
   * Sign function
   */
  sign(tensor: WasmTensor): WasmTensor;
  /**
   * Linear interpolation between two tensors
   */
  lerp(start: WasmTensor, end: WasmTensor, weight: number): WasmTensor;
  /**
   * Power function with scalar exponent
   */
  pow(base: WasmTensor, exponent: number): WasmTensor;
  /**
   * Element-wise power
   */
  pow_tensor(base: WasmTensor, exponent: WasmTensor): WasmTensor;
  /**
   * Round to nearest integer
   */
  round(tensor: WasmTensor): WasmTensor;
  /**
   * Floor function
   */
  floor(tensor: WasmTensor): WasmTensor;
  /**
   * Ceiling function
   */
  ceil(tensor: WasmTensor): WasmTensor;
  /**
   * Truncate to integer
   */
  trunc(tensor: WasmTensor): WasmTensor;
  /**
   * Check if values are finite
   */
  is_finite(tensor: WasmTensor): WasmTensor;
  /**
   * Check if values are infinite
   */
  is_infinite(tensor: WasmTensor): WasmTensor;
  /**
   * Check if values are NaN
   */
  is_nan(tensor: WasmTensor): WasmTensor;
}
/**
 * WASM wrapper for Anomaly Detector
 */
export class WasmAnomalyDetector {
  free(): void;
  /**
   * Create new anomaly detector
   */
  constructor(threshold: number, window_size: number);
  /**
   * Detect anomalies using statistical method
   */
  detect_statistical(data: WasmTensor): Array<any>;
  /**
   * Detect anomalies using isolation forest method (simplified)
   */
  detect_isolation_forest(data: WasmTensor, _n_trees: number): Array<any>;
  /**
   * Real-time anomaly detection for streaming data
   */
  detect_realtime(value: number): any;
  /**
   * Get detector statistics
   */
  get_statistics(): string;
  /**
   * Reset detector state
   */
  reset(): void;
  /**
   * Update threshold
   */
  set_threshold(threshold: number): void;
  /**
   * Get current threshold
   */
  get_threshold(): number;
}
/**
 * Batch Normalization layer for WASM
 * WASM用のバッチ正規化レイヤー
 */
export class WasmBatchNorm {
  free(): void;
  /**
   * Create a new Batch Normalization layer
   * 新しいバッチ正規化レイヤーを作成
   */
  constructor(num_features: number, momentum: number, epsilon: number);
  /**
   * Set training mode
   * 訓練モードを設定
   */
  set_training(training: boolean): void;
  /**
   * Set scale (gamma) parameters
   * スケール（ガンマ）パラメータを設定
   */
  set_gamma(gamma: Float32Array): void;
  /**
   * Set shift (beta) parameters
   * シフト（ベータ）パラメータを設定
   */
  set_beta(beta: Float32Array): void;
  /**
   * Forward pass through batch normalization
   * バッチ正規化の順伝播
   */
  forward(input: Float32Array, batch_size: number): Float32Array;
  /**
   * Get running mean for inspection
   * 実行中の平均値を取得（検査用）
   */
  get_running_mean(): Float32Array;
  /**
   * Get running variance for inspection
   * 実行中の分散値を取得（検査用）
   */
  get_running_var(): Float32Array;
}
/**
 * Bernoulli distribution for WASM
 * WASM用ベルヌーイ分布
 */
export class WasmBernoulli {
  free(): void;
  /**
   * Create new Bernoulli distribution
   */
  constructor(p: number, seed: number);
  /**
   * Sample single value (0 or 1)
   */
  sample(): number;
  /**
   * Sample multiple values
   */
  sample_n(n: number): Uint32Array;
  /**
   * Probability mass function
   */
  pmf(x: number): number;
  /**
   * Log probability mass function
   */
  log_pmf(x: number): number;
  /**
   * Get mean
   */
  mean(): number;
  /**
   * Get variance
   */
  variance(): number;
}
/**
 * WASM wrapper for CenterCrop transformation
 */
export class WasmCenterCrop {
  free(): void;
  /**
   * Create new center crop transform
   */
  constructor(height: number, width: number);
  /**
   * Apply center crop to tensor
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * WASM wrapper for ColorJitter transformation
 */
export class WasmColorJitter {
  free(): void;
  /**
   * Create new color jitter transform
   */
  constructor(brightness: number, contrast: number, saturation: number, hue: number);
  /**
   * Apply color jitter to tensor
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * 2D Convolutional layer for WASM
 * WASM用2次元畳み込みレイヤー
 */
export class WasmConv2d {
  free(): void;
  /**
   * Create new 2D convolutional layer
   * 新しい2次元畳み込みレイヤーを作成
   */
  constructor(in_channels: number, out_channels: number, kernel_size: number, stride: number, padding: number, bias: boolean);
  /**
   * Forward pass through convolution layer
   * 畳み込みレイヤーの順伝播
   */
  forward(input: Float32Array, batch_size: number, input_height: number, input_width: number): Float32Array;
  /**
   * Calculate output dimensions for given input
   * 入力に対する出力次元を計算
   */
  output_shape(input_height: number, input_width: number): Uint32Array;
  /**
   * Get layer weights
   */
  get_weights(): Float32Array;
  /**
   * Get layer bias
   */
  get_bias(): Float32Array | undefined;
  /**
   * Update weights
   */
  update_weights(new_weights: Float32Array): void;
  /**
   * Get layer configuration
   */
  get_config(): object;
}
/**
 * Exponential distribution for WASM
 * WASM用指数分布
 */
export class WasmExponential {
  free(): void;
  /**
   * Create new exponential distribution
   */
  constructor(rate: number, seed: number);
  /**
   * Create standard exponential distribution (rate=1)
   */
  static standard(seed: number): WasmExponential;
  /**
   * Sample single value using inverse transform sampling
   */
  sample(): number;
  /**
   * Sample multiple values
   */
  sample_n(n: number): Float32Array;
  /**
   * Probability density function
   */
  pdf(x: number): number;
  /**
   * Log probability density function
   */
  log_pdf(x: number): number;
  /**
   * Cumulative distribution function
   */
  cdf(x: number): number;
  /**
   * Get mean
   */
  mean(): number;
  /**
   * Get variance
   */
  variance(): number;
}
/**
 * Group Normalization for WASM
 * WASM用のグループ正規化
 */
export class WasmGroupNorm {
  free(): void;
  /**
   * Create a new Group Normalization layer
   * 新しいグループ正規化レイヤーを作成
   */
  constructor(num_groups: number, num_channels: number, epsilon: number);
  /**
   * Set scale (gamma) parameters
   * スケール（ガンマ）パラメータを設定
   */
  set_gamma(gamma: Float32Array): void;
  /**
   * Set shift (beta) parameters
   * シフト（ベータ）パラメータを設定
   */
  set_beta(beta: Float32Array): void;
  /**
   * Forward pass through group normalization
   * グループ正規化の順伝播
   */
  forward(input: Float32Array, batch_size: number, height: number, width: number): Float32Array;
}
/**
 * Learning rate scheduler for WASM optimizers
 * WASMオプティマイザ用学習率スケジューラ
 */
export class WasmLRScheduler {
  private constructor();
  free(): void;
  /**
   * Create StepLR scheduler
   */
  static step_lr(initial_lr: number, step_size: number, gamma: number): WasmLRScheduler;
  /**
   * Create ExponentialLR scheduler
   */
  static exponential_lr(initial_lr: number, gamma: number): WasmLRScheduler;
  /**
   * Create CosineAnnealingLR scheduler
   */
  static cosine_annealing_lr(initial_lr: number, t_max: number, eta_min: number): WasmLRScheduler;
  /**
   * Step the scheduler and get updated learning rate
   */
  step(): number;
  /**
   * Get current learning rate
   */
  get_lr(): number;
  /**
   * Reset scheduler
   */
  reset(): void;
}
/**
 * Layer Normalization for WASM
 * WASM用のレイヤー正規化
 */
export class WasmLayerNorm {
  free(): void;
  /**
   * Create a new Layer Normalization layer
   * 新しいレイヤー正規化レイヤーを作成
   */
  constructor(normalized_shape: Uint32Array, epsilon: number);
  /**
   * Set scale (gamma) parameters
   * スケール（ガンマ）パラメータを設定
   */
  set_gamma(gamma: Float32Array): void;
  /**
   * Set shift (beta) parameters
   * シフト（ベータ）パラメータを設定
   */
  set_beta(beta: Float32Array): void;
  /**
   * Forward pass through layer normalization
   * レイヤー正規化の順伝播
   */
  forward(input: Float32Array): Float32Array;
}
/**
 * Complete linear layer for WASM neural networks
 * WASM用完全な線形レイヤー
 */
export class WasmLinear {
  free(): void;
  /**
   * Create new linear layer with Xavier/Glorot initialization
   * Xavier/Glorot初期化による新しい線形レイヤーを作成
   */
  constructor(in_features: number, out_features: number, bias: boolean);
  /**
   * Create linear layer with custom initialization
   * カスタム初期化による線形レイヤーを作成
   */
  static with_weights(in_features: number, out_features: number, weights: Float32Array, bias?: Float32Array | null): WasmLinear;
  /**
   * Forward pass through linear layer
   * 線形レイヤーの順伝播
   */
  forward(input: Float32Array, batch_size: number): Float32Array;
  /**
   * Get layer parameters for training
   * 訓練用のレイヤーパラメータを取得
   */
  get_weights(): Float32Array;
  /**
   * Get bias parameters
   * バイアスパラメータを取得
   */
  get_bias(): Float32Array | undefined;
  /**
   * Update weights with new values
   * 新しい値で重みを更新
   */
  update_weights(new_weights: Float32Array): void;
  /**
   * Update bias with new values
   * 新しい値でバイアスを更新
   */
  update_bias(new_bias: Float32Array): void;
  /**
   * Get input features count
   */
  in_features(): number;
  /**
   * Get output features count  
   */
  out_features(): number;
  /**
   * Check if layer has bias
   */
  has_bias(): boolean;
}
/**
 * WASM-specific logging utilities
 */
export class WasmLogger {
  private constructor();
  free(): void;
  /**
   * Log info message
   */
  static info(message: string): void;
  /**
   * Log warning message  
   */
  static warn(message: string): void;
  /**
   * Log error message
   */
  static error(message: string): void;
  /**
   * Log debug message
   */
  static debug(message: string): void;
}
/**
 * WASM-compatible loss functions
 * WASM互換損失関数
 */
export class WasmLoss {
  private constructor();
  free(): void;
  /**
   * Mean Squared Error (MSE) loss
   * MSE(y_pred, y_true) = mean((y_pred - y_true)²)
   */
  static mse_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Mean Absolute Error (MAE) loss
   * MAE(y_pred, y_true) = mean(|y_pred - y_true|)
   */
  static mae_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Huber loss (smooth L1 loss)
   * Combines MSE and MAE for robustness
   */
  static huber_loss(predictions: Float32Array, targets: Float32Array, delta: number): number;
  /**
   * Cross-entropy loss for binary classification
   * Binary Cross-Entropy: -mean(y*log(p) + (1-y)*log(1-p))
   */
  static binary_cross_entropy_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Cross-entropy loss for multiclass classification
   * Input: logits (raw scores), targets (one-hot or class indices)
   */
  static cross_entropy_loss(logits: Float32Array, targets: Float32Array): number;
  /**
   * Sparse cross-entropy loss (targets as class indices instead of one-hot)
   * logits: \[batch_size * num_classes\], targets: \[batch_size\] (class indices)
   */
  static sparse_cross_entropy_loss(logits: Float32Array, targets: Uint32Array, num_classes: number): number;
  /**
   * KL Divergence loss
   * KL(P||Q) = sum(P * log(P/Q))
   */
  static kl_divergence_loss(p_distribution: Float32Array, q_distribution: Float32Array): number;
  /**
   * Focal loss for handling class imbalance
   * FL(pt) = -α(1-pt)^γ log(pt)
   */
  static focal_loss(predictions: Float32Array, targets: Float32Array, alpha: number, gamma: number): number;
  /**
   * Cosine similarity loss
   * Loss = 1 - cosine_similarity(pred, target)
   */
  static cosine_similarity_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Hinge loss for SVM-style classification
   * Hinge(y, f(x)) = max(0, 1 - y * f(x))
   */
  static hinge_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Squared hinge loss (smooth version of hinge loss)
   */
  static squared_hinge_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Log-cosh loss (smooth version of MAE)
   * LogCosh(y_pred, y_true) = mean(log(cosh(y_pred - y_true)))
   */
  static log_cosh_loss(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Combined loss function selector
   * 損失関数セレクター
   */
  static compute_loss(predictions: Float32Array, targets: Float32Array, loss_type: string): number;
  /**
   * Get loss function gradient for backpropagation
   * 逆伝播用の損失関数勾配を取得
   */
  static loss_gradient(predictions: Float32Array, targets: Float32Array, loss_type: string): Float32Array;
}
/**
 * Memory usage monitor for WASM
 */
export class WasmMemoryMonitor {
  free(): void;
  /**
   * Create a new memory usage monitor
   * 新しいメモリ使用量モニターを作成
   */
  constructor();
  /**
   * Record memory allocation
   */
  record_allocation(size: number): void;
  /**
   * Record memory deallocation
   */
  record_deallocation(size: number): void;
  /**
   * Get current memory usage
   */
  current_usage(): number;
  /**
   * Get peak memory usage
   */
  peak_usage(): number;
  /**
   * Reset statistics
   */
  reset(): void;
}
/**
 * Memory pool for efficient tensor allocation
 */
export class WasmMemoryPool {
  free(): void;
  /**
   * Create new memory pool for efficient buffer management
   */
  constructor();
  /**
   * Get a buffer from the pool or allocate new one
   */
  get_buffer(size: number): Float32Array;
  /**
   * Return a buffer to the pool
   */
  return_buffer(buffer: Float32Array): void;
  /**
   * Get pool statistics
   */
  get_stats(): string;
  /**
   * Clear all pools
   */
  clear(): void;
}
/**
 * Model evaluation metrics calculator
 * モデル評価メトリクス計算機
 */
export class WasmMetrics {
  private constructor();
  free(): void;
  /**
   * Calculate accuracy for classification tasks
   * 分類タスクの精度を計算
   */
  static accuracy(predictions: Uint32Array, targets: Uint32Array): number;
  /**
   * Calculate precision for binary classification
   * バイナリ分類の適合率を計算
   */
  static precision(predictions: Uint32Array, targets: Uint32Array, positive_class: number): number;
  /**
   * Calculate recall for binary classification
   * バイナリ分類の再現率を計算
   */
  static recall(predictions: Uint32Array, targets: Uint32Array, positive_class: number): number;
  /**
   * Calculate F1 score
   * F1スコアを計算
   */
  static f1_score(predictions: Uint32Array, targets: Uint32Array, positive_class: number): number;
  /**
   * Calculate confusion matrix for multi-class classification
   * 多クラス分類の混同行列を計算
   */
  static confusion_matrix(predictions: Uint32Array, targets: Uint32Array, num_classes: number): Uint32Array;
  /**
   * Calculate Mean Absolute Error (MAE) for regression
   * 回帰のための平均絶対誤差（MAE）を計算
   */
  static mae(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Calculate Mean Squared Error (MSE) for regression
   * 回帰のための平均二乗誤差（MSE）を計算
   */
  static mse(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Calculate Root Mean Squared Error (RMSE) for regression
   * 回帰のための平方根平均二乗誤差（RMSE）を計算
   */
  static rmse(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Calculate R-squared coefficient for regression
   * 回帰のための決定係数（R二乗）を計算
   */
  static r2_score(predictions: Float32Array, targets: Float32Array): number;
  /**
   * Calculate top-k accuracy for multi-class classification
   * 多クラス分類のためのtop-k精度を計算
   */
  static top_k_accuracy(logits: Float32Array, targets: Uint32Array, num_classes: number, k: number): number;
  /**
   * Calculate comprehensive classification report
   * 包括的な分類レポートを計算
   */
  static classification_report(predictions: Uint32Array, targets: Uint32Array, num_classes: number): object;
}
/**
 * Simple neural network model for WASM
 */
export class WasmModel {
  free(): void;
  /**
   * Create new neural network model
   */
  constructor();
  /**
   * Add linear layer
   */
  add_linear(in_features: number, out_features: number, _bias: boolean): void;
  /**
   * Add ReLU activation
   */
  add_relu(): void;
  /**
   * Get number of layers
   */
  num_layers(): number;
  /**
   * Simple forward pass (placeholder)
   */
  forward(input: WasmTensor): WasmTensor;
}
/**
 * Normal (Gaussian) distribution for WASM
 * WASM用正規（ガウス）分布
 */
export class WasmNormal {
  free(): void;
  /**
   * Create new normal distribution
   */
  constructor(mean: number, std_dev: number, seed: number);
  /**
   * Create standard normal distribution (mean=0, std=1)
   */
  static standard(seed: number): WasmNormal;
  /**
   * Sample single value using Box-Muller transform
   */
  sample(): number;
  /**
   * Sample multiple values
   */
  sample_n(n: number): Float32Array;
  /**
   * Probability density function
   */
  pdf(x: number): number;
  /**
   * Log probability density function
   */
  log_pdf(x: number): number;
  /**
   * Cumulative distribution function (using error function approximation)
   */
  cdf(x: number): number;
  /**
   * Get mean
   */
  mean(): number;
  /**
   * Get standard deviation
   */
  std_dev(): number;
  /**
   * Get variance
   */
  variance(): number;
}
/**
 * WASM wrapper for Normalize transformation
 */
export class WasmNormalize {
  free(): void;
  /**
   * Create new normalization transform
   */
  constructor(mean: Float32Array, std: Float32Array);
  /**
   * Apply normalization to tensor
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * Optimizer factory for creating different optimizers
 * 異なるオプティマイザを作成するファクトリ
 */
export class WasmOptimizerFactory {
  private constructor();
  free(): void;
  /**
   * Create optimizer by name
   */
  static create_sgd(learning_rate: number, momentum: number, weight_decay: number): WasmSGD;
  /**
   * Create Adam optimizer
   */
  static create_adam(learning_rate: number, beta1: number, beta2: number, epsilon: number, weight_decay: number): WasmAdam;
  /**
   * Create AdaGrad optimizer
   */
  static create_adagrad(learning_rate: number, epsilon: number): WasmAdaGrad;
  /**
   * Create RMSprop optimizer
   */
  static create_rmsprop(learning_rate: number, alpha: number, epsilon: number, momentum: number): WasmRMSprop;
}
/**
 * Performance monitoring for WASM
 */
export class WasmPerformance {
  free(): void;
  /**
   * Create a new performance monitor
   * 新しいパフォーマンスモニターを作成
   */
  constructor();
  /**
   * Start performance measurement
   */
  start(): void;
  /**
   * Get elapsed time in milliseconds
   */
  elapsed(): number;
  /**
   * Log performance result
   */
  log(operation_name: string): void;
}
/**
 * Data preprocessing utilities for neural networks
 * ニューラルネットワーク用のデータ前処理ユーティリティ
 */
export class WasmPreprocessor {
  private constructor();
  free(): void;
  /**
   * Normalize data using min-max normalization: (x - min) / (max - min)
   * min-max正規化を使用してデータを正規化: (x - min) / (max - min)
   */
  static min_max_normalize(data: Float32Array, min_val: number, max_val: number): Float32Array;
  /**
   * Standardize data using z-score normalization: (x - mean) / std
   * z-score正規化を使用してデータを標準化: (x - mean) / std
   */
  static z_score_normalize(data: Float32Array, mean: number, std: number): Float32Array;
  /**
   * Compute statistics (mean, std, min, max) for normalization
   * 正規化用の統計値（平均、標準偏差、最小値、最大値）を計算
   */
  static compute_stats(data: Float32Array): Float32Array;
  /**
   * One-hot encoding for categorical data
   * カテゴリカルデータのワンホットエンコーディング
   */
  static one_hot_encode(labels: Uint32Array, num_classes: number): Float32Array;
  /**
   * Convert one-hot encoding back to labels
   * ワンホットエンコーディングをラベルに戻す
   */
  static one_hot_decode(one_hot: Float32Array, num_classes: number): Uint32Array;
  /**
   * Data augmentation: add Gaussian noise
   * データ拡張: ガウシアンノイズの追加
   */
  static add_gaussian_noise(data: Float32Array, mean: number, std: number, seed: number): Float32Array;
  /**
   * Train-test split for datasets
   * データセットの訓練・テスト分割
   */
  static train_test_split(features: Float32Array, targets: Float32Array, feature_size: number, test_ratio: number, seed: number): object;
  /**
   * Batch data for training
   * 訓練用のデータバッチ化
   */
  static create_batches(features: Float32Array, targets: Float32Array, feature_size: number, batch_size: number): Array<any>;
}
/**
 * Processing pipeline for analysis operations
 */
export class WasmProcessingPipeline {
  free(): void;
  /**
   * Create new processing pipeline
   */
  constructor(parallel_execution: boolean);
  /**
   * Add operation to pipeline
   */
  add_operation(operation_name: string): void;
  /**
   * Get operation count
   */
  operation_count(): number;
  /**
   * Get pipeline configuration
   */
  get_config(): string;
}
/**
 * WASM wrapper for Quality Metrics
 */
export class WasmQualityMetrics {
  free(): void;
  /**
   * Create new quality metrics analyzer
   */
  constructor(threshold: number);
  /**
   * Calculate data completeness (percentage of non-NaN values)
   */
  completeness(tensor: WasmTensor): number;
  /**
   * Calculate data accuracy (values within expected range)
   */
  accuracy(tensor: WasmTensor, min_val: number, max_val: number): number;
  /**
   * Calculate data consistency (low variance indicator)
   */
  consistency(tensor: WasmTensor): number;
  /**
   * Calculate data validity (percentage of finite values)
   */
  validity(tensor: WasmTensor): number;
  /**
   * Calculate data uniqueness (ratio of unique values)
   */
  uniqueness(tensor: WasmTensor): number;
  /**
   * Comprehensive quality score
   */
  overall_quality(tensor: WasmTensor): number;
  /**
   * Get quality report as JSON string
   */
  quality_report(tensor: WasmTensor): string;
}
/**
 * RMSprop optimizer for WASM
 * WASM用RMSpropオプティマイザ
 */
export class WasmRMSprop {
  free(): void;
  /**
   * Create new RMSprop optimizer
   */
  constructor(learning_rate: number, alpha: number, epsilon: number);
  /**
   * Create RMSprop with momentum
   */
  static with_momentum(learning_rate: number, alpha: number, epsilon: number, momentum: number): WasmRMSprop;
  /**
   * Update parameters with RMSprop algorithm
   */
  step(param_id: string, parameters: Float32Array, gradients: Float32Array): Float32Array;
}
/**
 * WASM wrapper for RandomCrop transformation
 */
export class WasmRandomCrop {
  free(): void;
  /**
   * Create new random crop transform
   */
  constructor(height: number, width: number, padding?: number | null);
  /**
   * Apply random crop to tensor
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * Simple ReLU activation for WASM
 */
export class WasmReLU {
  free(): void;
  /**
   * Create new ReLU activation layer
   */
  constructor();
  /**
   * Apply ReLU activation function
   */
  forward(input: WasmTensor): WasmTensor;
}
/**
 * WASM wrapper for Resize transformation
 */
export class WasmResize {
  free(): void;
  /**
   * Create new resize transform
   */
  constructor(height: number, width: number, interpolation: string);
  /**
   * Apply resize to tensor
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * WASM-compatible random number generator using Linear Congruential Generator
 * WASM互換の線形合同法乱数生成器
 */
export class WasmRng {
  free(): void;
  /**
   * Create new RNG with seed
   */
  constructor(seed: number);
  /**
   * Generate next random u32
   */
  next_u32(): number;
  /**
   * Generate random f32 in [0, 1)
   */
  next_f32(): number;
  /**
   * Generate random f32 in [0, 1) (alternative name for consistency)
   */
  uniform(): number;
}
/**
 * SGD (Stochastic Gradient Descent) optimizer for WASM
 * WASM用SGD（確率的勾配降下法）オプティマイザ
 */
export class WasmSGD {
  free(): void;
  /**
   * Create new SGD optimizer
   */
  constructor(learning_rate: number);
  /**
   * Create SGD with momentum
   */
  static with_momentum(learning_rate: number, momentum: number): WasmSGD;
  /**
   * Create SGD with weight decay
   */
  static with_weight_decay(learning_rate: number, momentum: number, weight_decay: number): WasmSGD;
  /**
   * Update parameters with gradients
   */
  step(param_id: string, parameters: Float32Array, gradients: Float32Array): Float32Array;
  /**
   * Get learning rate
   */
  get_learning_rate(): number;
  /**
   * Set learning rate
   */
  set_learning_rate(lr: number): void;
  /**
   * Clear momentum buffers
   */
  zero_grad(): void;
}
/**
 * WASM-compatible FFT implementation using Cooley-Tukey algorithm
 * WASM互換のCooley-TukeyアルゴリズムFFT実装
 */
export class WasmSignal {
  private constructor();
  free(): void;
  /**
   * Discrete Fourier Transform (DFT) - basic O(N²) implementation
   * 離散フーリエ変換(DFT) - 基本O(N²)実装
   */
  static dft(real_input: Float32Array): object;
  /**
   * Inverse Discrete Fourier Transform (IDFT)
   * 逆離散フーリエ変換(IDFT)
   */
  static idft(real_input: Float32Array, imag_input: Float32Array): object;
  /**
   * Real Fast Fourier Transform (RFFT) - optimized for real inputs
   * 実数高速フーリエ変換(RFFT) - 実数入力用最適化
   */
  static rfft(real_input: Float32Array): object;
  /**
   * Compute power spectral density
   * パワースペクトル密度を計算
   */
  static power_spectrum(real_input: Float32Array): Float32Array;
  /**
   * Apply Hamming window to signal
   * ハミング窓をシグナルに適用
   */
  static hamming_window(signal: Float32Array): Float32Array;
  /**
   * Apply Hanning window to signal
   * ハン窓をシグナルに適用
   */
  static hanning_window(signal: Float32Array): Float32Array;
  /**
   * Apply Blackman window to signal
   * ブラックマン窓をシグナルに適用
   */
  static blackman_window(signal: Float32Array): Float32Array;
  /**
   * Compute magnitude spectrum
   * 振幅スペクトルを計算
   */
  static magnitude_spectrum(real_fft: Float32Array, imag_fft: Float32Array): Float32Array;
  /**
   * Compute phase spectrum
   * 位相スペクトルを計算
   */
  static phase_spectrum(real_fft: Float32Array, imag_fft: Float32Array): Float32Array;
  /**
   * Generate frequency bins for FFT result
   * FFT結果の周波数ビンを生成
   */
  static fft_frequencies(n: number, sample_rate: number): Float32Array;
  /**
   * Generate frequency bins for RFFT result
   * RFFT結果の周波数ビンを生成
   */
  static rfft_frequencies(n: number, sample_rate: number): Float32Array;
  /**
   * Apply low-pass filter (simple moving average)
   * ローパスフィルタを適用（単純移動平均）
   */
  static low_pass_filter(signal: Float32Array, window_size: number): Float32Array;
  /**
   * Apply high-pass filter (difference from moving average)
   * ハイパスフィルタを適用（移動平均との差分）
   */
  static high_pass_filter(signal: Float32Array, window_size: number): Float32Array;
  /**
   * Compute cross-correlation between two signals
   * 2つの信号間の相互相関を計算
   */
  static cross_correlation(signal_a: Float32Array, signal_b: Float32Array): Float32Array;
  /**
   * Compute autocorrelation of a signal
   * 信号の自己相関を計算
   */
  static autocorrelation(signal: Float32Array): Float32Array;
  /**
   * Generate sine wave
   * 正弦波を生成
   */
  static generate_sine_wave(frequency: number, sample_rate: number, duration: number, amplitude: number, phase: number): Float32Array;
  /**
   * Generate cosine wave
   * 余弦波を生成
   */
  static generate_cosine_wave(frequency: number, sample_rate: number, duration: number, amplitude: number, phase: number): Float32Array;
  /**
   * Generate white noise
   * ホワイトノイズを生成
   */
  static generate_white_noise(num_samples: number, amplitude: number, seed: number): Float32Array;
  /**
   * Compute signal energy
   * 信号エネルギーを計算
   */
  static signal_energy(signal: Float32Array): number;
  /**
   * Compute signal power (average energy)
   * 信号パワー（平均エネルギー）を計算
   */
  static signal_power(signal: Float32Array): number;
  /**
   * Compute root mean square (RMS) amplitude
   * 実効値（RMS）振幅を計算
   */
  static rms_amplitude(signal: Float32Array): number;
  /**
   * Find peaks in signal
   * 信号のピークを検出
   */
  static find_peaks(signal: Float32Array, threshold: number): Uint32Array;
  /**
   * Apply gain (amplification) to signal
   * 信号にゲイン（増幅）を適用
   */
  static apply_gain(signal: Float32Array, gain: number): Float32Array;
  /**
   * Normalize signal to range [-1, 1]
   * 信号を[-1, 1]範囲に正規化
   */
  static normalize_signal(signal: Float32Array): Float32Array;
  /**
   * Compute zero-crossing rate
   * ゼロクロッシング率を計算
   */
  static zero_crossing_rate(signal: Float32Array): number;
}
/**
 * Gamma function implementation for WASM
 * WASM用ガンマ関数実装
 */
export class WasmSpecial {
  private constructor();
  free(): void;
  /**
   * Gamma function Γ(x)
   * Using Lanczos approximation for accuracy
   */
  static gamma(x: number): number;
  /**
   * Natural logarithm of gamma function ln(Γ(x))
   */
  static lgamma(x: number): number;
  /**
   * Digamma function ψ(x) = d/dx ln(Γ(x))
   */
  static digamma(x: number): number;
  /**
   * Error function erf(x)
   */
  static erf(x: number): number;
  /**
   * Complementary error function erfc(x) = 1 - erf(x)
   */
  static erfc(x: number): number;
  /**
   * Beta function B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
   */
  static beta(a: number, b: number): number;
  /**
   * Bessel function of the first kind J₀(x)
   */
  static bessel_j0(x: number): number;
  /**
   * Bessel function of the first kind J₁(x)
   */
  static bessel_j1(x: number): number;
  /**
   * Modified Bessel function of the first kind I₀(x)
   */
  static bessel_i0(x: number): number;
}
/**
 * WASM wrapper for Statistical Analysis
 */
export class WasmStatisticalAnalyzer {
  free(): void;
  /**
   * Create new statistical analyzer
   */
  constructor();
  /**
   * Calculate basic statistics
   */
  basic_stats(tensor: WasmTensor): string;
  /**
   * Calculate percentiles
   */
  percentiles(tensor: WasmTensor, percentiles: Float32Array): Array<any>;
  /**
   * Detect outliers using IQR method
   */
  detect_outliers(tensor: WasmTensor): Array<any>;
}
/**
 * Advanced statistical functions for web applications
 */
export class WasmStatisticalFunctions {
  free(): void;
  /**
   * Create new statistical functions instance
   */
  constructor();
  /**
   * Calculate correlation coefficient between two tensors
   */
  correlation(x: WasmTensor, y: WasmTensor): number;
  /**
   * Calculate covariance between two tensors
   */
  covariance(x: WasmTensor, y: WasmTensor): number;
  /**
   * Calculate percentile
   */
  percentile(tensor: WasmTensor, percentile: number): number;
  /**
   * Calculate quantiles
   */
  quantiles(tensor: WasmTensor, q: Float32Array): Array<any>;
}
/**
 * WASM-compatible tensor wrapper
 * WASM互換テンソルラッパー
 */
export class WasmTensor {
  free(): void;
  /**
   * Create a new WASM tensor
   */
  constructor(data: Float32Array, shape: Uint32Array);
  /**
   * Element-wise addition
   */
  add(other: WasmTensor): WasmTensor;
  /**
   * Element-wise multiplication
   */
  multiply(other: WasmTensor): WasmTensor;
  /**
   * ReLU activation
   */
  relu(): WasmTensor;
  /**
   * Sigmoid activation
   */
  sigmoid(): WasmTensor;
  /**
   * Matrix multiplication (2D only)
   */
  matmul(other: WasmTensor): WasmTensor;
  /**
   * Create tensor filled with zeros
   */
  static zeros(shape: Uint32Array): WasmTensor;
  /**
   * Create tensor filled with ones
   */
  static ones(shape: Uint32Array): WasmTensor;
  /**
   * Create tensor with random values
   */
  static random(shape: Uint32Array): WasmTensor;
  /**
   * Reshape tensor
   */
  reshape(new_shape: Uint32Array): WasmTensor;
  /**
   * Get tensor size (total number of elements)
   */
  size(): number;
  /**
   * Get tensor dimensions (number of axes)
   */
  ndim(): number;
  /**
   * Transpose 2D tensor
   */
  transpose(): WasmTensor;
  /**
   * Element-wise subtraction
   */
  subtract(other: WasmTensor): WasmTensor;
  /**
   * Element-wise division
   */
  divide(other: WasmTensor): WasmTensor;
  /**
   * Scalar addition
   */
  add_scalar(scalar: number): WasmTensor;
  /**
   * Scalar multiplication
   */
  mul_scalar(scalar: number): WasmTensor;
  /**
   * Power function
   */
  pow(exponent: number): WasmTensor;
  /**
   * Square root
   */
  sqrt(): WasmTensor;
  /**
   * Exponential function
   */
  exp(): WasmTensor;
  /**
   * Natural logarithm
   */
  log(): WasmTensor;
  /**
   * Sum all elements
   */
  sum(): number;
  /**
   * Mean of all elements
   */
  mean(): number;
  /**
   * Maximum element
   */
  max(): number;
  /**
   * Minimum element
   */
  min(): number;
  /**
   * Tanh activation
   */
  tanh(): WasmTensor;
  /**
   * Get tensor data
   */
  readonly data: Float32Array;
  /**
   * Get tensor shape
   */
  readonly shape: Uint32Array;
}
/**
 * Memory-aware tensor buffer for WASM
 */
export class WasmTensorBuffer {
  free(): void;
  /**
   * Create new tensor buffer
   */
  constructor(data: Float32Array, shape: Uint32Array);
  /**
   * Create tensor buffer from memory pool
   */
  static from_pool(pool: WasmTensorPool, shape: Uint32Array): WasmTensorBuffer | undefined;
  /**
   * Get buffer size in bytes
   */
  size_bytes(): number;
  /**
   * Release buffer back to pool
   */
  release_to_pool(pool: WasmTensorPool): boolean;
  /**
   * Get buffer data
   */
  readonly data: Float32Array;
  /**
   * Get buffer shape
   */
  readonly shape: Uint32Array;
  /**
   * Get memory ID if allocated from pool
   */
  readonly memory_id: number | undefined;
}
/**
 * Advanced tensor operations for neural networks
 * ニューラルネットワーク用の高度なテンソル操作
 */
export class WasmTensorOps {
  private constructor();
  free(): void;
  /**
   * Matrix multiplication: A @ B
   * 行列積: A @ B
   */
  static matmul(a: Float32Array, a_rows: number, a_cols: number, b: Float32Array, b_rows: number, b_cols: number): Float32Array;
  /**
   * Transpose a 2D matrix
   * 2D行列の転置
   */
  static transpose(matrix: Float32Array, rows: number, cols: number): Float32Array;
  /**
   * Reshape tensor while preserving total elements
   * 総要素数を保持しながらテンソルをリシェイプ
   */
  static reshape(data: Float32Array, new_shape: Uint32Array): Float32Array;
  /**
   * Concatenate tensors along specified axis
   * 指定軸でテンソルを連結
   */
  static concatenate(tensors: Array<any>, shapes: Array<any>, axis: number): object;
  /**
   * Split tensor along specified axis
   * 指定軸でテンソルを分割
   */
  static split(data: Float32Array, shape: Uint32Array, axis: number, split_sizes: Uint32Array): Array<any>;
  /**
   * Compute tensor dot product (Einstein summation)
   * テンソル内積の計算（アインシュタイン記法）
   */
  static dot_product(a: Float32Array, b: Float32Array): number;
  /**
   * Element-wise operations
   * 要素ごとの操作
   */
  static element_wise_add(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * Element-wise multiplication
   * 要素ごとの乗算
   */
  static element_wise_mul(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * Element-wise subtraction
   * 要素ごとの減算
   */
  static element_wise_sub(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * Element-wise division
   * 要素ごとの除算
   */
  static element_wise_div(a: Float32Array, b: Float32Array): Float32Array;
  /**
   * Reduce operations
   * リダクション操作
   */
  static reduce_sum(data: Float32Array, axis: number | null | undefined, shape: Uint32Array): object;
  /**
   * Reduce mean
   * 平均値の計算
   */
  static reduce_mean(data: Float32Array, axis: number | null | undefined, shape: Uint32Array): object;
  /**
   * Broadcasting addition for tensors of different shapes
   * 異なる形状のテンソルのブロードキャスト加算
   */
  static broadcast_add(a: Float32Array, a_shape: Uint32Array, b: Float32Array, b_shape: Uint32Array): object;
  /**
   * Compute gradient clipping (useful for training)
   * 勾配クリッピングを計算（訓練に有用）
   */
  static clip_gradients(gradients: Float32Array, max_norm: number): Float32Array;
  /**
   * Apply dropout during training (sets random elements to zero)
   * 訓練中のドロップアウトを適用（ランダム要素をゼロに設定）
   */
  static dropout(input: Float32Array, dropout_rate: number, training: boolean, seed: number): Float32Array;
}
/**
 * Memory pool for WASM tensor operations
 */
export class WasmTensorPool {
  free(): void;
  /**
   * Create new memory pool with specified capacity
   */
  constructor(capacity_bytes: number);
  /**
   * Allocate memory block
   */
  allocate(size: number): number | undefined;
  /**
   * Deallocate memory block
   */
  deallocate(index: number): boolean;
  /**
   * Get total allocated memory in elements
   */
  get_total_allocated(): number;
  /**
   * Get memory usage statistics
   */
  get_usage_stats(): object;
  /**
   * Force garbage collection of unused blocks
   */
  garbage_collect(): number;
  /**
   * Clear all allocated memory
   */
  clear(): void;
}
/**
 * Tensor-based special functions for WASM
 */
export class WasmTensorSpecial {
  private constructor();
  free(): void;
  /**
   * Apply gamma function to tensor elements
   */
  static tensor_gamma(tensor: WasmTensor): WasmTensor;
  /**
   * Apply lgamma function to tensor elements
   */
  static tensor_lgamma(tensor: WasmTensor): WasmTensor;
  /**
   * Apply erf function to tensor elements
   */
  static tensor_erf(tensor: WasmTensor): WasmTensor;
  /**
   * Apply bessel_j0 function to tensor elements
   */
  static tensor_bessel_j0(tensor: WasmTensor): WasmTensor;
}
/**
 * WASM wrapper for Time Series Anomaly Detector
 */
export class WasmTimeSeriesDetector {
  free(): void;
  /**
   * Create new time series anomaly detector
   */
  constructor(window_size: number, seasonal_period?: number | null);
  /**
   * Add new data point and check for anomalies
   */
  add_point(timestamp: number, value: number): any;
  /**
   * Get trend analysis
   */
  get_trend_analysis(): string;
  /**
   * Get seasonal analysis
   */
  get_seasonal_analysis(): string;
}
/**
 * WASM wrapper for ToTensor transformation
 */
export class WasmToTensor {
  free(): void;
  /**
   * Create new to tensor transform
   */
  constructor();
  /**
   * Apply to tensor transformation (identity operation)
   */
  apply(tensor: WasmTensor): WasmTensor;
  /**
   * Get transformation name
   */
  name(): string;
}
/**
 * Simple pipeline for chaining transformations
 */
export class WasmTransformPipeline {
  free(): void;
  /**
   * Create new pipeline
   */
  constructor(cache_enabled: boolean);
  /**
   * Add transform to pipeline
   */
  add_transform(transform_name: string): void;
  /**
   * Get number of transforms in pipeline
   */
  length(): number;
  /**
   * Clear all transforms
   */
  clear(): void;
  /**
   * Execute pipeline on tensor (simplified)
   */
  execute(input: WasmTensor): WasmTensor;
  /**
   * Get pipeline statistics
   */
  get_stats(): string;
}
/**
 * Uniform distribution for WASM
 * WASM用一様分布
 */
export class WasmUniform {
  free(): void;
  /**
   * Create new uniform distribution
   */
  constructor(low: number, high: number, seed: number);
  /**
   * Create standard uniform distribution [0, 1)
   */
  static standard(seed: number): WasmUniform;
  /**
   * Sample single value
   */
  sample(): number;
  /**
   * Sample multiple values
   */
  sample_n(n: number): Float32Array;
  /**
   * Probability density function
   */
  pdf(x: number): number;
  /**
   * Log probability density function
   */
  log_pdf(x: number): number;
  /**
   * Cumulative distribution function
   */
  cdf(x: number): number;
  /**
   * Get mean
   */
  mean(): number;
  /**
   * Get variance
   */
  variance(): number;
}
/**
 * Vision utilities for WASM
 * WASM用画像処理ユーティリティ
 */
export class WasmVision {
  private constructor();
  free(): void;
  /**
   * Resize image using bilinear interpolation
   * バイリニア補間による画像リサイズ
   */
  static resize(image_data: Float32Array, original_height: number, original_width: number, new_height: number, new_width: number, channels: number): Float32Array;
  /**
   * Normalize image with mean and standard deviation
   * 平均と標準偏差による画像正規化
   */
  static normalize(image_data: Float32Array, mean: Float32Array, std: Float32Array, channels: number): Float32Array;
  /**
   * Convert RGB to grayscale
   * RGBからグレースケールに変換
   */
  static rgb_to_grayscale(rgb_data: Float32Array, height: number, width: number): Float32Array;
  /**
   * Apply Gaussian blur
   * ガウシアンブラーを適用
   */
  static gaussian_blur(image_data: Float32Array, height: number, width: number, channels: number, sigma: number): Float32Array;
  /**
   * Crop image to specified region
   * 指定領域に画像をクロップ
   */
  static crop(image_data: Float32Array, height: number, width: number, channels: number, start_y: number, start_x: number, crop_height: number, crop_width: number): Float32Array;
  /**
   * Flip image horizontally
   * 画像を水平反転
   */
  static flip_horizontal(image_data: Float32Array, height: number, width: number, channels: number): Float32Array;
  /**
   * Flip image vertically
   * 画像を垂直反転
   */
  static flip_vertical(image_data: Float32Array, height: number, width: number, channels: number): Float32Array;
  /**
   * Rotate image by 90 degrees clockwise
   * 画像を時計回りに90度回転
   */
  static rotate_90_cw(image_data: Float32Array, height: number, width: number, channels: number): Float32Array;
  /**
   * Apply center crop (crop from center of image)
   * センタークロップ（画像中央からクロップ）
   */
  static center_crop(image_data: Float32Array, height: number, width: number, channels: number, crop_size: number): Float32Array;
  /**
   * Adjust image brightness
   * 画像の明度を調整
   */
  static adjust_brightness(image_data: Float32Array, factor: number): Float32Array;
  /**
   * Adjust image contrast
   * 画像のコントラストを調整
   */
  static adjust_contrast(image_data: Float32Array, factor: number): Float32Array;
  /**
   * Add Gaussian noise to image (data augmentation)
   * 画像にガウシアンノイズを追加（データ拡張）
   */
  static add_gaussian_noise(image_data: Float32Array, std_dev: number): Float32Array;
  /**
   * Apply random rotation (for data augmentation)
   * ランダム回転を適用（データ拡張用）
   */
  static random_rotation(image_data: Float32Array, height: number, width: number, channels: number, max_angle_deg: number): Float32Array;
  /**
   * Apply edge detection (Sobel filter)
   * エッジ検出（Sobelフィルター）
   */
  static edge_detection(image_data: Float32Array, height: number, width: number): Float32Array;
  /**
   * Convert image from 0-255 range to 0-1 range
   * 画像を0-255範囲から0-1範囲に変換
   */
  static to_float(image_data: Uint8Array): Float32Array;
  /**
   * Convert image from 0-1 range to 0-255 range
   * 画像を0-1範囲から0-255範囲に変換
   */
  static to_uint8(image_data: Float32Array): Uint8Array;
  /**
   * Calculate image histogram
   * 画像のヒストグラムを計算
   */
  static histogram(image_data: Float32Array, bins: number): Uint32Array;
  /**
   * Apply histogram equalization
   * ヒストグラム均等化を適用
   */
  static histogram_equalization(image_data: Float32Array, bins: number): Float32Array;
}
export class WebGPUSimple {
  free(): void;
  constructor();
  initialize(): Promise<string>;
  check_webgpu_support(): Promise<boolean>;
  tensor_add_cpu(a: Float32Array, b: Float32Array): Float32Array;
  tensor_mul_cpu(a: Float32Array, b: Float32Array): Float32Array;
  matrix_multiply_cpu(a: Float32Array, b: Float32Array, m: number, n: number, k: number): Float32Array;
  relu_cpu(input: Float32Array): Float32Array;
  sigmoid_cpu(input: Float32Array): Float32Array;
  get_status(): string;
  get_chrome_info(): string;
}
/**
 * WebGPU Simple Demo struct for browser demonstration
 * ブラウザデモ用WebGPUシンプルデモ構造体
 */
export class WebGPUSimpleDemo {
  free(): void;
  constructor();
  initialize(): Promise<string>;
  run_tensor_addition_demo(): string;
  run_matrix_multiplication_demo(): string;
  run_activation_functions_demo(): string;
  run_performance_benchmark(): string;
  run_comprehensive_demo(): Promise<string>;
  get_all_results(): string[];
  cleanup(): void;
}
/**
 * Web Worker utilities for background computation
 */
export class WorkerManager {
  free(): void;
  /**
   * Create new web worker manager
   */
  constructor();
  /**
   * Create and start a web worker
   */
  create_worker(script_url: string): void;
  /**
   * Send tensor data to worker
   */
  send_tensor(tensor: WasmTensor): void;
  /**
   * Terminate worker
   */
  terminate(): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmlinear_free: (a: number, b: number) => void;
  readonly wasmlinear_new: (a: number, b: number, c: number) => number;
  readonly wasmlinear_with_weights: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly wasmlinear_forward: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly wasmlinear_get_weights: (a: number) => [number, number];
  readonly wasmlinear_get_bias: (a: number) => [number, number];
  readonly wasmlinear_update_weights: (a: number, b: number, c: number) => [number, number];
  readonly wasmlinear_update_bias: (a: number, b: number, c: number) => [number, number];
  readonly wasmlinear_in_features: (a: number) => number;
  readonly wasmlinear_out_features: (a: number) => number;
  readonly wasmlinear_has_bias: (a: number) => number;
  readonly __wbg_wasmconv2d_free: (a: number, b: number) => void;
  readonly wasmconv2d_new: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly wasmconv2d_forward: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
  readonly wasmconv2d_output_shape: (a: number, b: number, c: number) => [number, number];
  readonly wasmconv2d_get_weights: (a: number) => [number, number];
  readonly wasmconv2d_get_bias: (a: number) => [number, number];
  readonly wasmconv2d_update_weights: (a: number, b: number, c: number) => [number, number];
  readonly wasmconv2d_get_config: (a: number) => any;
  readonly wasmrelu_forward: (a: number, b: number) => number;
  readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
  readonly wasmmodel_new: () => number;
  readonly wasmmodel_add_linear: (a: number, b: number, c: number, d: number) => void;
  readonly wasmmodel_add_relu: (a: number) => void;
  readonly wasmmodel_num_layers: (a: number) => number;
  readonly wasmmodel_forward: (a: number, b: number) => number;
  readonly __wbg_browserstorage_free: (a: number, b: number) => void;
  readonly browserstorage_save_tensor: (a: number, b: number, c: number, d: number) => [number, number];
  readonly browserstorage_load_tensor: (a: number, b: number, c: number) => [number, number, number];
  readonly browserstorage_list_tensor_keys: (a: number) => [number, number, number];
  readonly browserstorage_clear_tensors: (a: number) => [number, number];
  readonly fileloader_create_file_input: (a: number) => [number, number, number];
  readonly __wbg_canvasrenderer_free: (a: number, b: number) => void;
  readonly canvasrenderer_new: (a: number, b: number) => [number, number, number];
  readonly canvasrenderer_render_heatmap: (a: number, b: number) => [number, number];
  readonly canvasrenderer_clear: (a: number) => void;
  readonly __wbg_workermanager_free: (a: number, b: number) => void;
  readonly workermanager_new: () => number;
  readonly workermanager_create_worker: (a: number, b: number, c: number) => [number, number];
  readonly workermanager_send_tensor: (a: number, b: number) => [number, number];
  readonly workermanager_terminate: (a: number) => void;
  readonly performancemonitor_get_memory_info: () => [number, number, number];
  readonly performancemonitor_time_function: (a: number, b: number) => void;
  readonly performancemonitor_time_end: (a: number, b: number) => void;
  readonly jsinterop_ones: (a: number, b: any) => number;
  readonly jsinterop_zeros: (a: number, b: any) => number;
  readonly jsinterop_random_tensor: (a: number, b: any, c: number, d: number) => number;
  readonly jsinterop_log_tensor: (a: number, b: number, c: number, d: number) => void;
  readonly tensor_from_float32_array: (a: any, b: any) => [number, number, number];
  readonly tensor_to_float32_array: (a: number) => any;
  readonly tensor_from_nested_array: (a: any) => [number, number, number];
  readonly tensor_to_nested_array: (a: number) => [number, number, number];
  readonly tensor_slice: (a: number, b: number, c: number) => [number, number, number];
  readonly __wbg_benchmarkresult_free: (a: number, b: number) => void;
  readonly benchmarkresult_operation: (a: number) => [number, number];
  readonly benchmarkresult_duration_ms: (a: number) => number;
  readonly benchmarkresult_throughput: (a: number) => number;
  readonly benchmark_matmul: (a: number, b: number) => number;
  readonly optimizedops_fast_matmul: (a: number, b: number, c: number) => [number, number, number];
  readonly optimizedops_vectorized_add: (a: number, b: number, c: number) => [number, number, number];
  readonly optimizedops_fused_relu_add: (a: number, b: number, c: number) => [number, number, number];
  readonly optimizedops_conv1d: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly optimizedops_batch_normalize: (a: number, b: number, c: number) => number;
  readonly __wbg_wasmmemorypool_free: (a: number, b: number) => void;
  readonly wasmmemorypool_new: () => number;
  readonly wasmmemorypool_get_buffer: (a: number, b: number) => [number, number];
  readonly wasmmemorypool_return_buffer: (a: number, b: number, c: number) => void;
  readonly wasmmemorypool_get_stats: (a: number) => [number, number];
  readonly wasmmemorypool_clear: (a: number) => void;
  readonly parallelops_parallel_sum: (a: number, b: number) => number;
  readonly parallelops_parallel_map_add: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensor_new: (a: number, b: number, c: number, d: number) => number;
  readonly wasmtensor_data: (a: number) => [number, number];
  readonly wasmtensor_shape: (a: number) => [number, number];
  readonly wasmtensor_add: (a: number, b: number) => [number, number, number];
  readonly wasmtensor_multiply: (a: number, b: number) => [number, number, number];
  readonly wasmtensor_relu: (a: number) => number;
  readonly wasmtensor_sigmoid: (a: number) => number;
  readonly wasmtensor_matmul: (a: number, b: number) => [number, number, number];
  readonly wasmtensor_zeros: (a: number, b: number) => number;
  readonly wasmtensor_ones: (a: number, b: number) => number;
  readonly wasmtensor_random: (a: number, b: number) => number;
  readonly wasmtensor_reshape: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmtensor_size: (a: number) => number;
  readonly wasmtensor_ndim: (a: number) => number;
  readonly wasmtensor_transpose: (a: number) => [number, number, number];
  readonly wasmtensor_subtract: (a: number, b: number) => [number, number, number];
  readonly wasmtensor_divide: (a: number, b: number) => [number, number, number];
  readonly wasmtensor_add_scalar: (a: number, b: number) => number;
  readonly wasmtensor_mul_scalar: (a: number, b: number) => number;
  readonly wasmtensor_pow: (a: number, b: number) => number;
  readonly wasmtensor_sqrt: (a: number) => number;
  readonly wasmtensor_exp: (a: number) => number;
  readonly wasmtensor_log: (a: number) => number;
  readonly wasmtensor_sum: (a: number) => number;
  readonly wasmtensor_mean: (a: number) => number;
  readonly wasmtensor_max: (a: number) => number;
  readonly wasmtensor_min: (a: number) => number;
  readonly wasmtensor_tanh: (a: number) => number;
  readonly wasmactivation_relu: (a: number, b: number) => [number, number];
  readonly wasmactivation_relu_derivative: (a: number, b: number) => [number, number];
  readonly wasmactivation_leaky_relu: (a: number, b: number, c: number) => [number, number];
  readonly wasmactivation_leaky_relu_derivative: (a: number, b: number, c: number) => [number, number];
  readonly wasmactivation_sigmoid: (a: number, b: number) => [number, number];
  readonly wasmactivation_sigmoid_derivative: (a: number, b: number) => [number, number];
  readonly wasmactivation_tanh: (a: number, b: number) => [number, number];
  readonly wasmactivation_tanh_derivative: (a: number, b: number) => [number, number];
  readonly wasmactivation_softmax: (a: number, b: number) => [number, number];
  readonly wasmactivation_log_softmax: (a: number, b: number) => [number, number];
  readonly wasmactivation_gelu: (a: number, b: number) => [number, number];
  readonly wasmactivation_gelu_derivative: (a: number, b: number) => [number, number];
  readonly wasmactivation_swish: (a: number, b: number) => [number, number];
  readonly wasmactivation_mish: (a: number, b: number) => [number, number];
  readonly wasmactivation_elu: (a: number, b: number, c: number) => [number, number];
  readonly wasmactivation_elu_derivative: (a: number, b: number, c: number) => [number, number];
  readonly wasmactivation_softplus: (a: number, b: number) => [number, number];
  readonly wasmactivation_softsign: (a: number, b: number) => [number, number];
  readonly wasmactivation_relu_2d: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmactivation_softmax_2d: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmactivation_apply_activation: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmrng_next_u32: (a: number) => number;
  readonly wasmrng_next_f32: (a: number) => number;
  readonly __wbg_wasmnormal_free: (a: number, b: number) => void;
  readonly wasmnormal_new: (a: number, b: number, c: number) => number;
  readonly wasmnormal_standard: (a: number) => number;
  readonly wasmnormal_sample: (a: number) => number;
  readonly wasmnormal_sample_n: (a: number, b: number) => [number, number];
  readonly wasmnormal_pdf: (a: number, b: number) => number;
  readonly wasmnormal_log_pdf: (a: number, b: number) => number;
  readonly wasmnormal_cdf: (a: number, b: number) => number;
  readonly wasmnormal_std_dev: (a: number) => number;
  readonly wasmnormal_variance: (a: number) => number;
  readonly __wbg_wasmuniform_free: (a: number, b: number) => void;
  readonly wasmuniform_new: (a: number, b: number, c: number) => number;
  readonly wasmuniform_standard: (a: number) => number;
  readonly wasmuniform_sample: (a: number) => number;
  readonly wasmuniform_sample_n: (a: number, b: number) => [number, number];
  readonly wasmuniform_pdf: (a: number, b: number) => number;
  readonly wasmuniform_log_pdf: (a: number, b: number) => number;
  readonly wasmuniform_cdf: (a: number, b: number) => number;
  readonly wasmuniform_mean: (a: number) => number;
  readonly wasmuniform_variance: (a: number) => number;
  readonly __wbg_wasmbernoulli_free: (a: number, b: number) => void;
  readonly wasmbernoulli_new: (a: number, b: number) => [number, number, number];
  readonly wasmbernoulli_sample: (a: number) => number;
  readonly wasmbernoulli_sample_n: (a: number, b: number) => [number, number];
  readonly wasmbernoulli_pmf: (a: number, b: number) => number;
  readonly wasmbernoulli_log_pmf: (a: number, b: number) => number;
  readonly wasmbernoulli_mean: (a: number) => number;
  readonly wasmbernoulli_variance: (a: number) => number;
  readonly wasmexponential_new: (a: number, b: number) => [number, number, number];
  readonly wasmexponential_standard: (a: number) => number;
  readonly wasmexponential_sample: (a: number) => number;
  readonly wasmexponential_sample_n: (a: number, b: number) => [number, number];
  readonly wasmexponential_pdf: (a: number, b: number) => number;
  readonly wasmexponential_log_pdf: (a: number, b: number) => number;
  readonly wasmexponential_cdf: (a: number, b: number) => number;
  readonly wasmexponential_mean: (a: number) => number;
  readonly wasmexponential_variance: (a: number) => number;
  readonly wasmloss_mse_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_mae_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_huber_loss: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmloss_binary_cross_entropy_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_cross_entropy_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_sparse_cross_entropy_loss: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmloss_kl_divergence_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_focal_loss: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly wasmloss_cosine_similarity_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_hinge_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_squared_hinge_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_log_cosh_loss: (a: number, b: number, c: number, d: number) => number;
  readonly wasmloss_compute_loss: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly wasmloss_loss_gradient: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly __wbg_wasmtensorpool_free: (a: number, b: number) => void;
  readonly wasmtensorpool_new: (a: number) => number;
  readonly wasmtensorpool_allocate: (a: number, b: number) => number;
  readonly wasmtensorpool_deallocate: (a: number, b: number) => number;
  readonly wasmtensorpool_get_total_allocated: (a: number) => number;
  readonly wasmtensorpool_get_usage_stats: (a: number) => any;
  readonly wasmtensorpool_garbage_collect: (a: number) => number;
  readonly wasmtensorpool_clear: (a: number) => void;
  readonly __wbg_wasmtensorbuffer_free: (a: number, b: number) => void;
  readonly wasmtensorbuffer_new: (a: number, b: number, c: number, d: number) => number;
  readonly wasmtensorbuffer_from_pool: (a: number, b: number, c: number) => number;
  readonly wasmtensorbuffer_data: (a: number) => [number, number];
  readonly wasmtensorbuffer_shape: (a: number) => [number, number];
  readonly wasmtensorbuffer_memory_id: (a: number) => number;
  readonly wasmtensorbuffer_size_bytes: (a: number) => number;
  readonly wasmtensorbuffer_release_to_pool: (a: number, b: number) => number;
  readonly wasmmemorymonitor_new: () => number;
  readonly wasmmemorymonitor_record_allocation: (a: number, b: number) => void;
  readonly wasmmemorymonitor_record_deallocation: (a: number, b: number) => void;
  readonly wasmmemorymonitor_current_usage: (a: number) => number;
  readonly wasmmemorymonitor_peak_usage: (a: number) => number;
  readonly wasmmemorymonitor_reset: (a: number) => void;
  readonly wasmsgd_new: (a: number) => number;
  readonly wasmsgd_with_momentum: (a: number, b: number) => number;
  readonly wasmsgd_with_weight_decay: (a: number, b: number, c: number) => number;
  readonly wasmsgd_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly wasmsgd_get_learning_rate: (a: number) => number;
  readonly wasmsgd_set_learning_rate: (a: number, b: number) => void;
  readonly wasmsgd_zero_grad: (a: number) => void;
  readonly __wbg_wasmadam_free: (a: number, b: number) => void;
  readonly wasmadam_new: (a: number) => number;
  readonly wasmadam_with_params: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmadam_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly wasmadam_get_learning_rate: (a: number) => number;
  readonly wasmadam_set_learning_rate: (a: number, b: number) => void;
  readonly wasmadam_get_step_count: (a: number) => number;
  readonly wasmadam_reset: (a: number) => void;
  readonly __wbg_wasmadagrad_free: (a: number, b: number) => void;
  readonly wasmadagrad_new: (a: number, b: number) => number;
  readonly wasmadagrad_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly wasmrmsprop_new: (a: number, b: number, c: number) => number;
  readonly wasmrmsprop_with_momentum: (a: number, b: number, c: number, d: number) => number;
  readonly wasmrmsprop_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly __wbg_wasmlrscheduler_free: (a: number, b: number) => void;
  readonly wasmlrscheduler_step_lr: (a: number, b: number, c: number) => number;
  readonly wasmlrscheduler_exponential_lr: (a: number, b: number) => number;
  readonly wasmlrscheduler_cosine_annealing_lr: (a: number, b: number, c: number) => number;
  readonly wasmlrscheduler_step: (a: number) => number;
  readonly wasmlrscheduler_get_lr: (a: number) => number;
  readonly wasmlrscheduler_reset: (a: number) => void;
  readonly wasmoptimizerfactory_create_sgd: (a: number, b: number, c: number) => number;
  readonly wasmoptimizerfactory_create_adam: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmoptimizerfactory_create_adagrad: (a: number, b: number) => number;
  readonly wasmoptimizerfactory_create_rmsprop: (a: number, b: number, c: number, d: number) => number;
  readonly wasmperformance_new: () => number;
  readonly wasmperformance_start: (a: number) => void;
  readonly wasmperformance_elapsed: (a: number) => number;
  readonly wasmperformance_log: (a: number, b: number, c: number) => void;
  readonly detect_wasm_features: () => any;
  readonly wasmlogger_info: (a: number, b: number) => void;
  readonly wasmlogger_warn: (a: number, b: number) => void;
  readonly wasmlogger_error: (a: number, b: number) => void;
  readonly wasmlogger_debug: (a: number, b: number) => void;
  readonly wasmsignal_dft: (a: number, b: number) => any;
  readonly wasmsignal_idft: (a: number, b: number, c: number, d: number) => any;
  readonly wasmsignal_rfft: (a: number, b: number) => any;
  readonly wasmsignal_power_spectrum: (a: number, b: number) => [number, number];
  readonly wasmsignal_hamming_window: (a: number, b: number) => [number, number];
  readonly wasmsignal_hanning_window: (a: number, b: number) => [number, number];
  readonly wasmsignal_blackman_window: (a: number, b: number) => [number, number];
  readonly wasmsignal_magnitude_spectrum: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmsignal_phase_spectrum: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmsignal_fft_frequencies: (a: number, b: number) => [number, number];
  readonly wasmsignal_rfft_frequencies: (a: number, b: number) => [number, number];
  readonly wasmsignal_low_pass_filter: (a: number, b: number, c: number) => [number, number];
  readonly wasmsignal_high_pass_filter: (a: number, b: number, c: number) => [number, number];
  readonly wasmsignal_cross_correlation: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmsignal_autocorrelation: (a: number, b: number) => [number, number];
  readonly wasmsignal_generate_sine_wave: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmsignal_generate_cosine_wave: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmsignal_generate_white_noise: (a: number, b: number, c: number) => [number, number];
  readonly wasmsignal_signal_energy: (a: number, b: number) => number;
  readonly wasmsignal_signal_power: (a: number, b: number) => number;
  readonly wasmsignal_rms_amplitude: (a: number, b: number) => number;
  readonly wasmsignal_find_peaks: (a: number, b: number, c: number) => [number, number];
  readonly wasmsignal_apply_gain: (a: number, b: number, c: number) => [number, number];
  readonly wasmsignal_normalize_signal: (a: number, b: number) => [number, number];
  readonly wasmsignal_zero_crossing_rate: (a: number, b: number) => number;
  readonly wasmspecial_gamma: (a: number) => number;
  readonly wasmspecial_lgamma: (a: number) => number;
  readonly wasmspecial_digamma: (a: number) => number;
  readonly wasmspecial_erf: (a: number) => number;
  readonly wasmspecial_erfc: (a: number) => number;
  readonly wasmspecial_beta: (a: number, b: number) => number;
  readonly wasmspecial_bessel_j0: (a: number) => number;
  readonly wasmspecial_bessel_j1: (a: number) => number;
  readonly wasmspecial_bessel_i0: (a: number) => number;
  readonly wasmtensorspecial_tensor_gamma: (a: number) => number;
  readonly wasmtensorspecial_tensor_lgamma: (a: number) => number;
  readonly wasmtensorspecial_tensor_erf: (a: number) => number;
  readonly wasmtensorspecial_tensor_bessel_j0: (a: number) => number;
  readonly wasmmetrics_accuracy: (a: number, b: number, c: number, d: number) => number;
  readonly wasmmetrics_precision: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmmetrics_recall: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmmetrics_f1_score: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmmetrics_confusion_matrix: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmmetrics_mae: (a: number, b: number, c: number, d: number) => number;
  readonly wasmmetrics_mse: (a: number, b: number, c: number, d: number) => number;
  readonly wasmmetrics_rmse: (a: number, b: number, c: number, d: number) => number;
  readonly wasmmetrics_r2_score: (a: number, b: number, c: number, d: number) => number;
  readonly wasmmetrics_top_k_accuracy: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly wasmmetrics_classification_report: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly __wbg_wasmbatchnorm_free: (a: number, b: number) => void;
  readonly wasmbatchnorm_new: (a: number, b: number, c: number) => number;
  readonly wasmbatchnorm_set_training: (a: number, b: number) => void;
  readonly wasmbatchnorm_set_gamma: (a: number, b: number, c: number) => void;
  readonly wasmbatchnorm_set_beta: (a: number, b: number, c: number) => void;
  readonly wasmbatchnorm_forward: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmbatchnorm_get_running_mean: (a: number) => [number, number];
  readonly wasmbatchnorm_get_running_var: (a: number) => [number, number];
  readonly __wbg_wasmlayernorm_free: (a: number, b: number) => void;
  readonly wasmlayernorm_new: (a: number, b: number, c: number) => number;
  readonly wasmlayernorm_set_gamma: (a: number, b: number, c: number) => void;
  readonly wasmlayernorm_set_beta: (a: number, b: number, c: number) => void;
  readonly wasmlayernorm_forward: (a: number, b: number, c: number) => [number, number];
  readonly __wbg_wasmgroupnorm_free: (a: number, b: number) => void;
  readonly wasmgroupnorm_new: (a: number, b: number, c: number) => number;
  readonly wasmgroupnorm_set_gamma: (a: number, b: number, c: number) => void;
  readonly wasmgroupnorm_set_beta: (a: number, b: number, c: number) => void;
  readonly wasmgroupnorm_forward: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly wasmpreprocessor_min_max_normalize: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmpreprocessor_z_score_normalize: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmpreprocessor_compute_stats: (a: number, b: number) => [number, number];
  readonly wasmpreprocessor_one_hot_encode: (a: number, b: number, c: number) => [number, number];
  readonly wasmpreprocessor_one_hot_decode: (a: number, b: number, c: number) => [number, number];
  readonly wasmpreprocessor_add_gaussian_noise: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmpreprocessor_train_test_split: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => any;
  readonly wasmpreprocessor_create_batches: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
  readonly wasmtensorops_matmul: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
  readonly wasmtensorops_transpose: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_reshape: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_concatenate: (a: any, b: any, c: number) => any;
  readonly wasmtensorops_split: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => any;
  readonly wasmtensorops_dot_product: (a: number, b: number, c: number, d: number) => number;
  readonly wasmtensorops_element_wise_add: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_element_wise_mul: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_element_wise_sub: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_element_wise_div: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmtensorops_reduce_sum: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly wasmtensorops_reduce_mean: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly wasmtensorops_broadcast_add: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
  readonly wasmtensorops_clip_gradients: (a: number, b: number, c: number) => [number, number];
  readonly wasmtensorops_dropout: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly wasmvision_resize: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
  readonly wasmvision_normalize: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
  readonly wasmvision_rgb_to_grayscale: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly wasmvision_gaussian_blur: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
  readonly wasmvision_crop: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
  readonly wasmvision_flip_horizontal: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly wasmvision_flip_vertical: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly wasmvision_rotate_90_cw: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly wasmvision_center_crop: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
  readonly wasmvision_adjust_brightness: (a: number, b: number, c: number) => [number, number];
  readonly wasmvision_adjust_contrast: (a: number, b: number, c: number) => [number, number];
  readonly wasmvision_add_gaussian_noise: (a: number, b: number, c: number) => [number, number];
  readonly wasmvision_random_rotation: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
  readonly wasmvision_edge_detection: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly wasmvision_to_float: (a: number, b: number) => [number, number];
  readonly wasmvision_to_uint8: (a: number, b: number) => [number, number];
  readonly wasmvision_histogram: (a: number, b: number, c: number) => [number, number];
  readonly wasmvision_histogram_equalization: (a: number, b: number, c: number) => [number, number];
  readonly wasmtransformpipeline_add_transform: (a: number, b: number, c: number) => [number, number];
  readonly wasmtransformpipeline_clear: (a: number) => void;
  readonly wasmtransformpipeline_execute: (a: number, b: number) => [number, number, number];
  readonly wasmtransformpipeline_get_stats: (a: number) => [number, number];
  readonly __wbg_wasmprocessingpipeline_free: (a: number, b: number) => void;
  readonly wasmprocessingpipeline_new: (a: number) => number;
  readonly wasmprocessingpipeline_add_operation: (a: number, b: number, c: number) => [number, number];
  readonly wasmprocessingpipeline_get_config: (a: number) => [number, number];
  readonly wasmadvancedmath_sinh: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_cosh: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_tanh: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_asin: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_acos: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_atan: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_atan2: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmadvancedmath_erf: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_erfc: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_gamma: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_lgamma: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_clamp: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmadvancedmath_sign: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_lerp: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmadvancedmath_pow: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmadvancedmath_pow_tensor: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmadvancedmath_round: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_floor: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_ceil: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_trunc: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_is_finite: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_is_infinite: (a: number, b: number) => [number, number, number];
  readonly wasmadvancedmath_is_nan: (a: number, b: number) => [number, number, number];
  readonly wasmstatisticalfunctions_correlation: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmstatisticalfunctions_covariance: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmstatisticalfunctions_percentile: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmstatisticalfunctions_quantiles: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasm_advanced_math_version: () => [number, number];
  readonly __wbg_wasmanomalydetector_free: (a: number, b: number) => void;
  readonly wasmanomalydetector_new: (a: number, b: number) => [number, number, number];
  readonly wasmanomalydetector_detect_statistical: (a: number, b: number) => [number, number, number];
  readonly wasmanomalydetector_detect_isolation_forest: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmanomalydetector_detect_realtime: (a: number, b: number) => [number, number, number];
  readonly wasmanomalydetector_get_statistics: (a: number) => [number, number, number, number];
  readonly wasmanomalydetector_reset: (a: number) => void;
  readonly wasmanomalydetector_set_threshold: (a: number, b: number) => [number, number];
  readonly wasmanomalydetector_get_threshold: (a: number) => number;
  readonly __wbg_wasmtimeseriesdetector_free: (a: number, b: number) => void;
  readonly wasmtimeseriesdetector_new: (a: number, b: number) => [number, number, number];
  readonly wasmtimeseriesdetector_add_point: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmtimeseriesdetector_get_trend_analysis: (a: number) => [number, number, number, number];
  readonly wasmtimeseriesdetector_get_seasonal_analysis: (a: number) => [number, number, number, number];
  readonly wasm_anomaly_detection_version: () => [number, number];
  readonly create_simple_detector: (a: number) => [number, number, number];
  readonly create_streaming_detector: (a: number) => [number, number, number];
  readonly detect_anomalies_batch: (a: number, b: number, c: number) => [number, number, number];
  readonly __wbg_wasmnormalize_free: (a: number, b: number) => void;
  readonly wasmnormalize_new: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmnormalize_apply: (a: number, b: number) => [number, number, number];
  readonly wasmnormalize_name: (a: number) => [number, number];
  readonly __wbg_wasmresize_free: (a: number, b: number) => void;
  readonly wasmresize_new: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmresize_apply: (a: number, b: number) => [number, number, number];
  readonly wasmresize_name: (a: number) => [number, number];
  readonly wasmcentercrop_new: (a: number, b: number) => [number, number, number];
  readonly wasmcentercrop_apply: (a: number, b: number) => [number, number, number];
  readonly wasmcentercrop_name: (a: number) => [number, number];
  readonly wasmrandomcrop_new: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmrandomcrop_apply: (a: number, b: number) => [number, number, number];
  readonly wasmrandomcrop_name: (a: number) => [number, number];
  readonly __wbg_wasmcolorjitter_free: (a: number, b: number) => void;
  readonly wasmcolorjitter_new: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmcolorjitter_apply: (a: number, b: number) => [number, number, number];
  readonly wasmcolorjitter_name: (a: number) => [number, number];
  readonly wasmtotensor_apply: (a: number, b: number) => [number, number, number];
  readonly wasmtotensor_name: (a: number) => [number, number];
  readonly wasm_transforms_version: () => [number, number];
  readonly create_imagenet_preprocessing: () => [number, number, number];
  readonly create_cifar_preprocessing: () => [number, number, number];
  readonly wasmqualitymetrics_new: (a: number) => [number, number, number];
  readonly wasmqualitymetrics_completeness: (a: number, b: number) => [number, number, number];
  readonly wasmqualitymetrics_accuracy: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmqualitymetrics_consistency: (a: number, b: number) => [number, number, number];
  readonly wasmqualitymetrics_validity: (a: number, b: number) => [number, number, number];
  readonly wasmqualitymetrics_uniqueness: (a: number, b: number) => [number, number, number];
  readonly wasmqualitymetrics_overall_quality: (a: number, b: number) => [number, number, number];
  readonly wasmqualitymetrics_quality_report: (a: number, b: number) => [number, number, number, number];
  readonly wasmstatisticalanalyzer_basic_stats: (a: number, b: number) => [number, number, number, number];
  readonly wasmstatisticalanalyzer_percentiles: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmstatisticalanalyzer_detect_outliers: (a: number, b: number) => [number, number, number];
  readonly wasm_quality_metrics_version: () => [number, number];
  readonly create_quality_analyzer: () => [number, number, number];
  readonly quick_quality_assessment: (a: number) => [number, number, number, number];
  readonly gamma_wasm: (a: number) => number;
  readonly lgamma_wasm: (a: number) => number;
  readonly digamma_wasm: (a: number) => number;
  readonly beta_wasm: (a: number, b: number) => number;
  readonly lbeta_wasm: (a: number, b: number) => number;
  readonly bessel_j_wasm: (a: number, b: number) => number;
  readonly bessel_y_wasm: (a: number, b: number) => number;
  readonly bessel_i_wasm: (a: number, b: number) => number;
  readonly bessel_k_wasm: (a: number, b: number) => number;
  readonly erf_wasm: (a: number) => number;
  readonly erfc_wasm: (a: number) => number;
  readonly erfinv_wasm: (a: number) => number;
  readonly erfcinv_wasm: (a: number) => number;
  readonly gamma_array_wasm: (a: number, b: number) => [number, number];
  readonly bessel_j_array_wasm: (a: number, b: number, c: number) => [number, number];
  readonly erf_array_wasm: (a: number, b: number) => [number, number];
  readonly factorial_wasm: (a: number) => number;
  readonly log_factorial_wasm: (a: number) => number;
  readonly __wbg_specialfunctionsbatch_free: (a: number, b: number) => void;
  readonly specialfunctionsbatch_gamma_batch: (a: number, b: number, c: number) => [number, number];
  readonly specialfunctionsbatch_bessel_j0_batch: (a: number, b: number, c: number) => [number, number];
  readonly specialfunctionsbatch_erf_batch: (a: number, b: number, c: number) => [number, number];
  readonly euler_gamma: () => number;
  readonly sqrt_2pi: () => number;
  readonly log_sqrt_2pi: () => number;
  readonly normaldistributionwasm_sample: (a: number) => number;
  readonly normaldistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly normaldistributionwasm_log_prob: (a: number, b: number) => number;
  readonly normaldistributionwasm_log_prob_array: (a: number, b: number, c: number) => [number, number];
  readonly normaldistributionwasm_variance: (a: number) => number;
  readonly uniformdistributionwasm_sample: (a: number) => number;
  readonly uniformdistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly uniformdistributionwasm_log_prob: (a: number, b: number) => number;
  readonly uniformdistributionwasm_mean: (a: number) => number;
  readonly uniformdistributionwasm_variance: (a: number) => number;
  readonly exponentialdistributionwasm_sample: (a: number) => number;
  readonly exponentialdistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly exponentialdistributionwasm_log_prob: (a: number, b: number) => number;
  readonly exponentialdistributionwasm_mean: (a: number) => number;
  readonly exponentialdistributionwasm_variance: (a: number) => number;
  readonly gammadistributionwasm_sample: (a: number) => number;
  readonly gammadistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly gammadistributionwasm_log_prob: (a: number, b: number) => number;
  readonly gammadistributionwasm_mean: (a: number) => number;
  readonly gammadistributionwasm_variance: (a: number) => number;
  readonly __wbg_betadistributionwasm_free: (a: number, b: number) => void;
  readonly betadistributionwasm_sample: (a: number) => number;
  readonly betadistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly betadistributionwasm_log_prob: (a: number, b: number) => number;
  readonly betadistributionwasm_mean: (a: number) => number;
  readonly betadistributionwasm_variance: (a: number) => number;
  readonly __wbg_bernoullidistributionwasm_free: (a: number, b: number) => void;
  readonly bernoullidistributionwasm_sample: (a: number) => number;
  readonly bernoullidistributionwasm_sample_array: (a: number, b: number) => [number, number];
  readonly bernoullidistributionwasm_sample_f64: (a: number) => number;
  readonly bernoullidistributionwasm_sample_f64_array: (a: number, b: number) => [number, number];
  readonly bernoullidistributionwasm_log_prob: (a: number, b: number) => number;
  readonly bernoullidistributionwasm_variance: (a: number) => number;
  readonly normal_cdf_wasm: (a: number, b: number, c: number) => number;
  readonly normal_quantile_wasm: (a: number, b: number, c: number) => number;
  readonly quick_stats_wasm: (a: number, b: number) => [number, number];
  readonly benchmark_special_functions_wasm: (a: number) => [number, number];
  readonly __wbg_sgdwasm_free: (a: number, b: number) => void;
  readonly sgdwasm_new: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly sgdwasm_step: (a: number, b: number, c: number, d: number, e: number, f: any, g: number, h: number) => void;
  readonly sgdwasm_reset_state: (a: number) => void;
  readonly __wbg_adamwasm_free: (a: number, b: number) => void;
  readonly adamwasm_new: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly adamwasm_step: (a: number, b: number, c: number, d: number, e: number, f: any, g: number, h: number) => void;
  readonly adamwasm_get_learning_rate: (a: number) => number;
  readonly adamwasm_set_learning_rate: (a: number, b: number) => void;
  readonly adamwasm_get_step_count: (a: number) => bigint;
  readonly adamwasm_reset_state: (a: number) => void;
  readonly __wbg_rmspropwasm_free: (a: number, b: number) => void;
  readonly rmspropwasm_new: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly rmspropwasm_step: (a: number, b: number, c: number, d: number, e: number, f: any, g: number, h: number) => void;
  readonly rmspropwasm_reset_state: (a: number) => void;
  readonly __wbg_adagradwasm_free: (a: number, b: number) => void;
  readonly adagradwasm_new: (a: number, b: number, c: number) => number;
  readonly adagradwasm_step: (a: number, b: number, c: number, d: number, e: number, f: any, g: number, h: number) => void;
  readonly adagradwasm_get_learning_rate: (a: number) => number;
  readonly adagradwasm_set_learning_rate: (a: number, b: number) => void;
  readonly adagradwasm_reset_state: (a: number) => void;
  readonly cosine_annealing_wasm: (a: number, b: bigint, c: bigint) => number;
  readonly __wbg_variablewasm_free: (a: number, b: number) => void;
  readonly variablewasm_new: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly variablewasm_data: (a: number) => [number, number];
  readonly variablewasm_shape: (a: number) => [number, number];
  readonly variablewasm_grad: (a: number) => [number, number];
  readonly variablewasm_requires_grad: (a: number) => number;
  readonly variablewasm_zero_grad: (a: number) => void;
  readonly variablewasm_backward: (a: number) => void;
  readonly variablewasm_sum: (a: number) => number;
  readonly variablewasm_mean: (a: number) => number;
  readonly variablewasm_pow: (a: number, b: number) => number;
  readonly __wbg_computationgraphwasm_free: (a: number, b: number) => void;
  readonly computationgraphwasm_new: () => number;
  readonly computationgraphwasm_create_variable: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly computationgraphwasm_get_variable_data: (a: number, b: number, c: number) => [number, number];
  readonly computationgraphwasm_get_variable_grad: (a: number, b: number, c: number) => [number, number];
  readonly computationgraphwasm_add_variables: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly computationgraphwasm_mul_variables: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly computationgraphwasm_backward: (a: number, b: number, c: number) => void;
  readonly computationgraphwasm_zero_grad_all: (a: number) => void;
  readonly computationgraphwasm_clear_graph: (a: number) => void;
  readonly computationgraphwasm_variable_count: (a: number) => number;
  readonly __wbg_linearlayerwasm_free: (a: number, b: number) => void;
  readonly linearlayerwasm_new: (a: number, b: number) => number;
  readonly linearlayerwasm_forward: (a: number, b: number, c: number) => [number, number];
  readonly linearlayerwasm_get_weights: (a: number) => [number, number];
  readonly linearlayerwasm_get_bias: (a: number) => [number, number];
  readonly linearlayerwasm_update_weights: (a: number, b: number, c: number) => number;
  readonly linearlayerwasm_update_bias: (a: number, b: number, c: number) => number;
  readonly relu_wasm: (a: number) => number;
  readonly relu_array_wasm: (a: number, b: number) => [number, number];
  readonly sigmoid_array_wasm: (a: number, b: number) => [number, number];
  readonly tanh_array_wasm: (a: number, b: number) => [number, number];
  readonly softmax_wasm: (a: number, b: number) => [number, number];
  readonly webgpusimple_new: () => number;
  readonly webgpusimple_initialize: (a: number) => any;
  readonly webgpusimple_check_webgpu_support: (a: number) => any;
  readonly webgpusimple_tensor_add_cpu: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly webgpusimple_tensor_mul_cpu: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
  readonly webgpusimple_matrix_multiply_cpu: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
  readonly webgpusimple_relu_cpu: (a: number, b: number, c: number) => [number, number];
  readonly webgpusimple_sigmoid_cpu: (a: number, b: number, c: number) => [number, number];
  readonly webgpusimple_get_status: (a: number) => [number, number];
  readonly webgpusimple_get_chrome_info: (a: number) => [number, number];
  readonly get_browser_webgpu_info: () => [number, number];
  readonly calculate_performance_estimate: (a: number, b: number, c: number) => number;
  readonly webgpusimpledemo_new: () => number;
  readonly webgpusimpledemo_initialize: (a: number) => any;
  readonly webgpusimpledemo_run_tensor_addition_demo: (a: number) => [number, number, number, number];
  readonly webgpusimpledemo_run_matrix_multiplication_demo: (a: number) => [number, number, number, number];
  readonly webgpusimpledemo_run_activation_functions_demo: (a: number) => [number, number, number, number];
  readonly webgpusimpledemo_run_performance_benchmark: (a: number) => [number, number, number, number];
  readonly webgpusimpledemo_run_comprehensive_demo: (a: number) => any;
  readonly webgpusimpledemo_get_all_results: (a: number) => [number, number];
  readonly webgpusimpledemo_cleanup: (a: number) => void;
  readonly init_wasm: () => void;
  readonly initialize_wasm_runtime: () => void;
  readonly wasmtransformpipeline_length: (a: number) => number;
  readonly wasmprocessingpipeline_operation_count: (a: number) => number;
  readonly sigmoid_wasm: (a: number) => number;
  readonly wasmrng_new: (a: number) => number;
  readonly specialfunctionsbatch_new: (a: number) => number;
  readonly exponentialdistributionwasm_new: (a: number) => number;
  readonly bernoullidistributionwasm_new: (a: number) => number;
  readonly rmspropwasm_set_learning_rate: (a: number, b: number) => void;
  readonly sgdwasm_set_learning_rate: (a: number, b: number) => void;
  readonly learning_rate_schedule_wasm: (a: number, b: bigint, c: number, d: bigint) => number;
  readonly __wbg_wasmtransformpipeline_free: (a: number, b: number) => void;
  readonly __wbg_webgpusimpledemo_free: (a: number, b: number) => void;
  readonly __wbg_wasmsgd_free: (a: number, b: number) => void;
  readonly normaldistributionwasm_new: (a: number, b: number) => number;
  readonly uniformdistributionwasm_new: (a: number, b: number) => number;
  readonly gammadistributionwasm_new: (a: number, b: number) => number;
  readonly betadistributionwasm_new: (a: number, b: number) => number;
  readonly wasmrng_uniform: (a: number) => number;
  readonly __wbg_wasmtensor_free: (a: number, b: number) => void;
  readonly wasmnormal_mean: (a: number) => number;
  readonly normaldistributionwasm_mean: (a: number) => number;
  readonly normaldistributionwasm_std_dev: (a: number) => number;
  readonly bernoullidistributionwasm_mean: (a: number) => number;
  readonly rmspropwasm_get_learning_rate: (a: number) => number;
  readonly sgdwasm_get_learning_rate: (a: number) => number;
  readonly tanh_wasm: (a: number) => number;
  readonly __wbg_wasmrelu_free: (a: number, b: number) => void;
  readonly __wbg_fileloader_free: (a: number, b: number) => void;
  readonly __wbg_performancemonitor_free: (a: number, b: number) => void;
  readonly __wbg_jsinterop_free: (a: number, b: number) => void;
  readonly __wbg_optimizedops_free: (a: number, b: number) => void;
  readonly __wbg_parallelops_free: (a: number, b: number) => void;
  readonly __wbg_wasmactivation_free: (a: number, b: number) => void;
  readonly __wbg_wasmexponential_free: (a: number, b: number) => void;
  readonly __wbg_wasmloss_free: (a: number, b: number) => void;
  readonly __wbg_wasmmemorymonitor_free: (a: number, b: number) => void;
  readonly __wbg_wasmoptimizerfactory_free: (a: number, b: number) => void;
  readonly __wbg_wasmlogger_free: (a: number, b: number) => void;
  readonly __wbg_wasmsignal_free: (a: number, b: number) => void;
  readonly __wbg_wasmspecial_free: (a: number, b: number) => void;
  readonly __wbg_wasmtensorspecial_free: (a: number, b: number) => void;
  readonly __wbg_wasmmetrics_free: (a: number, b: number) => void;
  readonly __wbg_wasmpreprocessor_free: (a: number, b: number) => void;
  readonly __wbg_wasmtensorops_free: (a: number, b: number) => void;
  readonly __wbg_wasmvision_free: (a: number, b: number) => void;
  readonly __wbg_wasmadvancedmath_free: (a: number, b: number) => void;
  readonly __wbg_wasmstatisticalfunctions_free: (a: number, b: number) => void;
  readonly __wbg_wasmcentercrop_free: (a: number, b: number) => void;
  readonly __wbg_wasmrandomcrop_free: (a: number, b: number) => void;
  readonly __wbg_wasmtotensor_free: (a: number, b: number) => void;
  readonly __wbg_wasmrng_free: (a: number, b: number) => void;
  readonly __wbg_wasmstatisticalanalyzer_free: (a: number, b: number) => void;
  readonly __wbg_wasmqualitymetrics_free: (a: number, b: number) => void;
  readonly __wbg_uniformdistributionwasm_free: (a: number, b: number) => void;
  readonly __wbg_wasmperformance_free: (a: number, b: number) => void;
  readonly __wbg_normaldistributionwasm_free: (a: number, b: number) => void;
  readonly __wbg_gammadistributionwasm_free: (a: number, b: number) => void;
  readonly __wbg_exponentialdistributionwasm_free: (a: number, b: number) => void;
  readonly __wbg_webgpusimple_free: (a: number, b: number) => void;
  readonly wasmrelu_new: () => number;
  readonly browserstorage_new: () => number;
  readonly fileloader_new: () => number;
  readonly jsinterop_new: () => number;
  readonly optimizedops_new: () => number;
  readonly wasmadvancedmath_new: () => number;
  readonly wasmstatisticalfunctions_new: () => number;
  readonly wasmtotensor_new: () => number;
  readonly wasmstatisticalanalyzer_new: () => number;
  readonly wasmtransformpipeline_new: (a: number) => number;
  readonly __wbg_wasmrmsprop_free: (a: number, b: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_6: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly closure55_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure100_externref_shim: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
