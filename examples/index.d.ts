// TypeScript definitions for RusTorch WASM
// RusTorch WASM用TypeScript型定義

/**
 * Core tensor data structure
 */
export class WasmTensor {
  constructor(data: Float32Array | number[], shape: number[]);
  data(): Float32Array;
  shape(): number[];
  free(): void;
}

/**
 * Advanced mathematical operations
 */
export class WasmAdvancedMath {
  constructor();
  
  // Hyperbolic functions
  sinh(tensor: WasmTensor): WasmTensor;
  cosh(tensor: WasmTensor): WasmTensor;
  tanh(tensor: WasmTensor): WasmTensor;
  
  // Inverse trigonometric functions
  asin(tensor: WasmTensor): WasmTensor;
  acos(tensor: WasmTensor): WasmTensor;
  atan(tensor: WasmTensor): WasmTensor;
  atan2(y: WasmTensor, x: WasmTensor): WasmTensor;
  
  // Special functions
  erf(tensor: WasmTensor): WasmTensor;
  erfc(tensor: WasmTensor): WasmTensor;
  gamma(tensor: WasmTensor): WasmTensor;
  lgamma(tensor: WasmTensor): WasmTensor;
  
  // Utility functions
  clamp(tensor: WasmTensor, min: number, max: number): WasmTensor;
  sign(tensor: WasmTensor): WasmTensor;
  lerp(start: WasmTensor, end: WasmTensor, weight: number): WasmTensor;
  pow(base: WasmTensor, exponent: number): WasmTensor;
  pow_tensor(base: WasmTensor, exponent: WasmTensor): WasmTensor;
  
  // Rounding functions
  round(tensor: WasmTensor): WasmTensor;
  floor(tensor: WasmTensor): WasmTensor;
  ceil(tensor: WasmTensor): WasmTensor;
  trunc(tensor: WasmTensor): WasmTensor;
  
  // Validation functions
  is_finite(tensor: WasmTensor): WasmTensor;
  is_infinite(tensor: WasmTensor): WasmTensor;
  is_nan(tensor: WasmTensor): WasmTensor;
  
  free(): void;
}

/**
 * Statistical functions
 */
export class WasmStatisticalFunctions {
  constructor();
  
  correlation(x: WasmTensor, y: WasmTensor): number;
  covariance(x: WasmTensor, y: WasmTensor): number;
  percentile(tensor: WasmTensor, percentile: number): number;
  quantiles(tensor: WasmTensor, quantiles: number[]): number[];
  
  free(): void;
}

/**
 * Data quality metrics
 */
export class WasmQualityMetrics {
  constructor(threshold: number);
  
  completeness(tensor: WasmTensor): number;
  accuracy(tensor: WasmTensor, min: number, max: number): number;
  consistency(tensor: WasmTensor): number;
  validity(tensor: WasmTensor): number;
  uniqueness(tensor: WasmTensor): number;
  overall_quality(tensor: WasmTensor): number;
  quality_report(tensor: WasmTensor): string;
  
  free(): void;
}

/**
 * Statistical analyzer
 */
export class WasmStatisticalAnalyzer {
  constructor();
  
  basic_stats(tensor: WasmTensor): string;
  percentiles(tensor: WasmTensor, percentiles: number[]): number[];
  detect_outliers(tensor: WasmTensor): any[];
  
  free(): void;
}

/**
 * Anomaly detection
 */
export class WasmAnomalyDetector {
  constructor(threshold: number, window_size: number);
  
  detect_statistical(tensor: WasmTensor): any[];
  detect_isolation_forest(tensor: WasmTensor, n_trees: number): any[];
  detect_realtime(value: number): any | null;
  get_statistics(): string;
  reset(): void;
  set_threshold(threshold: number): void;
  get_threshold(): number;
  
  free(): void;
}

/**
 * Time series anomaly detection
 */
export class WasmTimeSeriesDetector {
  constructor(window_size: number, seasonal_period?: number);
  
  add_point(timestamp: number, value: number): any | null;
  get_trend_analysis(): string;
  get_seasonal_analysis(): string;
  
  free(): void;
}

/**
 * Data transformations
 */
export class WasmNormalize {
  constructor(mean: number[], std: number[]);
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

export class WasmResize {
  constructor(height: number, width: number, interpolation: string);
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

export class WasmCenterCrop {
  constructor(height: number, width: number);
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

export class WasmRandomCrop {
  constructor(height: number, width: number, padding?: number);
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

export class WasmColorJitter {
  constructor(brightness: number, contrast: number, saturation: number, hue: number);
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

export class WasmToTensor {
  constructor();
  apply(tensor: WasmTensor): WasmTensor;
  name(): string;
  free(): void;
}

/**
 * Pipeline system
 */
export class WasmTransformPipeline {
  constructor(cache_enabled: boolean);
  add_transform(transform_name: string): void;
  length(): number;
  clear(): void;
  execute(input: WasmTensor): WasmTensor;
  get_stats(): string;
  free(): void;
}

export class WasmProcessingPipeline {
  constructor(parallel_execution: boolean);
  add_operation(operation_name: string): void;
  operation_count(): number;
  get_config(): string;
  free(): void;
}

/**
 * Utility functions
 */
export function wasm_advanced_math_version(): string;
export function wasm_quality_metrics_version(): string;
export function wasm_transforms_version(): string;
export function wasm_anomaly_detection_version(): string;

export function create_quality_analyzer(): WasmQualityMetrics;
export function quick_quality_assessment(tensor: WasmTensor): string;
export function create_imagenet_preprocessing(): WasmNormalize;
export function create_cifar_preprocessing(): WasmNormalize;
export function create_simple_detector(threshold: number): WasmAnomalyDetector;
export function create_streaming_detector(window_size: number): WasmTimeSeriesDetector;
export function detect_anomalies_batch(data: number[], threshold: number): any[];

/**
 * Error types
 */
export interface WasmError {
  message: string;
  error_type: string;
  context?: string;
}

/**
 * Result type for operations that may fail
 */
export type WasmResult<T> = T | WasmError;