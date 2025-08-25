//! ONNX format support for RusTorch
//! RusTorch向けONNX形式サポート

#[cfg(feature = "onnx")]
use crate::tensor::Tensor;
#[cfg(feature = "onnx")]
use ort::environment::Environment;
use ort::execution_providers::ExecutionProvider;
use ort::session::{Session, builder::{SessionBuilder, GraphOptimizationLevel}};
use ort::value::Value;
#[cfg(feature = "onnx")]
use std::collections::HashMap;
#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use num_traits::Float;

#[cfg(feature = "onnx")]
#[derive(Debug)]
pub enum OnnxError {
    OrtError(ort::Error),
    ConversionError(String),
    ShapeError(String),
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for OnnxError {
    fn from(error: ort::Error) -> Self {
        OnnxError::OrtError(error)
    }
}

#[cfg(feature = "onnx")]
impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::OrtError(e) => write!(f, "ONNX Runtime Error: {}", e),
            OnnxError::ConversionError(e) => write!(f, "Conversion Error: {}", e),
            OnnxError::ShapeError(e) => write!(f, "Shape Error: {}", e),
        }
    }
}

#[cfg(feature = "onnx")]
impl std::error::Error for OnnxError {}

/// ONNX model executor for RusTorch
/// RusTorch向けONNXモデル実行器
#[cfg(feature = "onnx")]
pub struct OnnxModel {
    /// ONNX Runtime session for model execution
    /// モデル実行のためのONNX Runtimeセッション
    session: Session,
    /// Names of input tensors
    /// 入力テンソルの名前
    input_names: Vec<String>,
    /// Names of output tensors
    /// 出力テンソルの名前
    output_names: Vec<String>,
}

#[cfg(feature = "onnx")]
impl OnnxModel {
    /// Load ONNX model from file
    /// ファイルからONNXモデルを読み込み
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, OnnxError> {
        let environment = Environment::builder().build()?;
        
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(4)?
            .with_model_from_file(path)?;

        // Get input and output names
        let input_names: Vec<String> = session.inputs.iter()
            .map(|input| input.name.clone())
            .collect();
        
        let output_names: Vec<String> = session.outputs.iter()
            .map(|output| output.name.clone())
            .collect();

        Ok(Self {
            session,
            input_names,
            output_names,
        })
    }

    /// Get input names
    /// 入力名を取得
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    /// 出力名を取得
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run inference with input tensors
    /// 入力テンソルで推論実行
    pub fn run<T: Float + 'static>(
        &self,
        inputs: HashMap<String, Tensor<T>>,
    ) -> Result<HashMap<String, Tensor<T>>, OnnxError> {
        // Convert RusTorch tensors to ONNX values
        let mut onnx_inputs = Vec::new();
        
        for input_name in &self.input_names {
            let tensor = inputs.get(input_name)
                .ok_or_else(|| OnnxError::ConversionError(
                    format!("Missing input tensor: {}", input_name)
                ))?;

            // Convert to f32 data
            let data: Vec<f32> = tensor.data.iter()
                .map(|&x| x.to_f32().unwrap())
                .collect();

            let shape: Vec<usize> = tensor.shape().to_vec();
            
            let onnx_value = Value::from_array((&data[..]).into())?;
            onnx_inputs.push(onnx_value);
        }

        // Run inference
        let outputs = self.session.run(onnx_inputs)?;

        // Convert ONNX outputs back to RusTorch tensors
        let mut result = HashMap::new();
        
        for (i, output_name) in self.output_names.iter().enumerate() {
            let onnx_output = &outputs[i];
            
            // Extract data and shape
            let output_tensor = self.extract_tensor_from_value::<T>(onnx_output)?;
            result.insert(output_name.clone(), output_tensor);
        }

        Ok(result)
    }

    /// Run inference with single input
    /// 単一入力で推論実行
    pub fn run_single<T: Float + 'static>(
        &self,
        input: Tensor<T>,
    ) -> Result<Tensor<T>, OnnxError> {
        if self.input_names.len() != 1 {
            return Err(OnnxError::ConversionError(
                "Model has multiple inputs, use run() instead".to_string()
            ));
        }

        let mut inputs = HashMap::new();
        inputs.insert(self.input_names[0].clone(), input);
        
        let mut outputs = self.run(inputs)?;
        
        if outputs.len() != 1 {
            return Err(OnnxError::ConversionError(
                "Model has multiple outputs".to_string()
            ));
        }

        let output_name = &self.output_names[0];
        outputs.remove(output_name)
            .ok_or_else(|| OnnxError::ConversionError("Output not found".to_string()))
    }

    fn extract_tensor_from_value<T: Float + 'static>(
        &self, 
        value: &Value
    ) -> Result<Tensor<T>, OnnxError> {
        match value {
            Value::Tensor(tensor) => {
                let shape = tensor.shape().ok_or_else(|| 
                    OnnxError::ShapeError("Cannot get tensor shape".to_string())
                )?;
                
                // Extract f32 data and convert to T
                let data_f32 = tensor.try_extract::<f32>()?;
                let data: Vec<T> = data_f32.view().iter()
                    .map(|&x| T::from(x).unwrap())
                    .collect();

                let shape_vec: Vec<usize> = shape.iter()
                    .map(|&dim| dim as usize)
                    .collect();

                Ok(Tensor::from_vec(data, shape_vec))
            },
            _ => Err(OnnxError::ConversionError(
                "Expected tensor output".to_string()
            )),
        }
    }
}

/// ONNX model builder for exporting RusTorch models
/// RusTorchモデルをエクスポートするためのONNXモデルビルダー
#[cfg(feature = "onnx")]
pub struct OnnxExporter;

#[cfg(feature = "onnx")]
impl OnnxExporter {
    /// Convert a RusTorch neural network to ONNX format
    /// RusTorchニューラルネットワークをONNX形式に変換
    pub fn export_model<T: Float + 'static, P: AsRef<Path>>(
        model: &dyn crate::nn::Module<T>,
        dummy_input: &Tensor<T>,
        path: P,
    ) -> Result<(), OnnxError> {
        // This is a placeholder for ONNX export functionality
        // Full implementation would require building ONNX graph from RusTorch operations
        
        // For now, return an informative error
        Err(OnnxError::ConversionError(
            "ONNX export not yet implemented. Use ONNX Runtime for inference only.".to_string()
        ))
    }
}

/// Utility functions for ONNX operations
/// ONNX操作のユーティリティ関数
#[cfg(feature = "onnx")]
pub mod utils {
    use super::*;

    /// Get available ONNX execution providers
    /// 利用可能なONNX実行プロバイダーを取得
    pub fn get_available_providers() -> Vec<String> {
        // TODO: Update for ort 2.0 API - this function doesn't exist anymore
        // For now return a default list of common providers
        vec![
            "CPUExecutionProvider".to_string(),
        ]
    }

    /// Create ONNX session with specific execution provider
    /// 特定の実行プロバイダーでONNXセッションを作成
    pub fn create_session_with_provider<P: AsRef<Path>>(
        model_path: P,
        provider: ExecutionProvider,
    ) -> Result<Session, OnnxError> {
        let environment = Environment::builder().build()?;
        
        let session = SessionBuilder::new(&environment)?
            .with_execution_providers([provider])?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_file(model_path)?;

        Ok(session)
    }

    /// Benchmark ONNX model inference
    /// ONNXモデル推論のベンチマーク
    pub fn benchmark_inference<T: Float + 'static>(
        model: &OnnxModel,
        inputs: HashMap<String, Tensor<T>>,
        iterations: usize,
    ) -> Result<(f64, HashMap<String, Tensor<T>>), OnnxError> {
        use std::time::Instant;

        let start = Instant::now();
        let mut result = HashMap::new();
        
        for _ in 0..iterations {
            result = model.run(inputs.clone())?;
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_secs_f64() / iterations as f64;
        
        Ok((avg_time, result))
    }
}

#[cfg(test)]
#[cfg(feature = "onnx")]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_onnx_providers() {
        let providers = utils::get_available_providers();
        println!("Available ONNX providers: {:?}", providers);
        // Should have at least CPU provider
        assert!(!providers.is_empty());
    }

    // Note: These tests require actual ONNX model files
    // In a real implementation, you would include test models
    
    #[test]
    #[ignore] // Ignore by default as it requires an ONNX model file
    fn test_load_onnx_model() {
        let model_path = PathBuf::from("test_models/simple_model.onnx");
        if model_path.exists() {
            let model = OnnxModel::from_file(&model_path);
            assert!(model.is_ok());
        }
    }

    #[test]
    #[ignore] // Ignore by default as it requires an ONNX model file
    fn test_onnx_inference() {
        let model_path = PathBuf::from("test_models/simple_model.onnx");
        if model_path.exists() {
            let model = OnnxModel::from_file(&model_path).unwrap();
            
            // Create dummy input
            let input_tensor = Tensor::<f32>::ones(vec![1, 3, 224, 224]);
            
            if model.input_names().len() == 1 {
                let result = model.run_single(input_tensor);
                assert!(result.is_ok());
            }
        }
    }
}