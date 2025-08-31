//! Integration tests for WASM modules
//! WASMモジュール統合テスト

#[cfg(feature = "wasm")]
#[cfg(test)]
mod tests {
    use rustorch::wasm::common::*;
    use rustorch::wasm::tensor::WasmTensor;

    #[test]
    fn test_pipeline_data_transforms() {
        // Test pipeline functionality with data transforms
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tensor = WasmTensor::new(data, vec![3, 3]);

        // Create normalization transform
        let normalize = rustorch::wasm::data_transforms::WasmNormalize::new(&[0.5], &[0.5])
            .expect("Failed to create normalize transform");

        // Apply transformation
        let result = normalize
            .apply(&tensor)
            .expect("Failed to apply normalization");

        // Verify result
        assert_eq!(result.shape(), vec![3, 3]);
        let result_data = result.data();
        assert_eq!(result_data.len(), 9);

        // Check normalization formula: (x - mean) / std
        for (i, &value) in result_data.iter().enumerate() {
            let expected = (tensor.data()[i] - 0.5) / 0.5;
            assert!(
                (value - expected).abs() < 1e-6,
                "Normalization failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_pipeline_multiple_transforms() {
        // Test chaining multiple transformations
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
        let tensor = WasmTensor::new(data, vec![3, 4]);

        // Create resize transform (3x4 -> 2x2)
        let resize = rustorch::wasm::data_transforms::WasmResize::new(2, 2, "nearest")
            .expect("Failed to create resize transform");

        // Apply resize
        let resized = resize.apply(&tensor).expect("Failed to apply resize");
        assert_eq!(resized.shape(), vec![2, 2]);
        assert_eq!(resized.data().len(), 4);

        // Create normalization on resized tensor
        let normalize = rustorch::wasm::data_transforms::WasmNormalize::new(&[0.5], &[0.25])
            .expect("Failed to create normalize transform");

        // Apply normalization to resized tensor
        let final_result = normalize
            .apply(&resized)
            .expect("Failed to apply normalization");
        assert_eq!(final_result.shape(), vec![2, 2]);
        assert_eq!(final_result.data().len(), 4);
    }

    #[test]
    fn test_advanced_math_integration() {
        // Test advanced math operations
        let data = vec![0.5, 1.0, 1.5, 2.0];
        let tensor = WasmTensor::new(data, vec![4]);

        let math = rustorch::wasm::advanced_math::WasmAdvancedMath::new();

        // Test multiple operations
        let sinh_result = math.sinh(&tensor).expect("sinh failed");
        let cosh_result = math.cosh(&tensor).expect("cosh failed");
        let tanh_result = math.tanh(&tensor).expect("tanh failed");

        assert_eq!(sinh_result.data().len(), 4);
        assert_eq!(cosh_result.data().len(), 4);
        assert_eq!(tanh_result.data().len(), 4);

        // Verify mathematical relationship: tanh(x) = sinh(x) / cosh(x)
        for i in 0..4 {
            let expected_tanh = sinh_result.data()[i] / cosh_result.data()[i];
            let actual_tanh = tanh_result.data()[i];
            assert!(
                (expected_tanh - actual_tanh).abs() < 1e-6,
                "tanh relationship failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_quality_metrics_integration() {
        // Test quality metrics with real data
        let clean_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dirty_data = vec![1.0, f32::NAN, 3.0, f32::INFINITY, 5.0];

        let clean_tensor = WasmTensor::new(clean_data, vec![5]);
        let dirty_tensor = WasmTensor::new(dirty_data, vec![5]);

        let metrics = rustorch::wasm::quality_metrics::WasmQualityMetrics::new(0.8)
            .expect("Failed to create quality metrics");

        // Test completeness
        let clean_completeness = metrics
            .completeness(&clean_tensor)
            .expect("Failed to calculate clean completeness");
        let dirty_completeness = metrics
            .completeness(&dirty_tensor)
            .expect("Failed to calculate dirty completeness");

        assert!(
            (clean_completeness - 100.0).abs() < 1e-6,
            "Clean data should be 100% complete"
        );
        assert!(
            dirty_completeness >= 60.0 && dirty_completeness <= 80.0,
            "Dirty data completeness should be 60-80%, got {:.2}%",
            dirty_completeness
        );

        // Test validity
        let clean_validity = metrics
            .validity(&clean_tensor)
            .expect("Failed to calculate clean validity");
        let dirty_validity = metrics
            .validity(&dirty_tensor)
            .expect("Failed to calculate dirty validity");

        assert!(
            (clean_validity - 100.0).abs() < 1e-6,
            "Clean data should be 100% valid"
        );
        assert!(
            dirty_validity >= 60.0 && dirty_validity <= 80.0,
            "Dirty data validity should be 60-80%, got {:.2}%",
            dirty_validity
        );
    }

    #[test]
    fn test_anomaly_detection_integration() {
        // Test anomaly detection with statistical functions
        let normal_data = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98];
        let data_with_anomaly = vec![1.0, 1.1, 0.9, 10.0, 0.95, 1.02, 0.98]; // 10.0 is anomaly

        let normal_tensor = WasmTensor::new(normal_data, vec![7]);
        let anomaly_tensor = WasmTensor::new(data_with_anomaly, vec![7]);

        let mut detector = rustorch::wasm::anomaly_detection::WasmAnomalyDetector::new(2.0, 10)
            .expect("Failed to create anomaly detector");

        // Test normal data (should find no anomalies)
        let normal_anomalies = detector
            .detect_statistical(&normal_tensor)
            .expect("Failed to detect on normal data");
        assert_eq!(
            normal_anomalies.length(),
            0,
            "Normal data should have no anomalies"
        );

        // Test data with anomaly (should find 1 anomaly)
        let found_anomalies = detector
            .detect_statistical(&anomaly_tensor)
            .expect("Failed to detect on anomaly data");
        assert!(
            found_anomalies.length() >= 1,
            "Should detect at least 1 anomaly, found {}",
            found_anomalies.length()
        );
    }

    #[test]
    fn test_trait_polymorphism() {
        // Test trait polymorphism works correctly
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = WasmTensor::new(data, vec![4]);

        // Create different analyzers
        let stats_analyzer = rustorch::wasm::quality_metrics::WasmStatisticalAnalyzer::new();
        let quality_analyzer = rustorch::wasm::quality_metrics::WasmQualityMetrics::new(0.8)
            .expect("Failed to create quality analyzer");

        // Test WasmAnalyzer trait
        let stats_result = stats_analyzer
            .analyze(&tensor)
            .expect("Stats analysis failed");
        let quality_result = quality_analyzer
            .analyze(&tensor)
            .expect("Quality analysis failed");

        // Verify both return valid JSON
        assert!(
            stats_result.contains("mean"),
            "Stats result should contain mean"
        );
        assert!(
            quality_result.contains("completeness"),
            "Quality result should contain completeness"
        );

        // Test analysis types
        assert_eq!(stats_analyzer.analysis_type(), "statistical_analysis");
        assert_eq!(quality_analyzer.analysis_type(), "quality_assessment");
    }

    #[test]
    fn test_memory_efficiency() {
        // Test memory management efficiency
        let large_data: Vec<f32> = (0..10000).map(|x| x as f32).collect();
        let tensor = WasmTensor::new(large_data, vec![100, 100]);

        let math = rustorch::wasm::advanced_math::WasmAdvancedMath::new();

        // Perform multiple operations to test memory reuse
        let result1 = math.sinh(&tensor).expect("sinh failed");
        let result2 = math.cosh(&tensor).expect("cosh failed");
        let result3 = math.tanh(&tensor).expect("tanh failed");

        // Verify all results are correct size
        assert_eq!(result1.data().len(), 10000);
        assert_eq!(result2.data().len(), 10000);
        assert_eq!(result3.data().len(), 10000);

        // Test large transformation
        let normalize = rustorch::wasm::data_transforms::WasmNormalize::new(&[50.0], &[25.0])
            .expect("Failed to create normalize");

        let normalized = normalize.apply(&tensor).expect("Normalization failed");
        assert_eq!(normalized.data().len(), 10000);
    }

    #[test]
    fn test_error_handling_consistency() {
        // Test consistent error handling across modules
        let empty_data: Vec<f32> = vec![];
        let empty_tensor = WasmTensor::new(empty_data, vec![0]);

        let math = rustorch::wasm::advanced_math::WasmAdvancedMath::new();
        let quality = rustorch::wasm::quality_metrics::WasmQualityMetrics::new(0.8)
            .expect("Failed to create quality metrics");

        // All operations should fail consistently on empty tensor
        assert!(math.sinh(&empty_tensor).is_err());
        assert!(quality.completeness(&empty_tensor).is_err());

        // Test invalid parameters
        let valid_data = vec![1.0, 2.0, 3.0];
        let valid_tensor = WasmTensor::new(valid_data, vec![3]);

        // Invalid range should fail
        assert!(math.clamp(&valid_tensor, 5.0, 1.0).is_err()); // min > max
        assert!(quality.accuracy(&valid_tensor, 10.0, 5.0).is_err()); // min > max
    }
}
