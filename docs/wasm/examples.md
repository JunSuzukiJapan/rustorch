# RusTorch WASM Examples

実践的なWASM使用例集

## 🎯 Basic Examples

### 1. Multi-Layer Perceptron with Linear Layers

```javascript
import init, * as rustorch from './pkg/rustorch.js';

class SimpleMLP {
    constructor() {
        // Create layers
        this.linear1 = new rustorch.WasmLinear(784, 128, true);  // Input layer
        this.linear2 = new rustorch.WasmLinear(128, 64, true);   // Hidden layer
        this.linear3 = new rustorch.WasmLinear(64, 10, true);    // Output layer
    }
    
    forward(input, batch_size) {
        // Forward pass through MLP
        let x = this.linear1.forward(input, batch_size);
        x = rustorch.relu(x);
        
        x = this.linear2.forward(x, batch_size);
        x = rustorch.relu(x);
        
        x = this.linear3.forward(x, batch_size);
        return rustorch.softmax(x);
    }
    
    getParameters() {
        return {
            layer1_weights: this.linear1.get_weights(),
            layer1_bias: this.linear1.get_bias(),
            layer2_weights: this.linear2.get_weights(),
            layer2_bias: this.linear2.get_bias(),
            layer3_weights: this.linear3.get_weights(),
            layer3_bias: this.linear3.get_bias(),
        };
    }
}

async function mlpExample() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    const model = new SimpleMLP();
    
    // MNIST-like input (28x28 flattened)
    const input = new Array(784).fill(0).map(() => Math.random());
    const batch_size = 1;
    
    // Forward pass
    const predictions = model.forward(input, batch_size);
    console.log('MLP predictions:', predictions);
    
    // Get model parameters for training
    const params = model.getParameters();
    console.log('Model has', params.layer1_weights.length + params.layer2_weights.length + params.layer3_weights.length, 'parameters');
}
```

### 2. Convolutional Neural Network for Image Classification

```javascript
class SimpleCNN {
    constructor() {
        // Convolutional layers
        this.conv1 = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);   // 3→32 channels
        this.conv2 = new rustorch.WasmConv2d(32, 64, 3, 2, 1, true);  // 32→64 channels, stride=2
        this.conv3 = new rustorch.WasmConv2d(64, 128, 3, 2, 1, true); // 64→128 channels, stride=2
        
        // Classifier (assuming 32x32 input → 8x8 after convolutions)
        this.classifier = new rustorch.WasmLinear(128 * 8 * 8, 10, true);
    }
    
    forward(image_data, batch_size, height, width) {
        // Convolutional feature extraction
        let x = this.conv1.forward(image_data, batch_size, height, width);
        x = rustorch.relu(x);
        
        // Calculate new dimensions after conv1 (same size due to padding=1, stride=1)
        let [channels1, h1, w1] = this.conv1.output_shape(height, width);
        
        x = this.conv2.forward(x, batch_size, h1, w1);
        x = rustorch.relu(x);
        
        let [channels2, h2, w2] = this.conv2.output_shape(h1, w1);
        
        x = this.conv3.forward(x, batch_size, h2, w2);
        x = rustorch.relu(x);
        
        let [channels3, h3, w3] = this.conv3.output_shape(h2, w2);
        
        // Flatten for classifier
        const flattened_size = channels3 * h3 * w3;
        
        // Classification
        const logits = this.classifier.forward(x, batch_size);
        return rustorch.softmax(logits);
    }
}

async function cnnExample() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    const model = new SimpleCNN();
    
    // RGB image (32x32x3)
    const image_data = new Array(32 * 32 * 3).fill(0).map(() => Math.random());
    
    // Forward pass
    const predictions = model.forward(image_data, 1, 32, 32);
    console.log('CNN predictions:', predictions);
    
    // Show model architecture
    console.log('Conv1 output shape for 32x32 input:', model.conv1.output_shape(32, 32));
    console.log('Conv2 output shape:', model.conv2.output_shape(32, 32));
    console.log('Conv3 output shape:', model.conv3.output_shape(16, 16));
}
```

### 3. Complete Image Processing Pipeline

```javascript
class ImageProcessor {
    constructor() {
        this.imagenet_mean = [0.485, 0.456, 0.406];
        this.imagenet_std = [0.229, 0.224, 0.225];
    }
    
    async preprocessImage(imageData, originalHeight, originalWidth) {
        await init();
        rustorch.initialize_wasm_runtime();
        
        // Convert uint8 to float32
        const floatImage = rustorch.WasmVision.to_float(imageData);
        
        // Resize to 256x256
        const resized = rustorch.WasmVision.resize(
            floatImage, originalHeight, originalWidth, 256, 256, 3
        );
        
        // Center crop to 224x224
        const cropped = rustorch.WasmVision.center_crop(resized, 256, 256, 3, 224);
        
        // Normalize with ImageNet statistics
        const normalized = rustorch.WasmVision.normalize(
            cropped, this.imagenet_mean, this.imagenet_std, 3
        );
        
        return normalized;
    }
    
    async augmentImage(imageData, height, width) {
        // Data augmentation pipeline
        let augmented = imageData;
        
        // Random horizontal flip (50% chance)
        if (Math.random() > 0.5) {
            augmented = rustorch.WasmVision.flip_horizontal(augmented, height, width, 3);
        }
        
        // Random rotation (-15 to +15 degrees)
        augmented = rustorch.WasmVision.random_rotation(augmented, height, width, 3, 15);
        
        // Adjust brightness randomly
        const brightness_factor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
        augmented = rustorch.WasmVision.adjust_brightness(augmented, brightness_factor - 1.0);
        
        // Add slight Gaussian noise
        augmented = rustorch.WasmVision.add_gaussian_noise(augmented, 0.02);
        
        return augmented;
    }
    
    async detectEdges(imageData, height, width) {
        // Convert to grayscale first
        const grayscale = rustorch.WasmVision.rgb_to_grayscale(imageData, height, width);
        
        // Apply Gaussian blur to reduce noise
        const blurred = rustorch.WasmVision.gaussian_blur(grayscale, height, width, 1, 1.0);
        
        // Edge detection
        const edges = rustorch.WasmVision.edge_detection(blurred, height, width);
        
        return edges;
    }
}

async function imageProcessingExample() {
    const processor = new ImageProcessor();
    
    // Simulate loading an image (normally from canvas/file)
    const width = 300, height = 300;
    const imageData = new Uint8Array(width * height * 3);
    for (let i = 0; i < imageData.length; i++) {
        imageData[i] = Math.floor(Math.random() * 256);
    }
    
    // Preprocess for ML model
    const preprocessed = await processor.preprocessImage(Array.from(imageData), height, width);
    console.log('Preprocessed image shape:', preprocessed.length); // Should be 224*224*3
    
    // Apply data augmentation
    const floatImage = rustorch.WasmVision.to_float(Array.from(imageData));
    const augmented = await processor.augmentImage(floatImage, height, width);
    console.log('Augmented image ready for training');
    
    // Edge detection
    const edges = await processor.detectEdges(floatImage, height, width);
    console.log('Edge detection completed');
    
    // Convert back to displayable format
    const displayImage = rustorch.WasmVision.to_uint8(edges.map(x => x * 255));
    console.log('Ready for display');
}
```

### 4. Data Classification Pipeline

```javascript
async function classificationPipeline() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    // Sample dataset
    const features = [
        1.2, 0.8, -0.5, 2.1,  // Sample 1
        0.5, 1.5, 0.3, -0.8,  // Sample 2  
        -1.0, 0.2, 1.8, 0.6,  // Sample 3
        2.0, -0.5, 0.9, 1.2   // Sample 4
    ];
    const targets = [0, 1, 1, 0]; // Binary classification
    
    // Preprocessing
    const preprocessor = new rustorch.WasmPreprocessor();
    const stats = preprocessor.compute_stats(features);
    const normalized = preprocessor.z_score_normalize(features, stats[0], stats[1]);
    
    // One-hot encode targets
    const one_hot_targets = preprocessor.one_hot_encode(targets, 2);
    
    // Train-test split
    const split = preprocessor.train_test_split(
        normalized, targets, 4, 0.25, 42
    );
    
    console.log('Dataset split:', {
        train_samples: split.trainFeatures.length / 4,
        test_samples: split.testFeatures.length / 4
    });
    
    // Simple prediction (placeholder weights)
    const weights = [0.1, 0.2, 0.3, 0.4];
    const predictions = [];
    
    for (let i = 0; i < split.testFeatures.length; i += 4) {
        const sample = split.testFeatures.slice(i, i + 4);
        const score = rustorch.WasmTensorOps.dot_product(sample, weights);
        predictions.push(score > 0.5 ? 1 : 0);
    }
    
    // Evaluate
    const accuracy = rustorch.WasmMetrics.accuracy(predictions, split.testTargets);
    console.log('Test accuracy:', accuracy);
}
```

## 🔬 Advanced Examples

### 3. Real-time Signal Processing

```javascript
class AudioMLProcessor {
    constructor() {
        this.sampleRate = 44100;
        this.bufferSize = 1024;
        this.processor = new rustorch.WasmPreprocessor();
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        
        // Setup audio context
        this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
        this.processor_node = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
        
        this.processor_node.onaudioprocess = (event) => {
            this.processAudioFrame(event);
        };
    }
    
    processAudioFrame(audioEvent) {
        const inputData = audioEvent.inputBuffer.getChannelData(0);
        const samples = Array.from(inputData);
        
        // Apply windowing
        const windowed = rustorch.apply_hann_window(samples);
        
        // Compute FFT for frequency analysis
        const fft_result = rustorch.dft(windowed);
        const magnitude = rustorch.compute_magnitude(fft_result);
        
        // Feature extraction
        const features = this.processor.min_max_normalize(magnitude, 0.0, 1.0);
        
        // ML classification (voice activity detection example)
        const voice_score = rustorch.WasmTensorOps.dot_product(
            features.slice(0, 100), // First 100 frequency bins
            new Array(100).fill(1.0) // Simple weights
        );
        
        const is_voice = voice_score > 50.0;
        
        // Output to console or UI
        if (is_voice) {
            console.log('Voice detected, score:', voice_score.toFixed(2));
        }
        
        // Pass through audio (or modify based on ML output)
        const outputData = audioEvent.outputBuffer.getChannelData(0);
        outputData.set(inputData);
    }
    
    async startProcessing() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = this.audioContext.createMediaStreamSource(stream);
        source.connect(this.processor_node);
        this.processor_node.connect(this.audioContext.destination);
    }
}

// Usage
const audioML = new AudioMLProcessor();
audioML.initialize().then(() => {
    document.getElementById('start-btn').onclick = () => audioML.startProcessing();
});
```

### 4. Computer Vision in Browser

```javascript
class VisionMLProcessor {
    constructor() {
        this.model_initialized = false;
        this.preprocessor = null;
        this.batchNorm = null;
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        
        this.preprocessor = new rustorch.WasmPreprocessor();
        this.batchNorm = new rustorch.WasmBatchNorm(224*224*3, 0.1, 1e-5);
        this.batchNorm.set_training(false); // Inference mode
        
        this.model_initialized = true;
    }
    
    async processImage(imageElement) {
        if (!this.model_initialized) {
            throw new Error('Model not initialized');
        }
        
        // Create canvas for image processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 224;
        canvas.height = 224;
        
        // Draw and resize image
        ctx.drawImage(imageElement, 0, 0, 224, 224);
        const imageData = ctx.getImageData(0, 0, 224, 224);
        
        // Convert to RGB array
        const pixels = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            pixels.push(imageData.data[i] / 255.0);     // R
            pixels.push(imageData.data[i + 1] / 255.0); // G  
            pixels.push(imageData.data[i + 2] / 255.0); // B
        }
        
        // Preprocessing
        const stats = this.preprocessor.compute_stats(pixels);
        const normalized = this.preprocessor.z_score_normalize(
            pixels, stats[0], stats[1]
        );
        
        // Batch normalization
        const bn_output = this.batchNorm.forward(normalized, 1);
        
        // Feature extraction (simplified CNN-like operations)
        const features = this.extractFeatures(bn_output);
        
        // Classification
        return this.classify(features);
    }
    
    extractFeatures(pixels) {
        // Simple feature extraction using convolution-like operations
        const features = [];
        const stride = 7; // Simplified stride for feature extraction
        
        for (let y = 0; y < 224 - stride; y += stride) {
            for (let x = 0; x < 224 - stride; x += stride) {
                let patch_sum = 0;
                for (let dy = 0; dy < stride; dy++) {
                    for (let dx = 0; dx < stride; dx++) {
                        const idx = ((y + dy) * 224 + (x + dx)) * 3; // RGB
                        patch_sum += pixels[idx] + pixels[idx + 1] + pixels[idx + 2];
                    }
                }
                features.push(patch_sum / (stride * stride * 3));
            }
        }
        
        return rustorch.relu(features);
    }
    
    classify(features) {
        // Simple classification weights (normally loaded from trained model)
        const class_weights = [
            new Array(features.length).fill(0.01),  // Class 0 weights
            new Array(features.length).fill(0.02),  // Class 1 weights  
            new Array(features.length).fill(0.015)  // Class 2 weights
        ];
        
        const scores = class_weights.map(weights => 
            rustorch.WasmTensorOps.dot_product(features, weights)
        );
        
        const probabilities = rustorch.softmax(scores);
        const predicted_class = probabilities.indexOf(Math.max(...probabilities));
        
        return {
            predicted_class,
            confidence: probabilities[predicted_class],
            all_probabilities: probabilities
        };
    }
}

// Usage with file input
document.getElementById('image-input').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const img = new Image();
    img.onload = async () => {
        const processor = new VisionMLProcessor();
        await processor.initialize();
        
        const result = await processor.processImage(img);
        
        document.getElementById('result').innerHTML = `
            <h3>Classification Result</h3>
            <p>Predicted Class: ${result.predicted_class}</p>
            <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
            <p>All Probabilities: ${result.all_probabilities.map(p => (p * 100).toFixed(2) + '%').join(', ')}</p>
        `;
    };
    
    img.src = URL.createObjectURL(file);
});
```

### 5. Time Series Analysis

```javascript
class TimeSeriesML {
    constructor(sequence_length = 50) {
        this.sequence_length = sequence_length;
        this.preprocessor = null;
        this.model_weights = null;
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        
        this.preprocessor = new rustorch.WasmPreprocessor();
        
        // Initialize simple LSTM-like weights
        this.model_weights = {
            input: new Array(this.sequence_length * 10).fill(0).map(() => (Math.random() - 0.5) * 0.2),
            hidden: new Array(10 * 10).fill(0).map(() => (Math.random() - 0.5) * 0.2),
            output: new Array(10).fill(0).map(() => (Math.random() - 0.5) * 0.2)
        };
    }
    
    async predict(timeseries_data) {
        // Normalize time series
        const stats = this.preprocessor.compute_stats(timeseries_data);
        const normalized = this.preprocessor.z_score_normalize(
            timeseries_data, stats[0], stats[1]
        );
        
        // Create sliding windows
        const sequences = [];
        for (let i = 0; i <= normalized.length - this.sequence_length; i++) {
            sequences.push(normalized.slice(i, i + this.sequence_length));
        }
        
        // Process each sequence
        const predictions = [];
        for (const sequence of sequences) {
            // Simple recurrent processing
            let hidden_state = new Array(10).fill(0);
            
            for (let t = 0; t < sequence.length; t++) {
                // Input processing
                const input_contribution = sequence[t] * this.model_weights.input[t % this.model_weights.input.length];
                
                // Hidden state update (simplified)
                hidden_state = hidden_state.map((h, i) => {
                    const weight_idx = i * 10 + (t % 10);
                    return rustorch.tanh([h * this.model_weights.hidden[weight_idx] + input_contribution])[0];
                });
            }
            
            // Output prediction
            const prediction = rustorch.WasmTensorOps.dot_product(
                hidden_state, 
                this.model_weights.output
            );
            predictions.push(prediction);
        }
        
        return predictions;
    }
    
    async detectAnomalies(data, threshold = 2.0) {
        const predictions = await this.predict(data);
        
        // Calculate prediction errors
        const actual_next_values = data.slice(this.sequence_length);
        const errors = predictions.map((pred, i) => 
            Math.abs(pred - actual_next_values[i])
        );
        
        // Statistical anomaly detection
        const error_stats = this.preprocessor.compute_stats(errors);
        const anomaly_threshold = error_stats[0] + threshold * error_stats[1]; // mean + 2*std
        
        const anomalies = errors.map((error, i) => ({
            index: i + this.sequence_length,
            error: error,
            is_anomaly: error > anomaly_threshold,
            severity: error / anomaly_threshold
        }));
        
        return {
            predictions,
            errors,
            anomalies: anomalies.filter(a => a.is_anomaly),
            threshold: anomaly_threshold
        };
    }
}

// Usage
async function runTimeSeriesAnalysis() {
    const tsml = new TimeSeriesML(20);
    await tsml.initialize();
    
    // Generate sample time series data
    const data = [];
    for (let i = 0; i < 200; i++) {
        // Sine wave with noise and occasional spikes (anomalies)
        let value = Math.sin(i * 0.1) + Math.random() * 0.1;
        
        // Add anomalies at specific points
        if (i === 50 || i === 150) {
            value += 3.0; // Anomaly spike
        }
        
        data.push(value);
    }
    
    const analysis = await tsml.detectAnomalies(data, 2.0);
    
    console.log('Anomalies detected:', analysis.anomalies.length);
    analysis.anomalies.forEach(anomaly => {
        console.log(`Anomaly at index ${anomaly.index}: severity ${anomaly.severity.toFixed(2)}`);
    });
}
```

## 🧮 Mathematical Computing

### 6. Statistical Analysis

```javascript
async function statisticalAnalysis() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    // Generate sample data from different distributions
    const normal_dist = new rustorch.WasmNormalDistribution(0.0, 1.0, 42);
    const uniform_dist = new rustorch.WasmUniformDistribution(0.0, 1.0, 43);
    
    const normal_samples = [];
    const uniform_samples = [];
    
    for (let i = 0; i < 1000; i++) {
        normal_samples.push(normal_dist.sample());
        uniform_samples.push(uniform_dist.sample());
    }
    
    // Compute statistics
    const normal_stats = rustorch.WasmPreprocessor.compute_stats(normal_samples);
    const uniform_stats = rustorch.WasmPreprocessor.compute_stats(uniform_samples);
    
    console.log('Normal Distribution Stats:', {
        mean: normal_stats[0].toFixed(3),
        std: normal_stats[1].toFixed(3),
        min: normal_stats[2].toFixed(3),
        max: normal_stats[3].toFixed(3)
    });
    
    console.log('Uniform Distribution Stats:', {
        mean: uniform_stats[0].toFixed(3),
        std: uniform_stats[1].toFixed(3), 
        min: uniform_stats[2].toFixed(3),
        max: uniform_stats[3].toFixed(3)
    });
    
    // Statistical tests
    const correlation = rustorch.cross_correlation(
        normal_samples.slice(0, 100),
        uniform_samples.slice(0, 100)
    );
    
    console.log('Cross-correlation peak:', Math.max(...correlation));
}
```

### 7. Signal Filtering and Analysis

```javascript
class SignalProcessor {
    constructor() {
        this.preprocessor = null;
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        this.preprocessor = new rustorch.WasmPreprocessor();
    }
    
    generateTestSignal(length = 512, frequency = 5, noise_level = 0.1) {
        const signal = [];
        for (let i = 0; i < length; i++) {
            const t = i / length;
            const clean_signal = Math.sin(2 * Math.PI * frequency * t);
            const noise = (Math.random() - 0.5) * 2 * noise_level;
            signal.push(clean_signal + noise);
        }
        return signal;
    }
    
    async analyzeSignal(signal) {
        // Apply windowing to reduce spectral leakage
        const windowed = rustorch.apply_hann_window(signal);
        
        // Compute FFT
        const fft_result = rustorch.dft(windowed);
        const magnitude = rustorch.compute_magnitude(fft_result);
        
        // Find peak frequency
        const peak_index = magnitude.indexOf(Math.max(...magnitude));
        const peak_frequency = peak_index * (44100 / 2) / magnitude.length;
        
        // Compute signal statistics
        const signal_stats = this.preprocessor.compute_stats(signal);
        const magnitude_stats = this.preprocessor.compute_stats(magnitude);
        
        return {
            peak_frequency,
            signal_power: signal_stats[1], // Use std as power measure
            spectral_centroid: this.computeSpectralCentroid(magnitude),
            spectral_rolloff: this.computeSpectralRolloff(magnitude, 0.85)
        };
    }
    
    computeSpectralCentroid(magnitude) {
        let weighted_sum = 0;
        let total_magnitude = 0;
        
        for (let i = 0; i < magnitude.length; i++) {
            weighted_sum += i * magnitude[i];
            total_magnitude += magnitude[i];
        }
        
        return total_magnitude > 0 ? weighted_sum / total_magnitude : 0;
    }
    
    computeSpectralRolloff(magnitude, rolloff_percent) {
        const total_energy = magnitude.reduce((sum, val) => sum + val * val, 0);
        const threshold = total_energy * rolloff_percent;
        
        let cumulative_energy = 0;
        for (let i = 0; i < magnitude.length; i++) {
            cumulative_energy += magnitude[i] * magnitude[i];
            if (cumulative_energy >= threshold) {
                return i;
            }
        }
        
        return magnitude.length - 1;
    }
}

// Usage
async function audioFeatureExtraction() {
    const processor = new SignalProcessor();
    await processor.initialize();
    
    // Generate test signals
    const clean_signal = processor.generateTestSignal(512, 5, 0.05);
    const noisy_signal = processor.generateTestSignal(512, 5, 0.3);
    
    const clean_analysis = await processor.analyzeSignal(clean_signal);
    const noisy_analysis = await processor.analyzeSignal(noisy_signal);
    
    console.log('Clean Signal Analysis:', clean_analysis);
    console.log('Noisy Signal Analysis:', noisy_analysis);
    
    // Compare signal quality
    const snr_improvement = clean_analysis.signal_power / noisy_analysis.signal_power;
    console.log('Signal quality ratio:', snr_improvement.toFixed(2));
}
```

## 🎮 Interactive ML Demos

### 8. Live Training Visualization

```javascript
class LiveTrainingDemo {
    constructor() {
        this.optimizer = null;
        this.loss_history = [];
        this.accuracy_history = [];
        this.epoch = 0;
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        
        this.optimizer = new rustorch.WasmOptimizer('adam', 0.01);
        this.preprocessor = new rustorch.WasmPreprocessor();
        this.metrics = new rustorch.WasmMetrics();
        
        // Setup UI
        this.setupChart();
    }
    
    setupChart() {
        this.canvas = document.getElementById('training-chart');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 400;
    }
    
    async trainStep(features, targets) {
        // Forward pass
        const batch_size = targets.length;
        const feature_size = features.length / batch_size;
        
        // Simple linear model: y = Wx + b
        const weights = this.optimizer.get_parameters('weights') || 
                       new Array(feature_size).fill(0.1);
        const bias = this.optimizer.get_parameters('bias') || [0.0];
        
        // Predictions
        const predictions = [];
        for (let i = 0; i < batch_size; i++) {
            const sample = features.slice(i * feature_size, (i + 1) * feature_size);
            const prediction = rustorch.WasmTensorOps.dot_product(sample, weights) + bias[0];
            predictions.push(prediction > 0.5 ? 1.0 : 0.0);
        }
        
        // Loss calculation
        const loss = rustorch.mse_loss(predictions, targets);
        
        // Gradient computation (simplified)
        const weight_gradients = new Array(feature_size).fill(0);
        let bias_gradient = 0;
        
        for (let i = 0; i < batch_size; i++) {
            const sample = features.slice(i * feature_size, (i + 1) * feature_size);
            const error = predictions[i] - targets[i];
            
            for (let j = 0; j < feature_size; j++) {
                weight_gradients[j] += error * sample[j] / batch_size;
            }
            bias_gradient += error / batch_size;
        }
        
        // Optimization step
        const updated_weights = this.optimizer.step('weights', weights, weight_gradients);
        const updated_bias = this.optimizer.step('bias', bias, [bias_gradient]);
        
        // Metrics
        const pred_classes = predictions.map(p => p > 0.5 ? 1 : 0);
        const target_classes = targets.map(t => Math.round(t));
        const accuracy = this.metrics.accuracy(pred_classes, target_classes);
        
        // Update history
        this.loss_history.push(loss);
        this.accuracy_history.push(accuracy);
        this.epoch++;
        
        // Update visualization
        this.updateChart();
        
        return { loss, accuracy, epoch: this.epoch };
    }
    
    updateChart() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw axes
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(50, height - 50);
        ctx.lineTo(width - 50, height - 50); // X-axis
        ctx.moveTo(50, 50);
        ctx.lineTo(50, height - 50); // Y-axis
        ctx.stroke();
        
        // Draw loss curve (red)
        if (this.loss_history.length > 1) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const max_loss = Math.max(...this.loss_history);
            
            this.loss_history.forEach((loss, i) => {
                const x = 50 + (i / Math.max(this.loss_history.length - 1, 1)) * (width - 100);
                const y = height - 50 - (loss / max_loss) * (height - 100);
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw accuracy curve (blue)
        if (this.accuracy_history.length > 1) {
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            this.accuracy_history.forEach((acc, i) => {
                const x = 50 + (i / Math.max(this.accuracy_history.length - 1, 1)) * (width - 100);
                const y = height - 50 - acc * (height - 100);
                
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Labels
        ctx.fillStyle = 'black';
        ctx.font = '14px Arial';
        ctx.fillText('Epochs', width / 2, height - 10);
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Loss / Accuracy', 0, 0);
        ctx.restore();
        
        // Legend
        ctx.fillStyle = 'red';
        ctx.fillText('Loss', width - 100, 30);
        ctx.fillStyle = 'blue';
        ctx.fillText('Accuracy', width - 100, 50);
    }
}

// Interactive demo
async function startInteractiveTraining() {
    const demo = new LiveTrainingDemo();
    await demo.initialize();
    
    // Generate XOR-like dataset
    const generateBatch = () => {
        const features = [];
        const targets = [];
        
        for (let i = 0; i < 32; i++) {
            const x1 = Math.random();
            const x2 = Math.random();
            features.push(x1, x2);
            targets.push((x1 > 0.5) !== (x2 > 0.5) ? 1 : 0); // XOR logic
        }
        
        return { features, targets };
    };
    
    // Training loop
    const trainButton = document.getElementById('train-btn');
    const stopButton = document.getElementById('stop-btn');
    const statusDiv = document.getElementById('status');
    
    let training = false;
    
    trainButton.onclick = () => {
        training = true;
        trainButton.disabled = true;
        stopButton.disabled = false;
        
        const trainLoop = async () => {
            while (training) {
                const batch = generateBatch();
                const result = await demo.trainStep(batch.features, batch.targets);
                
                statusDiv.innerHTML = `
                    <p>Epoch: ${result.epoch}</p>
                    <p>Loss: ${result.loss.toFixed(4)}</p>
                    <p>Accuracy: ${(result.accuracy * 100).toFixed(2)}%</p>
                `;
                
                // Small delay for visualization
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        };
        
        trainLoop();
    };
    
    stopButton.onclick = () => {
        training = false;
        trainButton.disabled = false;
        stopButton.disabled = true;
    };
}
```

## 📊 Data Science Workflows

### 9. Exploratory Data Analysis

```javascript
class WasmDataAnalyzer {
    constructor() {
        this.preprocessor = null;
        this.metrics = null;
    }
    
    async initialize() {
        await init();
        rustorch.initialize_wasm_runtime();
        
        this.preprocessor = new rustorch.WasmPreprocessor();
        this.metrics = new rustorch.WasmMetrics();
    }
    
    async analyzeDataset(features, targets, feature_names = []) {
        const num_samples = targets.length;
        const feature_size = features.length / num_samples;
        
        const analysis = {
            dataset_info: {
                num_samples,
                feature_size,
                feature_names: feature_names.length > 0 ? feature_names : 
                             Array.from({length: feature_size}, (_, i) => `feature_${i}`)
            },
            feature_stats: {},
            target_distribution: {},
            correlations: []
        };
        
        // Analyze each feature
        for (let f = 0; f < feature_size; f++) {
            const feature_data = [];
            for (let s = 0; s < num_samples; s++) {
                feature_data.push(features[s * feature_size + f]);
            }
            
            const stats = this.preprocessor.compute_stats(feature_data);
            analysis.feature_stats[analysis.dataset_info.feature_names[f]] = {
                mean: stats[0],
                std: stats[1],
                min: stats[2],
                max: stats[3]
            };
        }
        
        // Target distribution
        const unique_targets = [...new Set(targets)];
        for (const target of unique_targets) {
            const count = targets.filter(t => t === target).length;
            analysis.target_distribution[target] = {
                count,
                percentage: (count / num_samples) * 100
            };
        }
        
        // Feature correlations (simplified)
        for (let i = 0; i < feature_size; i++) {
            for (let j = i + 1; j < feature_size; j++) {
                const feature1 = [];
                const feature2 = [];
                
                for (let s = 0; s < num_samples; s++) {
                    feature1.push(features[s * feature_size + i]);
                    feature2.push(features[s * feature_size + j]);
                }
                
                // Simple correlation using cross-correlation
                const correlation = rustorch.cross_correlation(feature1, feature2);
                const max_correlation = Math.max(...correlation.map(x => Math.abs(x)));
                
                analysis.correlations.push({
                    feature1: analysis.dataset_info.feature_names[i],
                    feature2: analysis.dataset_info.feature_names[j],
                    correlation: max_correlation
                });
            }
        }
        
        return analysis;
    }
    
    generateReport(analysis) {
        let report = `
# Data Analysis Report

## Dataset Overview
- Samples: ${analysis.dataset_info.num_samples}
- Features: ${analysis.dataset_info.feature_size}

## Feature Statistics
`;
        
        for (const [name, stats] of Object.entries(analysis.feature_stats)) {
            report += `
### ${name}
- Mean: ${stats.mean.toFixed(3)}
- Std Dev: ${stats.std.toFixed(3)}  
- Range: [${stats.min.toFixed(3)}, ${stats.max.toFixed(3)}]
`;
        }
        
        report += `
## Target Distribution
`;
        for (const [target, info] of Object.entries(analysis.target_distribution)) {
            report += `- Class ${target}: ${info.count} samples (${info.percentage.toFixed(1)}%)\n`;
        }
        
        report += `
## Top Feature Correlations
`;
        const top_correlations = analysis.correlations
            .sort((a, b) => b.correlation - a.correlation)
            .slice(0, 5);
            
        for (const corr of top_correlations) {
            report += `- ${corr.feature1} ↔ ${corr.feature2}: ${corr.correlation.toFixed(3)}\n`;
        }
        
        return report;
    }
}

// Usage
async function exploreDataset() {
    const analyzer = new WasmDataAnalyzer();
    await analyzer.initialize();
    
    // Load or generate dataset
    const { features, targets, feature_names } = loadDataset(); // Your data loading function
    
    const analysis = await analyzer.analyzeDataset(features, targets, feature_names);
    const report = analyzer.generateReport(analysis);
    
    document.getElementById('analysis-report').innerHTML = 
        `<pre>${report}</pre>`;
    
    console.log('Full analysis:', analysis);
}
```

## 🚀 Performance Examples

### 10. Benchmark Suite

```javascript
async function runWASMBenchmarks() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    const performance_monitor = new rustorch.WasmPerformance();
    const memory_monitor = new rustorch.WasmMemoryMonitor();
    
    const benchmarks = [
        {
            name: 'ReLU Activation',
            operation: () => {
                const data = new Array(10000).fill(0).map(() => Math.random() - 0.5);
                return rustorch.relu(data);
            }
        },
        {
            name: 'Matrix Multiplication 100x100',
            operation: () => {
                const a = new Array(10000).fill(0).map(() => Math.random());
                const b = new Array(10000).fill(0).map(() => Math.random());
                return rustorch.WasmTensorOps.matmul(a, 100, 100, b, 100, 100);
            }
        },
        {
            name: 'Batch Normalization',
            operation: () => {
                const batchNorm = new rustorch.WasmBatchNorm(256, 0.1, 1e-5);
                const data = new Array(256 * 32).fill(0).map(() => Math.random());
                return batchNorm.forward(data, 32);
            }
        },
        {
            name: 'FFT 1024 points',
            operation: () => {
                const signal = new Array(1024).fill(0).map((_, i) => 
                    Math.sin(2 * Math.PI * 5 * i / 1024) + 0.1 * Math.random()
                );
                return rustorch.dft(signal);
            }
        }
    ];
    
    console.log('🚀 WASM Performance Benchmarks');
    console.log('================================');
    
    for (const benchmark of benchmarks) {
        // Warm-up run
        benchmark.operation();
        
        // Actual benchmark
        const iterations = 10;
        const times = [];
        
        for (let i = 0; i < iterations; i++) {
            memory_monitor.record_allocation(0); // Reset
            performance_monitor.start();
            
            const result = benchmark.operation();
            
            const elapsed = performance_monitor.elapsed();
            times.push(elapsed);
        }
        
        const avg_time = times.reduce((sum, t) => sum + t, 0) / times.length;
        const min_time = Math.min(...times);
        const max_time = Math.max(...times);
        
        console.log(`${benchmark.name}:`);
        console.log(`  Average: ${avg_time.toFixed(2)}ms`);
        console.log(`  Range: ${min_time.toFixed(2)}-${max_time.toFixed(2)}ms`);
        console.log(`  Peak Memory: ${memory_monitor.peak_usage()} bytes`);
        console.log('');
    }
}
```

## 🎨 Styling and UI

### CSS for ML Dashboards

```css
/* ml-dashboard.css */
.ml-dashboard {
    font-family: 'Monaco', 'Consolas', monospace;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    min-height: 100vh;
    padding: 20px;
}

.training-controls {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

.ml-button {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
}

.ml-button:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

.ml-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 15px;
    backdrop-filter: blur(10px);
}

.chart-container {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
}

.progress-bar {
    width: 100%;
    height: 20px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    transition: width 0.3s;
}
```

これらの例により、ブラウザでの機械学習アプリケーション開発の完全なワークフローを提供します。