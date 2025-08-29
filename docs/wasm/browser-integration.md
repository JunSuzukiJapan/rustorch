# Browser Integration Guide

RusTorch WASMモジュールをブラウザアプリケーションに統合する完全ガイド

## 🚀 Setup & Installation

### 1. Build WASM Package

```bash
# Project root directory
cd rustorch
wasm-pack build --target web --features wasm --no-default-features

# Generated files will be in pkg/ directory
ls pkg/
# rustorch.js, rustorch_bg.wasm, rustorch.d.ts, package.json
```

### 2. HTML Setup

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RusTorch ML in Browser</title>
</head>
<body>
    <script type="module">
        import init, * as rustorch from './pkg/rustorch.js';
        
        async function run() {
            // Initialize WASM module
            await init();
            rustorch.initialize_wasm_runtime();
            
            console.log('RusTorch WASM initialized!');
            
            // Your ML code here
            main();
        }
        
        run();
    </script>
</body>
</html>
```

### 3. Modern JavaScript/TypeScript Project

```javascript
// main.js
import init, * as rustorch from './pkg/rustorch.js';

async function initializeML() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    return rustorch;
}

export { initializeML };
```

```typescript
// types.d.ts
declare module './pkg/rustorch.js' {
    export function initialize_wasm_runtime(): void;
    export class WasmPreprocessor {
        static compute_stats(data: number[]): number[];
        static min_max_normalize(data: number[], min: number, max: number): number[];
    }
    // ... other type definitions
}
```

## 🔬 Complete ML Pipeline Example

### Real-time Image Classification

```html
<!DOCTYPE html>
<html>
<head>
    <title>Real-time ML Classification</title>
</head>
<body>
    <video id="video" width="640" height="480" autoplay></video>
    <canvas id="canvas" width="224" height="224" style="display: none;"></canvas>
    <div id="results"></div>

    <script type="module">
        import init, * as rustorch from './pkg/rustorch.js';
        
        class MLPipeline {
            constructor() {
                // Vision processing
                this.vision = rustorch.WasmVision;
                this.imagenet_mean = [0.485, 0.456, 0.406];
                this.imagenet_std = [0.229, 0.224, 0.225];
                
                // Neural network model
                this.conv1 = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
                this.conv2 = new rustorch.WasmConv2d(32, 64, 3, 2, 1, true);
                this.conv3 = new rustorch.WasmConv2d(64, 128, 3, 2, 1, true);
                this.classifier = new rustorch.WasmLinear(128 * 56 * 56, 1000, true);
                
                this.metrics = new rustorch.WasmMetrics();
                this.performance = new rustorch.WasmPerformance();
            }
            
            async processFrame(imageData, width, height) {
                this.performance.start();
                
                // Convert canvas ImageData to float array
                const rgbData = [];
                for (let i = 0; i < imageData.data.length; i += 4) {
                    rgbData.push(imageData.data[i] / 255.0);     // R
                    rgbData.push(imageData.data[i + 1] / 255.0); // G
                    rgbData.push(imageData.data[i + 2] / 255.0); // B
                }
                
                // Resize to 224x224 for model input
                let processed = this.vision.resize(rgbData, height, width, 224, 224, 3);
                
                // Normalize with ImageNet statistics
                processed = this.vision.normalize(processed, this.imagenet_mean, this.imagenet_std, 3);
                
                // Forward pass through CNN
                let features = this.conv1.forward(processed, 1, 224, 224);
                features = rustorch.relu(features);
                
                features = this.conv2.forward(features, 1, 224, 224);
                features = rustorch.relu(features);
                
                features = this.conv3.forward(features, 1, 112, 112);  // After stride=2
                features = rustorch.relu(features);
                
                // Classification
                const logits = this.classifier.forward(features, 1);
                const predictions = rustorch.softmax(logits);
                
                const processingTime = this.performance.elapsed();
                
                return {
                    predictions: predictions,
                    processingTime: processingTime
                };
            }
        }
        
        async function setupWebcam() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const resultsDiv = document.getElementById('results');
            
            // Initialize RusTorch
            await init();
            rustorch.initialize_wasm_runtime();
            const mlPipeline = new MLPipeline();
            
            // Setup webcam
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            // Process frames
            const processFrame = () => {
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, 224, 224);
                const imageData = ctx.getImageData(0, 0, 224, 224);
                
                // Run inference
                mlPipeline.processFrame(imageData, 224, 224).then(result => {
                    const topPrediction = Math.max(...result.predictions);
                    const topIndex = result.predictions.indexOf(topPrediction);
                    
                    resultsDiv.innerHTML = `
                        <h3>Prediction Results</h3>
                        <p>Top class: ${topIndex} (confidence: ${(topPrediction * 100).toFixed(2)}%)</p>
                        <p>Processing time: ${result.processingTime.toFixed(2)}ms</p>
                    `;
                });
                
                requestAnimationFrame(processFrame);
            };
            
            video.addEventListener('loadeddata', processFrame);
        }
        
        setupWebcam();
    </script>
</body>
</html>
```

## 📈 Training in Browser

### Mini-batch Training Example

```javascript
import init, * as rustorch from './pkg/rustorch.js';

class BrowserTrainer {
    constructor() {
        this.optimizer = new rustorch.WasmOptimizer('adam', 0.001);
        this.batchNorm = new rustorch.WasmBatchNorm(10, 0.1, 1e-5);
        this.preprocessor = new rustorch.WasmPreprocessor();
        this.metrics = new rustorch.WasmMetrics();
    }
    
    async train(features, targets, epochs = 100, batch_size = 32) {
        // Preprocessing
        const stats = this.preprocessor.compute_stats(features);
        const normalized_features = this.preprocessor.z_score_normalize(
            features, stats[0], stats[1]
        );
        
        // Train-test split
        const split = this.preprocessor.train_test_split(
            normalized_features, targets, 10, 0.2, 42
        );
        
        const train_history = [];
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Create batches
            const batches = this.preprocessor.create_batches(
                split.trainFeatures, split.trainTargets, 10, batch_size
            );
            
            let epoch_loss = 0.0;
            
            // Training loop
            for (let batch_idx = 0; batch_idx < batches.length; batch_idx++) {
                const batch = batches[batch_idx];
                
                // Forward pass
                this.batchNorm.set_training(true);
                let output = this.batchNorm.forward(batch.features, batch.batchSize);
                output = rustorch.relu(output);
                
                // Simple linear layer (placeholder)
                const weights = new Array(output.length).fill(0.5);
                const predictions = output.map((x, i) => x * weights[i % weights.length]);
                
                // Loss calculation
                const loss = rustorch.mse_loss(predictions, batch.targets);
                epoch_loss += loss;
                
                // Compute gradients (simplified)
                const gradients = output.map((x, i) => 
                    2 * (predictions[i] - batch.targets[i]) * x / batch.batchSize
                );
                
                // Optimization step
                const updated_weights = this.optimizer.step('weights', weights, gradients);
            }
            
            // Validation
            if (epoch % 10 === 0) {
                this.batchNorm.set_training(false);
                const val_output = this.batchNorm.forward(split.testFeatures, split.testFeatures.length / 10);
                const val_predictions = val_output.map(x => x > 0.5 ? 1 : 0);
                const val_accuracy = this.metrics.accuracy(val_predictions, split.testTargets);
                
                train_history.push({
                    epoch,
                    loss: epoch_loss / batches.length,
                    val_accuracy
                });
                
                console.log(`Epoch ${epoch}: Loss=${train_history[train_history.length-1].loss.toFixed(4)}, Val Acc=${val_accuracy.toFixed(4)}`);
            }
        }
        
        return train_history;
    }
}

// Usage
async function trainModel() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    const trainer = new BrowserTrainer();
    
    // Generate sample data
    const features = new Array(1000 * 10).fill(0).map(() => Math.random());
    const targets = new Array(1000).fill(0).map(() => Math.round(Math.random()));
    
    const history = await trainer.train(features, targets, 100, 32);
    console.log('Training completed:', history);
}
```

## 🎮 Interactive Visualization

### Real-time Training Dashboard

```javascript
class TrainingDashboard {
    constructor() {
        this.chart_canvas = document.getElementById('loss-chart');
        this.ctx = this.chart_canvas.getContext('2d');
        this.loss_history = [];
        this.accuracy_history = [];
    }
    
    updateMetrics(epoch, loss, accuracy) {
        this.loss_history.push({ epoch, loss });
        this.accuracy_history.push({ epoch, accuracy });
        
        // Redraw chart
        this.drawChart();
    }
    
    drawChart() {
        const ctx = this.ctx;
        const width = this.chart_canvas.width;
        const height = this.chart_canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw loss curve
        ctx.strokeStyle = 'red';
        ctx.beginPath();
        this.loss_history.forEach((point, i) => {
            const x = (i / this.loss_history.length) * width;
            const y = height - (point.loss / Math.max(...this.loss_history.map(p => p.loss))) * height;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // Draw accuracy curve
        ctx.strokeStyle = 'blue';
        ctx.beginPath();
        this.accuracy_history.forEach((point, i) => {
            const x = (i / this.accuracy_history.length) * width;
            const y = height - point.accuracy * height;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}
```

## 🔧 Performance Optimization

### Memory Management

```javascript
// Efficient memory usage
const monitor = new rustorch.WasmMemoryMonitor();

function processLargeBatch(data) {
    const chunks = [];
    const chunk_size = 1000;
    
    for (let i = 0; i < data.length; i += chunk_size) {
        const chunk = data.slice(i, i + chunk_size);
        
        monitor.record_allocation(chunk.length * 4); // 4 bytes per f32
        
        const result = rustorch.relu(chunk);
        chunks.push(result);
        
        monitor.record_deallocation(chunk.length * 4);
    }
    
    console.log('Peak memory usage:', monitor.peak_usage(), 'bytes');
    return chunks.flat();
}
```

### Parallel Processing with Web Workers

```javascript
// main.js
class ParallelMLProcessor {
    constructor(num_workers = 4) {
        this.workers = [];
        this.task_queue = [];
        
        for (let i = 0; i < num_workers; i++) {
            const worker = new Worker('./ml-worker.js', { type: 'module' });
            this.workers.push(worker);
        }
    }
    
    async processParallel(batches) {
        const promises = batches.map((batch, i) => {
            const worker = this.workers[i % this.workers.length];
            return new Promise(resolve => {
                worker.onmessage = (e) => resolve(e.data);
                worker.postMessage({ batch, operation: 'forward_pass' });
            });
        });
        
        return Promise.all(promises);
    }
}

// ml-worker.js
import init, * as rustorch from './pkg/rustorch.js';

let initialized = false;

self.onmessage = async function(e) {
    if (!initialized) {
        await init();
        rustorch.initialize_wasm_runtime();
        initialized = true;
    }
    
    const { batch, operation } = e.data;
    
    let result;
    switch (operation) {
        case 'forward_pass':
            const activated = rustorch.relu(batch.features);
            const output = rustorch.softmax(activated);
            result = { output, batch_id: batch.id };
            break;
        default:
            result = { error: 'Unknown operation' };
    }
    
    self.postMessage(result);
};
```

## 📊 Integration with Chart Libraries

### Chart.js Integration

```javascript
import Chart from 'chart.js/auto';

class MLVisualization {
    constructor() {
        this.loss_chart = new Chart(document.getElementById('loss-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'red',
                    backgroundColor: 'rgba(255, 0, 0, 0.1)'
                }, {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'blue',
                    backgroundColor: 'rgba(0, 0, 255, 0.1)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    updateTrainingMetrics(epoch, loss, accuracy) {
        this.loss_chart.data.labels.push(epoch);
        this.loss_chart.data.datasets[0].data.push(loss);
        this.loss_chart.data.datasets[1].data.push(accuracy);
        this.loss_chart.update();
    }
}
```

## 🎯 Production Deployment

### Bundle Optimization

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
    entry: './src/index.js',
    mode: 'production',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.js',
    },
    experiments: {
        asyncWebAssembly: true,
        syncWebAssembly: true
    },
    module: {
        rules: [
            {
                test: /\.wasm$/,
                type: 'webassembly/async'
            }
        ]
    },
    optimization: {
        splitChunks: {
            chunks: 'all',
            cacheGroups: {
                wasm: {
                    test: /\.wasm$/,
                    name: 'wasm',
                    chunks: 'all'
                }
            }
        }
    }
};
```

### CDN Deployment

```html
<!-- Load from CDN -->
<script type="module">
    import init from 'https://cdn.jsdelivr.net/npm/rustorch-wasm@latest/rustorch.js';
    // ... rest of your code
</script>
```

## 🔒 Security Considerations

### Content Security Policy

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               wasm-eval 'unsafe-eval'; 
               script-src 'self' 'wasm-unsafe-eval';">
```

### Safe Data Handling

```javascript
class SecureMLProcessor {
    constructor() {
        this.max_input_size = 1024 * 1024; // 1MB limit
        this.sanitizer = new rustorch.WasmPreprocessor();
    }
    
    validateInput(data) {
        if (data.length > this.max_input_size) {
            throw new Error('Input too large');
        }
        
        // Check for NaN or infinite values
        if (data.some(x => !isFinite(x))) {
            throw new Error('Invalid numeric data');
        }
        
        return true;
    }
    
    processSecurely(data) {
        this.validateInput(data);
        
        try {
            return this.sanitizer.min_max_normalize(data, 0.0, 1.0);
        } catch (error) {
            console.error('Processing failed:', error);
            return null;
        }
    }
}
```

## 🧪 Testing

### Unit Testing with Jest

```javascript
// ml.test.js
import init, * as rustorch from '../pkg/rustorch.js';

describe('RusTorch WASM', () => {
    beforeAll(async () => {
        await init();
        rustorch.initialize_wasm_runtime();
    });
    
    test('activation functions', () => {
        const input = [-1.0, 0.0, 1.0];
        
        const relu_output = rustorch.relu(input);
        expect(relu_output).toEqual([0.0, 0.0, 1.0]);
        
        const sigmoid_output = rustorch.sigmoid(input);
        expect(sigmoid_output[1]).toBeCloseTo(0.5, 5);
    });
    
    test('preprocessing utilities', () => {
        const data = [1, 2, 3, 4, 5];
        const normalized = rustorch.WasmPreprocessor.min_max_normalize(data, 1, 5);
        expect(normalized).toEqual([0.0, 0.25, 0.5, 0.75, 1.0]);
    });
    
    test('metrics calculation', () => {
        const predictions = [0, 1, 1, 0];
        const targets = [0, 1, 0, 0];
        const accuracy = rustorch.WasmMetrics.accuracy(predictions, targets);
        expect(accuracy).toBe(0.75);
    });
});
```

### Performance Testing

```javascript
async function benchmarkWASM() {
    await init();
    rustorch.initialize_wasm_runtime();
    
    const perf = new rustorch.WasmPerformance();
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const data = new Array(size).fill(0).map(() => Math.random());
        
        perf.start();
        const result = rustorch.relu(data);
        const time = perf.elapsed();
        
        console.log(`ReLU ${size} elements: ${time.toFixed(2)}ms`);
        
        perf.start();
        const normalized = rustorch.WasmPreprocessor.min_max_normalize(data, 0, 1);
        const normalize_time = perf.elapsed();
        
        console.log(`Normalize ${size} elements: ${normalize_time.toFixed(2)}ms`);
    }
}
```

## 🔄 Integration Patterns

### React Integration

```jsx
import React, { useEffect, useState } from 'react';
import init, * as rustorch from './pkg/rustorch.js';

function MLComponent() {
    const [wasmReady, setWasmReady] = useState(false);
    const [result, setResult] = useState(null);
    
    useEffect(() => {
        async function initWasm() {
            await init();
            rustorch.initialize_wasm_runtime();
            setWasmReady(true);
        }
        initWasm();
    }, []);
    
    const processData = () => {
        if (!wasmReady) return;
        
        const data = [1, 2, 3, 4, 5];
        const output = rustorch.relu(data);
        setResult(output);
    };
    
    return (
        <div>
            <button onClick={processData} disabled={!wasmReady}>
                Process Data
            </button>
            {result && <div>Result: {result.join(', ')}</div>}
        </div>
    );
}
```

### Vue.js Integration

```vue
<template>
    <div>
        <button @click="processData" :disabled="!wasmReady">
            Process Data
        </button>
        <div v-if="result">Result: {{ result.join(', ') }}</div>
    </div>
</template>

<script>
import init, * as rustorch from './pkg/rustorch.js';

export default {
    data() {
        return {
            wasmReady: false,
            result: null
        };
    },
    
    async mounted() {
        await init();
        rustorch.initialize_wasm_runtime();
        this.wasmReady = true;
    },
    
    methods: {
        processData() {
            const data = [1, 2, 3, 4, 5];
            this.result = rustorch.relu(data);
        }
    }
};
</script>
```

## 📱 Mobile Considerations

### Progressive Web App (PWA)

```javascript
// sw.js - Service Worker for offline ML
const CACHE_NAME = 'ml-cache-v1';
const urlsToCache = [
    '/',
    '/pkg/rustorch.js',
    '/pkg/rustorch_bg.wasm'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});
```

## 🐛 Debugging

### WASM Debugging Tools

```javascript
// Debug utilities
class WasmDebugger {
    constructor() {
        this.monitor = new rustorch.WasmMemoryMonitor();
        this.performance = new rustorch.WasmPerformance();
    }
    
    profile(operation, ...args) {
        this.performance.start();
        const start_memory = this.monitor.current_usage();
        
        try {
            const result = operation(...args);
            
            const end_memory = this.monitor.current_usage();
            const elapsed = this.performance.elapsed();
            
            console.log('Profile:', {
                operation: operation.name,
                time_ms: elapsed,
                memory_delta: end_memory - start_memory,
                peak_memory: this.monitor.peak_usage()
            });
            
            return result;
        } catch (error) {
            console.error('Operation failed:', error);
            throw error;
        }
    }
}

// Usage
const debugger = new WasmDebugger();
const result = debugger.profile(rustorch.relu, [1, 2, 3]);
```

## 🔗 External Resources

- [WebAssembly Documentation](https://webassembly.org/)
- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [RusTorch Main Documentation](https://docs.rs/rustorch)
- [Performance Benchmarks](./benchmarks.md)