// RusTorch WASM Neural Network Example
// This example demonstrates how to create and train a simple neural network using RusTorch WASM

import init, * as rustorch from '../pkg/rustorch.js';

class NeuralNetworkDemo {
    constructor() {
        this.model = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            await init();
            console.log('✓ RusTorch WASM initialized successfully');
            this.initialized = true;
        } catch (error) {
            console.error('❌ Failed to initialize WASM:', error);
            throw error;
        }
    }

    // Create a simple XOR network
    createXORNetwork() {
        if (!this.initialized) {
            throw new Error('WASM not initialized');
        }

        this.model = new rustorch.WasmModel();
        
        // Network architecture: 2 inputs → 4 hidden (ReLU) → 1 output
        this.model.add_linear(2, 4, true);  // Input to hidden layer
        this.model.add_relu();              // ReLU activation
        this.model.add_linear(4, 1, true);  // Hidden to output layer
        
        console.log(`✓ Created XOR network with ${this.model.num_layers()} layers`);
        return this.model;
    }

    // Create training data for XOR problem
    generateXORData() {
        const inputs = [
            [0, 0],  // XOR: 0
            [0, 1],  // XOR: 1
            [1, 0],  // XOR: 1
            [1, 1]   // XOR: 0
        ];
        
        const targets = [0, 1, 1, 0];
        
        return { inputs, targets };
    }

    // Forward pass through the network
    forward(input) {
        if (!this.model) {
            throw new Error('Model not created');
        }

        const inputTensor = new rustorch.WasmTensor(input, [1, input.length]);
        return this.model.forward(inputTensor);
    }

    // Evaluate the model on XOR data
    evaluateXOR() {
        const { inputs, targets } = this.generateXORData();
        const loss = new rustorch.WasmMSELoss();
        
        console.log('\n=== XOR Evaluation ===');
        let totalLoss = 0;
        
        for (let i = 0; i < inputs.length; i++) {
            const output = this.forward(inputs[i]);
            const prediction = output.data()[0];
            const target = targets[i];
            const error = Math.abs(prediction - target);
            
            console.log(`Input: [${inputs[i].join(', ')}] → Prediction: ${prediction.toFixed(4)}, Target: ${target}, Error: ${error.toFixed(4)}`);
            
            const targetTensor = new rustorch.WasmTensor([target], [1]);
            const sampleLoss = loss.compute(output, targetTensor);
            totalLoss += sampleLoss;
        }
        
        const avgLoss = totalLoss / inputs.length;
        console.log(`Average Loss: ${avgLoss.toFixed(6)}`);
        return avgLoss;
    }

    // Demonstrate tensor operations
    demonstrateTensorOps() {
        console.log('\n=== Tensor Operations Demo ===');
        
        const interop = new rustorch.JsInterop();
        
        // Create random tensors
        const shape = new Array(3, 3);
        const tensor1 = interop.random_tensor(shape, -1.0, 1.0);
        const tensor2 = interop.random_tensor(shape, -1.0, 1.0);
        
        console.log('Tensor 1:', Array.from(tensor1.data()).map(x => x.toFixed(3)));
        console.log('Tensor 2:', Array.from(tensor2.data()).map(x => x.toFixed(3)));
        
        // Addition
        const sum = tensor1.add(tensor2);
        console.log('Sum:', Array.from(sum.data()).map(x => x.toFixed(3)));
        
        // Element-wise multiplication
        const product = tensor1.multiply(tensor2);
        console.log('Product:', Array.from(product.data()).map(x => x.toFixed(3)));
        
        // Activation functions
        const relu = tensor1.relu();
        const sigmoid = tensor1.sigmoid();
        
        console.log('ReLU:', Array.from(relu.data()).map(x => x.toFixed(3)));
        console.log('Sigmoid:', Array.from(sigmoid.data()).map(x => x.toFixed(3)));
        
        // Statistics
        console.log(`Sum of elements: ${tensor1.sum().toFixed(3)}`);
        console.log(`Mean: ${tensor1.mean().toFixed(3)}`);
    }

    // Performance benchmark
    async performanceBenchmark() {
        console.log('\n=== Performance Benchmark ===');
        
        const interop = new rustorch.JsInterop();
        const runtime = new rustorch.JsRuntime();
        
        // Reset statistics
        runtime.reset_stats();
        
        // Run benchmarks
        const iterations = 1000;
        const results = interop.benchmark_operations(iterations);
        
        console.log(`Matrix Multiplication (${iterations} iterations): ${results.matmulTimeMs.toFixed(2)} ms`);
        console.log(`Element-wise Addition (${iterations} iterations): ${results.addTimeMs.toFixed(2)} ms`);
        console.log(`ReLU Activation (${iterations} iterations): ${results.reluTimeMs.toFixed(2)} ms`);
        console.log(`Operations per second: ${results.operationsPerSecond.toFixed(0)}`);
        
        // Memory statistics
        const memory = new rustorch.JsMemoryManager();
        console.log(`Memory usage: ${memory.get_memory_usage_mb().toFixed(2)} MB`);
        console.log(`Memory utilization: ${memory.get_memory_utilization().toFixed(1)}%`);
        console.log(`Fragmentation ratio: ${memory.get_fragmentation_ratio().toFixed(3)}`);
    }

    // Memory management demo
    memoryDemo() {
        console.log('\n=== Memory Management Demo ===');
        
        const memory = new rustorch.JsMemoryManager();
        const interop = new rustorch.JsInterop();
        
        console.log(`Initial memory: ${memory.get_memory_usage()} bytes`);
        
        // Create many tensors
        const tensors = [];
        for (let i = 0; i < 100; i++) {
            const shape = new Array(10, 10);
            const tensor = interop.random_tensor(shape, 0, 1);
            tensors.push(tensor);
        }
        
        console.log(`After creating tensors: ${memory.get_memory_usage()} bytes`);
        
        // Clear references (simulate going out of scope)
        tensors.length = 0;
        
        // Trigger garbage collection
        const freedBytes = memory.garbage_collect();
        console.log(`Garbage collected: ${freedBytes} bytes`);
        console.log(`After GC: ${memory.get_memory_usage()} bytes`);
        
        // Memory recommendations
        const recommendations = memory.get_memory_recommendations();
        console.log(`Recommendations: ${recommendations}`);
    }

    // Comprehensive demo
    async runDemo() {
        try {
            await this.initialize();
            
            // Create and evaluate XOR network
            this.createXORNetwork();
            this.evaluateXOR();
            
            // Demonstrate tensor operations
            this.demonstrateTensorOps();
            
            // Performance benchmark
            await this.performanceBenchmark();
            
            // Memory management
            this.memoryDemo();
            
            // System information
            console.log('\n=== System Information ===');
            const interop = new rustorch.JsInterop();
            const sysInfo = interop.get_system_info();
            
            console.log(`Runtime: ${sysInfo.runtime}`);
            console.log(`Architecture: ${sysInfo.architecture}`);
            console.log(`Hardware threads: ${sysInfo.hardwareConcurrency}`);
            
            console.log('\n✓ Demo completed successfully!');
            
        } catch (error) {
            console.error('❌ Demo failed:', error);
        }
    }
}

// Export for use in other modules or browsers
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuralNetworkDemo;
} else {
    window.NeuralNetworkDemo = NeuralNetworkDemo;
}

// Auto-run demo if this script is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    const demo = new NeuralNetworkDemo();
    demo.runDemo();
}