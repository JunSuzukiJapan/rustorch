// RusTorch WASM Performance Testing Suite
// Comprehensive performance testing for tensor operations and neural networks

import init, * as rustorch from '../pkg/rustorch.js';

class PerformanceTester {
    constructor() {
        this.initialized = false;
        this.results = {};
    }

    async initialize() {
        try {
            await init();
            this.initialized = true;
            console.log('✓ RusTorch WASM Performance Tester initialized');
        } catch (error) {
            console.error('❌ Failed to initialize:', error);
            throw error;
        }
    }

    // Benchmark tensor operations with different sizes
    benchmarkTensorOperations() {
        console.log('\n=== Tensor Operations Benchmark ===');
        
        const sizes = [
            { name: 'Small', shape: [10, 10], elements: 100 },
            { name: 'Medium', shape: [100, 100], elements: 10000 },
            { name: 'Large', shape: [500, 500], elements: 250000 }
        ];

        const operations = ['add', 'multiply', 'matmul'];
        const results = {};

        for (const size of sizes) {
            console.log(`\nTesting ${size.name} tensors (${size.shape.join('x')}):`);
            results[size.name] = {};

            // Create test tensors
            const interop = new rustorch.JsInterop();
            const tensor1 = interop.random_tensor(new Array(...size.shape), 0, 1);
            const tensor2 = interop.random_tensor(new Array(...size.shape), 0, 1);

            // Test each operation
            for (const op of operations) {
                const startTime = performance.now();
                const iterations = size.elements > 10000 ? 10 : 100;

                try {
                    for (let i = 0; i < iterations; i++) {
                        let result;
                        switch (op) {
                            case 'add':
                                result = tensor1.add(tensor2);
                                break;
                            case 'multiply':
                                result = tensor1.multiply(tensor2);
                                break;
                            case 'matmul':
                                result = tensor1.matmul(tensor2);
                                break;
                        }
                    }

                    const endTime = performance.now();
                    const totalTime = endTime - startTime;
                    const avgTime = totalTime / iterations;
                    const throughput = (size.elements * iterations) / (totalTime / 1000);

                    results[size.name][op] = {
                        totalTime,
                        avgTime,
                        throughput,
                        iterations
                    };

                    console.log(`  ${op.padEnd(8)}: ${avgTime.toFixed(3)} ms/op, ${throughput.toFixed(0)} elements/sec`);

                } catch (error) {
                    console.log(`  ${op.padEnd(8)}: FAILED - ${error.message}`);
                    results[size.name][op] = { error: error.message };
                }
            }
        }

        return results;
    }

    // Benchmark activation functions
    benchmarkActivations() {
        console.log('\n=== Activation Functions Benchmark ===');
        
        const shapes = [
            [1000],
            [100, 100],
            [50, 50, 8]
        ];

        const activations = ['relu', 'sigmoid'];
        const results = {};

        for (const shape of shapes) {
            const shapeName = shape.join('x');
            console.log(`\nTesting shape ${shapeName}:`);
            results[shapeName] = {};

            const interop = new rustorch.JsInterop();
            const tensor = interop.random_tensor(new Array(...shape), -2, 2);

            for (const activation of activations) {
                const iterations = 1000;
                const startTime = performance.now();

                for (let i = 0; i < iterations; i++) {
                    let result;
                    switch (activation) {
                        case 'relu':
                            result = tensor.relu();
                            break;
                        case 'sigmoid':
                            result = tensor.sigmoid();
                            break;
                    }
                }

                const endTime = performance.now();
                const totalTime = endTime - startTime;
                const avgTime = totalTime / iterations;
                const elements = shape.reduce((a, b) => a * b, 1);
                const throughput = (elements * iterations) / (totalTime / 1000);

                results[shapeName][activation] = {
                    avgTime,
                    throughput,
                    elements
                };

                console.log(`  ${activation.padEnd(8)}: ${avgTime.toFixed(3)} ms/op, ${throughput.toFixed(0)} elements/sec`);
            }
        }

        return results;
    }

    // Benchmark neural network forward passes
    benchmarkNeuralNetworks() {
        console.log('\n=== Neural Network Benchmark ===');
        
        const architectures = [
            { name: 'Small', layers: [10, 20, 10, 1] },
            { name: 'Medium', layers: [100, 200, 100, 10] },
            { name: 'Large', layers: [500, 1000, 500, 100] }
        ];

        const results = {};

        for (const arch of architectures) {
            console.log(`\nTesting ${arch.name} network (${arch.layers.join(' → ')}):`);
            
            // Create network
            const model = new rustorch.WasmModel();
            for (let i = 0; i < arch.layers.length - 1; i++) {
                model.add_linear(arch.layers[i], arch.layers[i + 1], true);
                if (i < arch.layers.length - 2) {
                    model.add_relu();
                }
            }

            // Create input
            const input = new rustorch.WasmTensor(
                Array.from({ length: arch.layers[0] }, () => Math.random()),
                [1, arch.layers[0]]
            );

            // Benchmark forward passes
            const iterations = arch.layers[0] > 100 ? 10 : 100;
            const startTime = performance.now();

            for (let i = 0; i < iterations; i++) {
                const output = model.forward(input);
            }

            const endTime = performance.now();
            const totalTime = endTime - startTime;
            const avgTime = totalTime / iterations;

            results[arch.name] = {
                avgTime,
                totalTime,
                iterations,
                parameters: this.countParameters(arch.layers)
            };

            console.log(`  Forward pass: ${avgTime.toFixed(3)} ms/op (${iterations} iterations)`);
            console.log(`  Parameters: ${results[arch.name].parameters}`);
        }

        return results;
    }

    // Count network parameters
    countParameters(layers) {
        let total = 0;
        for (let i = 0; i < layers.length - 1; i++) {
            total += layers[i] * layers[i + 1] + layers[i + 1]; // weights + bias
        }
        return total;
    }

    // Memory usage during operations
    benchmarkMemoryUsage() {
        console.log('\n=== Memory Usage Benchmark ===');
        
        const memory = new rustorch.JsMemoryManager();
        const interop = new rustorch.JsInterop();
        
        console.log(`Initial memory: ${memory.get_memory_usage_mb().toFixed(2)} MB`);

        // Test memory usage with different tensor sizes
        const testSizes = [
            { name: 'Small tensors', count: 100, size: [10, 10] },
            { name: 'Medium tensors', count: 50, size: [50, 50] },
            { name: 'Large tensors', count: 10, size: [200, 200] }
        ];

        const results = {};

        for (const test of testSizes) {
            console.log(`\n${test.name}:`);
            
            const initialMemory = memory.get_memory_usage();
            const tensors = [];

            // Allocate tensors
            for (let i = 0; i < test.count; i++) {
                const tensor = interop.random_tensor(new Array(...test.size), 0, 1);
                tensors.push(tensor);
                
                if (i % Math.max(1, Math.floor(test.count / 5)) === 0) {
                    const currentMemory = memory.get_memory_usage_mb();
                    console.log(`  After ${i + 1} tensors: ${currentMemory.toFixed(2)} MB`);
                }
            }

            const peakMemory = memory.get_memory_usage();
            const memoryIncrease = peakMemory - initialMemory;

            // Clear references
            tensors.length = 0;
            
            // Force garbage collection
            const freedBytes = memory.garbage_collect();
            const finalMemory = memory.get_memory_usage();

            results[test.name] = {
                initialMemory,
                peakMemory,
                memoryIncrease,
                freedBytes,
                finalMemory,
                efficiency: (memoryIncrease - freedBytes) / memoryIncrease
            };

            console.log(`  Peak memory: ${(peakMemory / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  Memory increase: ${(memoryIncrease / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  Freed by GC: ${(freedBytes / 1024 / 1024).toFixed(2)} MB`);
            console.log(`  Memory efficiency: ${(results[test.name].efficiency * 100).toFixed(1)}%`);
        }

        return results;
    }

    // Comprehensive performance test
    async runFullBenchmark() {
        try {
            await this.initialize();

            console.log('🚀 Starting RusTorch WASM Performance Benchmark Suite');
            console.log('=' .repeat(60));

            // Run all benchmarks
            this.results.tensorOps = this.benchmarkTensorOperations();
            this.results.activations = this.benchmarkActivations();
            this.results.neuralNetworks = this.benchmarkNeuralNetworks();
            this.results.memoryUsage = this.benchmarkMemoryUsage();

            // Runtime statistics
            console.log('\n=== Runtime Statistics ===');
            const runtime = new rustorch.JsRuntime();
            console.log(`Total operations: ${runtime.get_operations_count()}`);
            console.log(`Average operation time: ${runtime.get_average_operation_time().toFixed(3)} ms`);

            // System information
            console.log('\n=== System Information ===');
            const interop = new rustorch.JsInterop();
            const sysInfo = interop.get_system_info();
            console.log(`Runtime: ${sysInfo.runtime}`);
            console.log(`Architecture: ${sysInfo.architecture}`);
            console.log(`Hardware threads: ${sysInfo.hardwareConcurrency}`);

            console.log('\n✓ Benchmark suite completed successfully!');
            return this.results;

        } catch (error) {
            console.error('❌ Benchmark suite failed:', error);
            throw error;
        }
    }

    // Export results to JSON
    exportResults() {
        const timestamp = new Date().toISOString();
        const report = {
            timestamp,
            system: this.getSystemInfo(),
            results: this.results
        };

        if (typeof require !== 'undefined') {
            // Node.js environment
            const fs = require('fs');
            const filename = `benchmark_results_${timestamp.replace(/[:.]/g, '-')}.json`;
            fs.writeFileSync(filename, JSON.stringify(report, null, 2));
            console.log(`Results exported to ${filename}`);
        } else {
            // Browser environment
            console.log('Benchmark Results:', JSON.stringify(report, null, 2));
        }

        return report;
    }

    getSystemInfo() {
        if (typeof navigator !== 'undefined') {
            return {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                hardwareConcurrency: navigator.hardwareConcurrency
            };
        } else if (typeof process !== 'undefined') {
            return {
                platform: process.platform,
                arch: process.arch,
                nodeVersion: process.version
            };
        }
        return {};
    }
}

// Auto-run if executed directly
if (typeof require !== 'undefined' && require.main === module) {
    const tester = new PerformanceTester();
    tester.runFullBenchmark()
        .then(() => tester.exportResults())
        .catch(console.error);
}

// Export for browser use
if (typeof window !== 'undefined') {
    window.PerformanceTester = PerformanceTester;
}

export default PerformanceTester;