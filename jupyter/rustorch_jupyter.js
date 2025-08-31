/**
 * RusTorch Jupyter Lab Integration
 * Jupyter Lab専用のWASMラッパー
 * 
 * This module provides a Jupyter Lab-optimized interface for RusTorch WASM,
 * handling initialization, async operations, and notebook-specific requirements.
 */

class RusTorchJupyter {
    constructor() {
        this.wasmModule = null;
        this.initialized = false;
        this.initPromise = null;
    }

    /**
     * Initialize RusTorch WASM module for Jupyter Lab
     * Jupyter Lab環境でRusTorch WASMモジュールを初期化
     */
    async initialize() {
        if (this.initialized) {
            return this.wasmModule;
        }

        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = this._initializeWasm();
        return this.initPromise;
    }

    async _initializeWasm() {
        try {
            // For Jupyter Lab, we need to handle different loading contexts
            let wasmModule;
            
            // Try different import strategies for Jupyter environments
            try {
                // Standard ES6 import (local Jupyter)
                wasmModule = await import('../pkg/rustorch.js');
            } catch (e) {
                try {
                    // CDN or different path (cloud Jupyter)
                    wasmModule = await import('./rustorch.js');
                } catch (e2) {
                    throw new Error(`Failed to load RusTorch WASM: ${e2.message}`);
                }
            }

            // Initialize the WASM module
            await wasmModule.default();
            this.wasmModule = wasmModule;
            this.initialized = true;

            console.log('✅ RusTorch WASM initialized for Jupyter Lab');
            return wasmModule;
        } catch (error) {
            console.error('❌ RusTorch initialization failed:', error);
            throw error;
        }
    }

    /**
     * Create a tensor with Jupyter-friendly error handling
     * Jupyter対応のエラーハンドリング付きテンソル作成
     */
    async createTensor(data, shape) {
        const module = await this.initialize();
        try {
            return module.WasmTensor.from_vec(data, shape);
        } catch (error) {
            throw new Error(`Tensor creation failed: ${error.message}`);
        }
    }

    /**
     * Enhanced mathematical functions for Jupyter notebooks
     * Jupyter notebook用の拡張数学関数
     */
    async mathFunctions() {
        const module = await this.initialize();
        return {
            gamma: (values) => module.WasmSpecial.gamma_batch(values),
            bessel_i: (n, values) => module.WasmSpecial.bessel_i_batch(n, values),
            bessel_j: (n, values) => module.WasmSpecial.bessel_j_batch(n, values),
            erf: (values) => module.WasmSpecial.erf_batch(values),
            erfc: (values) => module.WasmSpecial.erfc_batch(values),
            log_gamma: (values) => module.WasmSpecial.log_gamma_batch(values)
        };
    }

    /**
     * Statistical distributions for data science workflows
     * データサイエンスワークフロー用の統計分布
     */
    async distributions() {
        const module = await this.initialize();
        const dist = new module.WasmDistributions();
        return {
            normal: (count, mean = 0.0, std = 1.0) => dist.normal_sample_batch(count, mean, std),
            uniform: (count, low = 0.0, high = 1.0) => dist.uniform_sample_batch(count, low, high),
            exponential: (count, rate = 1.0) => dist.exponential_sample_batch(count, rate),
            gamma: (count, shape, scale) => dist.gamma_sample_batch(count, shape, scale),
            beta: (count, alpha, beta) => dist.beta_sample_batch(count, alpha, beta),
            bernoulli: (count, p = 0.5) => dist.bernoulli_sample_batch(count, p)
        };
    }

    /**
     * Neural network components for ML experiments
     * ML実験用のニューラルネットワークコンポーネント
     */
    async neuralNetwork() {
        const module = await this.initialize();
        return {
            createLinear: (input_size, output_size, bias = true) => 
                new module.WasmLinear(input_size, output_size, bias),
            createConv2d: (in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = true) =>
                new module.WasmConv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        };
    }

    /**
     * Optimizers for training in Jupyter notebooks
     * Jupyter notebook用のトレーニングオプティマイザ
     */
    async optimizers() {
        const module = await this.initialize();
        const optimizer = new module.WasmOptimizer();
        return {
            sgd: (lr, momentum = 0.0) => {
                optimizer.sgd_init(lr, momentum);
                return {
                    step: (params, grads) => optimizer.sgd_step(params, grads),
                    zero_grad: () => optimizer.sgd_zero_grad()
                };
            },
            adam: (lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) => {
                optimizer.adam_init(lr, beta1, beta2, eps);
                return {
                    step: (params, grads) => optimizer.adam_step(params, grads),
                    zero_grad: () => optimizer.adam_zero_grad()
                };
            }
        };
    }

    /**
     * WebGPU acceleration for supported browsers
     * サポート対象ブラウザでのWebGPU加速
     */
    async webgpu() {
        const module = await this.initialize();
        
        if (!module.WebGPUSimple) {
            console.warn('⚠️ WebGPU not available in this build');
            return null;
        }

        const webgpu = new module.WebGPUSimple();
        const supported = await webgpu.check_webgpu_support();
        
        if (supported) {
            console.log('🚀 WebGPU acceleration available');
            await webgpu.initialize();
            return webgpu;
        } else {
            console.log('ℹ️ WebGPU not supported, using CPU operations');
            return null;
        }
    }

    /**
     * Jupyter display utilities for visualization
     * 可視化用のJupyter表示ユーティリティ
     */
    displayTensor(tensor, title = "Tensor") {
        try {
            const shape = tensor.shape();
            const data = tensor.as_slice();
            
            // Create HTML table for small tensors
            if (data.length <= 100) {
                const html = this._createTensorTable(data, shape, title);
                this._displayHTML(html);
            } else {
                // Summary for large tensors
                const summary = this._createTensorSummary(data, shape, title);
                this._displayHTML(summary);
            }
        } catch (error) {
            console.error('Display error:', error);
        }
    }

    _createTensorTable(data, shape, title) {
        let html = `<div style="font-family: monospace; margin: 10px 0;">`;
        html += `<strong>${title}</strong> - Shape: [${shape.join(', ')}]<br>`;
        html += `<table style="border-collapse: collapse; margin-top: 5px;">`;
        
        if (shape.length === 1) {
            // 1D tensor
            html += '<tr>';
            for (let i = 0; i < Math.min(data.length, 10); i++) {
                html += `<td style="border: 1px solid #ccc; padding: 3px 6px;">${data[i].toFixed(4)}</td>`;
            }
            if (data.length > 10) html += '<td>...</td>';
            html += '</tr>';
        } else if (shape.length === 2) {
            // 2D tensor (matrix)
            const rows = shape[0];
            const cols = shape[1];
            for (let i = 0; i < Math.min(rows, 5); i++) {
                html += '<tr>';
                for (let j = 0; j < Math.min(cols, 10); j++) {
                    const idx = i * cols + j;
                    html += `<td style="border: 1px solid #ccc; padding: 3px 6px;">${data[idx].toFixed(4)}</td>`;
                }
                if (cols > 10) html += '<td>...</td>';
                html += '</tr>';
            }
            if (rows > 5) {
                html += '<tr><td colspan="100%" style="text-align: center;">...</td></tr>';
            }
        }
        
        html += '</table></div>';
        return html;
    }

    _createTensorSummary(data, shape, title) {
        const min = Math.min(...data);
        const max = Math.max(...data);
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        
        return `
            <div style="font-family: monospace; margin: 10px 0; padding: 10px; border: 1px solid #ddd;">
                <strong>${title}</strong><br>
                Shape: [${shape.join(', ')}] (${data.length} elements)<br>
                Statistics: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}
            </div>
        `;
    }

    _displayHTML(html) {
        // For Jupyter Lab environment
        if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {
            const element = document.createElement('div');
            element.innerHTML = html;
            Jupyter.notebook.get_selected_cell().output_area.element.append(element);
        } else if (typeof IPython !== 'undefined') {
            // For IPython environments
            const element = document.createElement('div');
            element.innerHTML = html;
            element.style.display = 'block';
            document.body.appendChild(element);
        } else {
            // Fallback to console
            console.log(html.replace(/<[^>]*>/g, ''));
        }
    }

    /**
     * Jupyter-specific utilities
     * Jupyter専用ユーティリティ
     */
    async utils() {
        return {
            display: this.displayTensor.bind(this),
            benchmark: async (operation, iterations = 1000) => {
                const start = performance.now();
                for (let i = 0; i < iterations; i++) {
                    await operation();
                }
                const end = performance.now();
                const avgTime = (end - start) / iterations;
                console.log(`📊 Benchmark: ${avgTime.toFixed(2)}ms/operation (${iterations} iterations)`);
                return avgTime;
            },
            checkCapabilities: async () => {
                const module = await this.initialize();
                const webgpu = await this.webgpu();
                return {
                    wasm: true,
                    webgpu: webgpu !== null,
                    enhanced_math: !!module.WasmSpecial,
                    distributions: !!module.WasmDistributions,
                    optimizers: !!module.WasmOptimizer,
                    neural_networks: !!(module.WasmLinear && module.WasmConv2d)
                };
            }
        };
    }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    // Node.js environment
    module.exports = { RusTorchJupyter };
} else if (typeof globalThis !== 'undefined') {
    // Browser/Jupyter environment
    globalThis.RusTorchJupyter = RusTorchJupyter;
} else {
    // Fallback
    window.RusTorchJupyter = RusTorchJupyter;
}

// Auto-initialize for Jupyter environments
if (typeof Jupyter !== 'undefined' || typeof IPython !== 'undefined') {
    globalThis.rustorch_jupyter = new RusTorchJupyter();
    console.log('🎓 RusTorch Jupyter integration loaded');
}