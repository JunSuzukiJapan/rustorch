# Browser Compatibility Guide for RusTorch WASM
# RusTorch WASM ブラウザ互換性ガイド

> 📋 **Complete API Reference**: [WASM API Documentation](WASM_API_DOCUMENTATION.md)  
> 🔗 **Setup Guide**: [WASM Guide](WASM_GUIDE.md)

## Supported Browsers / サポートブラウザ

### Chrome (Primary Target) / Chrome（主要ターゲット）

| Feature | Min Version | Status | Performance | Notes |
|---------|-------------|--------|-------------|-------|
| **WASM** | 69+ | ✅ Full | Excellent | Native support |
| **WebGPU** | 113+ | ✅ Full | Optimal | Hardware accelerated |
| **SharedArrayBuffer** | 68+ | ✅ Full | High | Multi-threading support |
| **WebAssembly SIMD** | 91+ | ✅ Full | High | Vectorized operations |

**Setup Instructions:**
```bash
# Enable WebGPU in Chrome
chrome://flags/#enable-unsafe-webgpu -> Enabled
chrome://flags/#enable-webgpu-developer-features -> Enabled

# Verify WebGPU support
# Open DevTools -> Console
console.log('WebGPU supported:', !!navigator.gpu);
```

### Firefox / Firefox

| Feature | Min Version | Status | Performance | Notes |
|---------|-------------|--------|-------------|-------|
| **WASM** | 52+ | ✅ Full | Good | Native support |
| **WebGPU** | 113+ | ⚠️ Experimental | Limited | Requires flags |
| **SharedArrayBuffer** | 79+ | ✅ Full | Good | Multi-threading |
| **WebAssembly SIMD** | 89+ | ✅ Full | Good | Vectorized operations |

**Setup Instructions:**
```bash
# Enable WebGPU in Firefox
# Open about:config
dom.webgpu.enabled -> true
gfx.webgpu.force-enabled -> true

# Note: WebGPU support is experimental
```

### Safari / Safari

| Feature | Min Version | Status | Performance | Notes |
|---------|-------------|--------|-------------|-------|
| **WASM** | 11+ | ✅ Full | Good | Native support |
| **WebGPU** | Not supported | ❌ None | N/A | CPU fallback only |
| **SharedArrayBuffer** | 15.2+ | ✅ Limited | Medium | Security restrictions |
| **WebAssembly SIMD** | 16.4+ | ✅ Full | Good | ARM optimized |

### Edge / Edge

| Feature | Min Version | Status | Performance | Notes |
|---------|-------------|--------|-------------|-------|
| **WASM** | 79+ | ✅ Full | Excellent | Chromium-based |
| **WebGPU** | 113+ | ✅ Full | Optimal | Same as Chrome |
| **SharedArrayBuffer** | 79+ | ✅ Full | High | Full support |
| **WebAssembly SIMD** | 91+ | ✅ Full | High | Vectorized operations |

## Feature Detection / 機能検出

### Runtime Detection / ランタイム検出

```javascript
// Comprehensive browser capability detection
async function detectBrowserCapabilities() {
    const capabilities = {
        wasm: typeof WebAssembly !== 'undefined',
        webgpu: !!navigator.gpu,
        sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
        simd: await checkSIMDSupport(),
        threads: await checkThreadSupport()
    };
    
    console.log('Browser capabilities:', capabilities);
    return capabilities;
}

async function checkSIMDSupport() {
    try {
        await WebAssembly.instantiate(new Uint8Array([
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b
        ]));
        return true;
    } catch {
        return false;
    }
}

async function checkThreadSupport() {
    return typeof Worker !== 'undefined' && 
           typeof SharedArrayBuffer !== 'undefined';
}
```

### WebGPU Capability Detection / WebGPU機能検出

```javascript
async function getWebGPUCapabilities() {
    if (!navigator.gpu) {
        return { supported: false, reason: 'WebGPU not available' };
    }
    
    try {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        
        if (!adapter) {
            return { supported: false, reason: 'No adapter found' };
        }
        
        const device = await adapter.requestDevice();
        const limits = device.limits;
        
        return {
            supported: true,
            adapter: {
                vendor: adapter.info?.vendor || 'Unknown',
                architecture: adapter.info?.architecture || 'Unknown',
                device: adapter.info?.device || 'Unknown'
            },
            limits: {
                maxTextureDimension1D: limits.maxTextureDimension1D,
                maxTextureDimension2D: limits.maxTextureDimension2D,
                maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
                maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup
            }
        };
    } catch (error) {
        return { 
            supported: false, 
            reason: `WebGPU initialization failed: ${error.message}` 
        };
    }
}
```

## Performance Characteristics / パフォーマンス特性

### Browser Performance Comparison / ブラウザパフォーマンス比較

| Operation | Chrome WebGPU | Chrome CPU | Firefox CPU | Safari CPU |
|-----------|---------------|------------|-------------|------------|
| **Vector Add (1K)** | 0.1ms | 0.2ms | 0.3ms | 0.25ms |
| **Matrix Mul (256x256)** | 2ms | 20ms | 25ms | 18ms |
| **Activation (10K)** | 0.5ms | 1.5ms | 2ms | 1.8ms |
| **Special Functions** | N/A | 0.8ms | 1.2ms | 1ms |

### Memory Usage / メモリ使用量

```javascript
// Monitor memory usage across browsers
function monitorMemoryUsage() {
    if (performance.memory) {
        // Chrome/Edge specific
        const memory = performance.memory;
        return {
            used: memory.usedJSHeapSize / 1024 / 1024,
            total: memory.totalJSHeapSize / 1024 / 1024,
            limit: memory.jsHeapSizeLimit / 1024 / 1024
        };
    } else {
        // Firefox/Safari fallback
        return { estimated: 'unavailable' };
    }
}
```

## Loading Strategies / ロード戦略

### Lazy Loading / 遅延読み込み

```javascript
// Load modules on demand
class LazyRusTorchLoader {
    constructor() {
        this.modules = new Map();
    }
    
    async loadCore() {
        if (!this.modules.has('core')) {
            const module = await import('./pkg/rustorch.js');
            await module.default(); // Initialize WASM
            this.modules.set('core', module);
        }
        return this.modules.get('core');
    }
    
    async loadWebGPU() {
        const core = await this.loadCore();
        return {
            WebGPUSimple: core.WebGPUSimple,
            WebGPUSimpleDemo: core.WebGPUSimpleDemo
        };
    }
    
    async loadVision() {
        const core = await this.loadCore();
        return {
            WasmVision: core.WasmVision,
            WasmPreprocessor: core.WasmPreprocessor
        };
    }
}
```

### Progressive Loading / プログレッシブロード

```javascript
// Load features progressively based on capability
async function initializeProgressively() {
    const loader = new LazyRusTorchLoader();
    
    // Always load core
    const core = await loader.loadCore();
    console.log('✅ Core WASM loaded');
    
    // Load WebGPU if supported
    if (await checkWebGPUSupport()) {
        const webgpu = await loader.loadWebGPU();
        console.log('✅ WebGPU acceleration loaded');
        return { core, webgpu };
    }
    
    console.log('ℹ️ Using CPU-only mode');
    return { core };
}
```

## Testing Across Browsers / ブラウザ横断テスト

### Automated Testing / 自動テスト

```bash
# Test all supported browsers
wasm-pack test --chrome --headless --features webgpu
wasm-pack test --firefox --headless --features wasm
wasm-pack test --safari --headless --features wasm # Requires macOS
```

### Manual Testing Checklist / 手動テストチェックリスト

#### Chrome Testing / Chromeテスト
- [ ] WebGPU initialization successful
- [ ] All tensor operations working
- [ ] Performance benchmarks show expected speedup
- [ ] Memory usage within expected bounds
- [ ] No console errors or warnings

#### Firefox Testing / Firefoxテスト
- [ ] WASM modules load correctly
- [ ] CPU fallback works when WebGPU disabled
- [ ] All basic operations functional
- [ ] No security policy violations

#### Safari Testing / Safariテスト
- [ ] WASM loads without SharedArrayBuffer warnings
- [ ] CPU operations perform well
- [ ] SIMD optimizations active
- [ ] Memory usage stable

### Cross-Browser Test Suite / ブラウザ横断テストスイート

```javascript
// Automated browser testing framework
class BrowserTestSuite {
    constructor() {
        this.results = new Map();
    }
    
    async runFullSuite() {
        const tests = [
            'testWASMLoading',
            'testTensorOperations', 
            'testWebGPUFallback',
            'testMemoryManagement',
            'testPerformanceBenchmarks'
        ];
        
        for (const test of tests) {
            try {
                const result = await this[test]();
                this.results.set(test, { status: 'passed', result });
            } catch (error) {
                this.results.set(test, { status: 'failed', error: error.message });
            }
        }
        
        return this.generateReport();
    }
    
    generateReport() {
        const passed = Array.from(this.results.values()).filter(r => r.status === 'passed').length;
        const total = this.results.size;
        
        return {
            summary: `${passed}/${total} tests passed`,
            details: Object.fromEntries(this.results),
            browser: this.detectBrowser(),
            timestamp: new Date().toISOString()
        };
    }
}
```

## Migration Guide / 移行ガイド

### From CPU-only to WebGPU / CPUのみからWebGPUへ

```javascript
// Before: CPU-only implementation
const tensor = new WasmTensor([1, 2, 3, 4], [2, 2]);
const result = WasmTensorOps.add(tensor, tensor);

// After: WebGPU-enhanced implementation
const engine = new WebGPUSimple();
await engine.initialize();

const a = [1, 2, 3, 4];
const b = [1, 2, 3, 4];
const result = engine.tensor_add_cpu(a, b); // Auto-fallback to CPU if needed
```

### Legacy Browser Support / レガシーブラウザサポート

```javascript
// Polyfill for older browsers
if (!globalThis.structuredClone) {
    globalThis.structuredClone = (obj) => JSON.parse(JSON.stringify(obj));
}

// Feature detection and graceful degradation
function createCompatibleEngine() {
    if (window.WebGPUSimple) {
        return new WebGPUSimple();
    } else if (window.WasmTensor) {
        return new LegacyCPUEngine();
    } else {
        throw new Error('No compatible ML engine available');
    }
}
```

---

**📝 This guide covers comprehensive browser compatibility for RusTorch WASM deployment**  
**このガイドでは、RusTorch WASM デプロイメントの包括的なブラウザ互換性を説明しています**