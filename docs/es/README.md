# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Una biblioteca de deep learning lista para producción en Rust con API similar a PyTorch, aceleración GPU y rendimiento de nivel empresarial**

RusTorch es una biblioteca de deep learning completamente funcional que aprovecha la seguridad y el rendimiento de Rust, proporcionando operaciones tensoriales completas, diferenciación automática, capas de redes neuronales, arquitecturas transformer, aceleración GPU multi-backend (CUDA/Metal/OpenCL), optimizaciones SIMD avanzadas, gestión de memoria de nivel empresarial, validación de datos y aseguramiento de calidad, y sistemas completos de depuración y logging.

## ✨ Características

- 🔥 **Operaciones Tensoriales Completas**: Operaciones matemáticas, broadcasting, indexación y estadísticas, utilidades avanzadas Phase 8
- 🤖 **Arquitectura Transformer**: Implementación completa de transformer con atención multi-head
- 🧮 **Descomposición Matricial**: SVD, QR, descomposición de autovalores con compatibilidad PyTorch
- 🧠 **Diferenciación Automática**: Grafo computacional basado en cinta para cálculo de gradientes
- 🚀 **Motor de Ejecución Dinámico**: Compilación JIT y optimización en tiempo de ejecución
- 🏗️ **Capas de Red Neuronal**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, y más
- ⚡ **Optimizaciones Multi-Plataforma**: SIMD (AVX2/SSE/NEON), específicas de plataforma y optimizaciones hardware-aware
- 🎮 **Integración GPU**: Soporte CUDA/Metal/OpenCL con selección automática de dispositivo
- 🌐 **Soporte WebAssembly**: ML completo en navegador con capas de red neuronal, visión computacional e inferencia en tiempo real
- 🎮 **Integración WebGPU**: Aceleración GPU optimizada para Chrome con fallback CPU para compatibilidad cross-browser
- 📁 **Soporte Formatos de Modelo**: Safetensors, inferencia ONNX, compatibilidad state dict PyTorch
- ✅ **Listo para Producción**: 968 tests aprobados, sistema de manejo de errores unificado
- 📐 **Funciones Matemáticas Mejoradas**: Conjunto completo de funciones matemáticas (exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **Sobrecarga de Operadores Avanzada**: Soporte completo de operadores para tensores con operaciones escalares y asignaciones in-place
- 📈 **Optimizadores Avanzados**: SGD, Adam, AdamW, RMSprop, AdaGrad con programadores de tasa de aprendizaje
- 🔍 **Validación de Datos y Aseguramiento de Calidad**: Análisis estadístico, detección de anomalías, verificación de consistencia, monitoreo en tiempo real
- 🐛 **Depuración y Logging Completos**: Logging estructurado, profiling de rendimiento, seguimiento de memoria, alertas automatizadas
- 🎯 **Utilidades de Tensor Phase 8**: Operaciones condicionales (where, masked_select, masked_fill), operaciones de indexación (gather, scatter, index_select), operaciones estadísticas (topk, kthvalue), y utilidades avanzadas (unique, histogram)

## 🚀 Inicio Rápido

**📓 Para la guía completa de configuración de Jupyter, ver [README_JUPYTER.md](../../README_JUPYTER.md)**

### Demo Python Jupyter Lab

📓 **[Guía Completa Jupyter](../../README_JUPYTER.md)** | **[Guía Jupyter](jupyter-guide.md)**

#### Demo CPU Estándar
Lanza RusTorch con Jupyter Lab en un comando:

```bash
./start_jupyter.sh
```

#### Demo Acelerada WebGPU
Lanza RusTorch con soporte WebGPU para aceleración GPU basada en navegador:

```bash
./start_jupyter_webgpu.sh
```

Ambos scripts harán:
- 📦 Crear entorno virtual automáticamente
- 🔧 Construir bindings Python de RusTorch
- 🚀 Lanzar Jupyter Lab con notebook demo
- 📍 Abrir notebook demo listo para ejecutar

**Características WebGPU:**
- 🌐 Aceleración GPU basada en navegador
- ⚡ Operaciones matriciales de alto rendimiento en navegador
- 🔄 Fallback automático a CPU cuando GPU no disponible
- 🎯 Optimizado Chrome/Edge (navegadores recomendados)

#### Kernel Rust para Jupyter
Lanza el kernel Rust nativo en Jupyter (evcxr_jupyter):

```bash
./quick_start_rust_kernel.sh
```

Esto:
- 🦀 Instala el kernel Rust evcxr_jupyter
- 📓 Crea notebook demo del kernel Rust
- 🚀 Lanza Jupyter con soporte Rust nativo
- 📍 Operaciones tensoriales directas en Rust

### Instalación

Añade esto a tu `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.5.10"

# Características opcionales
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Operaciones álgebra lineal (SVD, QR, autovalores)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # Soporte WebAssembly para ML navegador
webgpu = ["rustorch/webgpu"]            # Aceleración WebGPU optimizada Chrome

# Para desactivar características linalg (evitar dependencias OpenBLAS/LAPACK):
rustorch = { version = "0.5.10", default-features = false }
```

### Uso Básico

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // Crear tensores
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Operaciones básicas con sobrecarga de operadores
    let c = &a + &b;  // Suma elemento por elemento
    let d = &a - &b;  // Resta elemento por elemento
    let e = &a * &b;  // Multiplicación elemento por elemento
    let f = &a / &b;  // División elemento por elemento
    
    // Operaciones escalares
    let g = &a + 10.0;  // Añadir escalar a todos los elementos
    let h = &a * 2.0;   // Multiplicar por escalar
    
    // Funciones matemáticas
    let exp_result = a.exp();   // Función exponencial
    let ln_result = a.ln();     // Logaritmo natural
    let sin_result = a.sin();   // Función seno
    let sqrt_result = a.sqrt(); // Raíz cuadrada
    
    // Operaciones matriciales
    let matmul_result = a.matmul(&b);  // Multiplicación matricial
    
    // Operaciones álgebra lineal (requiere característica linalg)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // Descomposición SVD
        let qr_result = a.qr();         // Descomposición QR
        let eig_result = a.eigh();      // Descomposición autovalores
    }
    
    // Optimizadores avanzados con programación tasa de aprendizaje
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Warmup a 0.1 sobre 5 épocas
    
    println!("Forma: {:?}", c.shape());
    println!("Resultado: {:?}", c.as_slice());
}
```

### Uso WebAssembly

Para aplicaciones ML basadas en navegador:

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // Capas red neuronal
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // Funciones matemáticas mejoradas
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // Distribuciones estadísticas
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // Optimizadores para entrenamiento
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // tasa_aprendizaje, momento
    
    // Procesamiento de imágenes
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // Forward pass
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('Predicciones ML navegador:', predictions);
}
```

## 📚 Documentación

- **[Guía de Inicio](../getting-started.md)** - Uso básico y ejemplos
- **[Características](../features.md)** - Lista completa de características y especificaciones
- **[Rendimiento](../performance.md)** - Benchmarks y detalles de optimización
- **[Guía Jupyter WASM](jupyter-guide.md)** - Configuración paso a paso de Jupyter Notebook

### WebAssembly y ML Navegador
- **[Guía WebAssembly](../WASM_GUIDE.md)** - Integración WASM completa y referencia API
- **[Integración WebGPU](../WEBGPU_INTEGRATION.md)** - Aceleración GPU optimizada Chrome

### Producción y Operaciones
- **[Guía Aceleración GPU](../GPU_ACCELERATION_GUIDE.md)** - Configuración y uso GPU
- **[Guía Producción](../PRODUCTION_GUIDE.md)** - Despliegue y escalado

## 📊 Rendimiento

**Resultados de benchmarks recientes:**

| Operación | Rendimiento | Detalles |
|-----------|-------------|----------|
| **Descomposición SVD** | ~1ms (matriz 8x8) | ✅ Basado en LAPACK |
| **Descomposición QR** | ~24μs (matriz 8x8) | ✅ Descomposición rápida |
| **Autovalores** | ~165μs (matriz 8x8) | ✅ Matrices simétricas |
| **FFT Compleja** | 10-312μs (8-64 muestras) | ✅ Optimizada Cooley-Tukey |
| **Red Neuronal** | 1-7s entrenamiento | ✅ Demo Boston housing |
| **Funciones Activación** | <1μs | ✅ ReLU, Sigmoid, Tanh |

## 🧪 Testing

**968 tests aprobados** - Aseguramiento de calidad listo para producción con sistema de manejo de errores unificado.

```bash
# Ejecutar todos los tests
cargo test --no-default-features

# Ejecutar tests con características álgebra lineal
cargo test --features linalg
```

## 🤝 Contribuir

¡Damos la bienvenida a contribuciones! Ve áreas donde se necesita especialmente ayuda:

- **🎯 Precisión Funciones Especiales**: Mejorar precisión numérica
- **⚡ Optimización Rendimiento**: Mejoras SIMD, optimización GPU
- **🧪 Testing**: Casos de test más completos
- **📚 Documentación**: Ejemplos, tutoriales, mejoras
- **🌐 Soporte Plataformas**: WebAssembly, plataformas móviles

## Licencia

Licenciado bajo cualquiera de:

 * Licencia Apache, Versión 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) o http://www.apache.org/licenses/LICENSE-2.0)
 * Licencia MIT ([LICENSE-MIT](../../LICENSE-MIT) o http://opensource.org/licenses/MIT)

a tu elección.