# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Una libreria di deep learning pronta per la produzione in Rust con API simile a PyTorch, accelerazione GPU e prestazioni di livello enterprise**

RusTorch è una libreria di deep learning completamente funzionale che sfrutta la sicurezza e le prestazioni di Rust, fornendo operazioni tensoriali complete, differenziazione automatica, strati di reti neurali, architetture transformer, accelerazione GPU multi-backend (CUDA/Metal/OpenCL), ottimizzazioni SIMD avanzate, gestione della memoria di livello enterprise, validazione dati e garanzia di qualità, e sistemi completi di debug e logging.

## ✨ Funzionalità

- 🔥 **Operazioni Tensoriali Complete**: Operazioni matematiche, broadcasting, indicizzazione e statistiche, utilità avanzate Phase 8
- 🤖 **Architettura Transformer**: Implementazione completa di transformer con attenzione multi-head
- 🧮 **Decomposizione Matriciale**: SVD, QR, decomposizione autovalori con compatibilità PyTorch
- 🧠 **Differenziazione Automatica**: Grafo computazionale basato su tape per il calcolo del gradiente
- 🚀 **Motore di Esecuzione Dinamico**: Compilazione JIT e ottimizzazione runtime
- 🏗️ **Strati di Rete Neurale**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, e altro
- ⚡ **Ottimizzazioni Multi-Piattaforma**: SIMD (AVX2/SSE/NEON), specifiche per piattaforma e ottimizzazioni hardware-aware
- 🎮 **Integrazione GPU**: Supporto CUDA/Metal/OpenCL con selezione automatica dispositivo
- 🌐 **Supporto WebAssembly**: ML completo su browser con strati di rete neurale, computer vision e inferenza in tempo reale
- 🎮 **Integrazione WebGPU**: Accelerazione GPU ottimizzata per Chrome con fallback CPU per compatibilità cross-browser
- 📁 **Supporto Formati Modello**: Safetensors, inferenza ONNX, compatibilità state dict PyTorch
- ✅ **Pronto per Produzione**: 968 test superati, sistema di gestione errori unificato
- 📐 **Funzioni Matematiche Avanzate**: Set completo di funzioni matematiche (exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **Overload Operatori Avanzati**: Supporto completo operatori per tensori con operazioni scalari e assegnazioni in-place
- 📈 **Ottimizzatori Avanzati**: SGD, Adam, AdamW, RMSprop, AdaGrad con scheduler learning rate
- 🔍 **Validazione Dati e Garanzia Qualità**: Analisi statistica, rilevamento anomalie, controllo consistenza, monitoraggio real-time
- 🐛 **Debug e Logging Completi**: Logging strutturato, profiling prestazioni, tracking memoria, alert automatizzati
- 🎯 **Utilità Tensor Phase 8**: Operazioni condizionali (where, masked_select, masked_fill), operazioni di indicizzazione (gather, scatter, index_select), operazioni statistiche (topk, kthvalue), e utilità avanzate (unique, histogram)

## 🚀 Avvio Rapido

**📓 Per la guida completa alla configurazione di Jupyter, vedere [README_JUPYTER.md](../../README_JUPYTER.md)**

### Demo Python Jupyter Lab

📓 **[Guida Completa Jupyter](../../README_JUPYTER.md)** | **[Guida Jupyter](jupyter-guide.md)**

#### Demo CPU Standard
Lancia RusTorch con Jupyter Lab in un comando:

```bash
./start_jupyter.sh
```

#### Demo Accelerata WebGPU
Lancia RusTorch con supporto WebGPU per accelerazione GPU basata su browser:

```bash
./start_jupyter_webgpu.sh
```

Entrambi gli script:
- 📦 Creano ambiente virtuale automaticamente
- 🔧 Costruiscono i binding Python RusTorch
- 🚀 Lanciano Jupyter Lab con notebook demo
- 📍 Aprono notebook demo pronto per l'esecuzione

**Funzionalità WebGPU:**
- 🌐 Accelerazione GPU basata su browser
- ⚡ Operazioni matriciali ad alte prestazioni nel browser
- 🔄 Fallback automatico a CPU quando GPU non disponibile
- 🎯 Ottimizzato Chrome/Edge (browser raccomandati)

#### Kernel Rust per Jupyter
Lancia il kernel Rust nativo in Jupyter (evcxr_jupyter):

```bash
./quick_start_rust_kernel.sh
```

Questo:
- 🦀 Installa il kernel Rust evcxr_jupyter
- 📓 Crea notebook demo kernel Rust
- 🚀 Lancia Jupyter con supporto Rust nativo
- 📍 Operazioni tensoriali dirette in Rust

### Installazione

Aggiungi questo al tuo `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.5.10"

# Funzionalità opzionali
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Operazioni algebra lineare (SVD, QR, autovalori)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # Supporto WebAssembly per ML browser
webgpu = ["rustorch/webgpu"]            # Accelerazione WebGPU ottimizzata Chrome

# Per disabilitare funzionalità linalg (evitare dipendenze OpenBLAS/LAPACK):
rustorch = { version = "0.5.10", default-features = false }
```

### Uso Base

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // Creare tensori
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Operazioni base con overload operatori
    let c = &a + &b;  // Addizione elemento per elemento
    let d = &a - &b;  // Sottrazione elemento per elemento
    let e = &a * &b;  // Moltiplicazione elemento per elemento
    let f = &a / &b;  // Divisione elemento per elemento
    
    // Operazioni scalari
    let g = &a + 10.0;  // Aggiungere scalare a tutti gli elementi
    let h = &a * 2.0;   // Moltiplicare per scalare
    
    // Funzioni matematiche
    let exp_result = a.exp();   // Funzione esponenziale
    let ln_result = a.ln();     // Logaritmo naturale
    let sin_result = a.sin();   // Funzione seno
    let sqrt_result = a.sqrt(); // Radice quadrata
    
    // Operazioni matriciali
    let matmul_result = a.matmul(&b);  // Moltiplicazione matriciale
    
    // Operazioni algebra lineare (richiede feature linalg)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // Decomposizione SVD
        let qr_result = a.qr();         // Decomposizione QR
        let eig_result = a.eigh();      // Decomposizione autovalori
    }
    
    // Ottimizzatori avanzati con scheduling learning rate
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Warmup a 0.1 su 5 epoche
    
    println!("Forma: {:?}", c.shape());
    println!("Risultato: {:?}", c.as_slice());
}
```

### Uso WebAssembly

Per applicazioni ML basate su browser:

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // Strati rete neurale
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // Funzioni matematiche avanzate
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // Distribuzioni statistiche
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // Ottimizzatori per training
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // learning_rate, momentum
    
    // Elaborazione immagini
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // Forward pass
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('Predizioni ML browser:', predictions);
}
```

## 📚 Documentazione

- **[Guida Iniziale](../getting-started.md)** - Uso base ed esempi
- **[Funzionalità](../features.md)** - Lista completa funzionalità e specifiche
- **[Prestazioni](../performance.md)** - Benchmark e dettagli ottimizzazione
- **[Guida Jupyter WASM](jupyter-guide.md)** - Setup passo-passo Jupyter Notebook

### WebAssembly e ML Browser
- **[Guida WebAssembly](../WASM_GUIDE.md)** - Integrazione WASM completa e riferimento API
- **[Integrazione WebGPU](../WEBGPU_INTEGRATION.md)** - Accelerazione GPU ottimizzata Chrome

### Produzione e Operazioni
- **[Guida Accelerazione GPU](../GPU_ACCELERATION_GUIDE.md)** - Setup e uso GPU
- **[Guida Produzione](../PRODUCTION_GUIDE.md)** - Deployment e scaling

## 📊 Prestazioni

**Risultati benchmark recenti:**

| Operazione | Prestazioni | Dettagli |
|-----------|-------------|----------|
| **Decomposizione SVD** | ~1ms (matrice 8x8) | ✅ Basato su LAPACK |
| **Decomposizione QR** | ~24μs (matrice 8x8) | ✅ Decomposizione veloce |
| **Autovalori** | ~165μs (matrice 8x8) | ✅ Matrici simmetriche |
| **FFT Complessa** | 10-312μs (8-64 campioni) | ✅ Ottimizzata Cooley-Tukey |
| **Rete Neurale** | 1-7s training | ✅ Demo Boston housing |
| **Funzioni Attivazione** | <1μs | ✅ ReLU, Sigmoid, Tanh |

## 🧪 Testing

**968 test superati** - Garanzia qualità pronta per produzione con sistema gestione errori unificato.

```bash
# Eseguire tutti i test
cargo test --no-default-features

# Eseguire test con funzionalità algebra lineare
cargo test --features linalg
```

## 🤝 Contribuire

Accogliamo contributi! Vedi aree dove serve particolarmente aiuto:

- **🎯 Precisione Funzioni Speciali**: Migliorare accuratezza numerica
- **⚡ Ottimizzazione Prestazioni**: Miglioramenti SIMD, ottimizzazione GPU
- **🧪 Testing**: Casi test più completi
- **📚 Documentazione**: Esempi, tutorial, miglioramenti
- **🌐 Supporto Piattaforme**: WebAssembly, piattaforme mobile

## Licenza

Licenziato sotto:

 * Licenza Apache, Versione 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) o http://www.apache.org/licenses/LICENSE-2.0)
 * Licenza MIT ([LICENSE-MIT](../../LICENSE-MIT) o http://opensource.org/licenses/MIT)

a tua scelta.