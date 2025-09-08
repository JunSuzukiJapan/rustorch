# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-1128%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Produktionsbereite Deep Learning Bibliothek in Rust mit PyTorch-ähnlicher API, GPU-Beschleunigung und Enterprise-Performance**

RusTorch ist eine voll funktionsfähige Deep Learning Bibliothek, die Rusts Sicherheit und Performance nutzt und umfassende Tensor-Operationen, automatische Differenzierung, neurale Netzwerk-Layer, Transformer-Architekturen, Multi-Backend GPU-Beschleunigung (CUDA/Metal/OpenCL), erweiterte SIMD-Optimierungen, Enterprise-Level Speicherverwaltung, Datenvalidierung und Qualitätssicherung sowie umfassende Debug- und Logging-Systeme bietet.

## 📚 Dokumentation

- **[Vollständige API-Referenz](API_DOCUMENTATION.md)** - Umfassende API-Dokumentation für alle Module
- **[WASM API-Referenz](WASM_API_DOCUMENTATION.md)** - WebAssembly-spezifische API-Dokumentation
- **[Jupyter Leitfaden](jupyter-guide.md)** - Anleitung zur Verwendung von Jupyter Notebooks

## ✨ Features

- 🔥 **Umfassende Tensor-Operationen**: Mathematische Operationen, Broadcasting, Indizierung und Statistiken, Phase 8 erweiterte Utilities
- 🤖 **Transformer-Architektur**: Vollständige Transformer-Implementierung mit Multi-Head-Attention
- 🧮 **Matrix-Zerlegung**: SVD, QR, Eigenwert-Zerlegung mit PyTorch-Kompatibilität
- 🧠 **Automatische Differenzierung**: Tape-basierter Berechnungsgraph für Gradientenberechnung
- 🚀 **Dynamische Ausführungs-Engine**: JIT-Kompilation und Laufzeit-Optimierung
- 🏗️ **Neurale Netzwerk-Layer**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout und mehr
- ⚡ **Cross-Platform-Optimierungen**: SIMD (AVX2/SSE/NEON), plattformspezifische und hardware-bewusste Optimierungen
- 🎮 **GPU-Integration**: CUDA/Metal/OpenCL-Support mit automatischer Gerätewahl
- 🌐 **WebAssembly-Support**: Vollständiges Browser-ML mit neuronalen Netzwerk-Layern, Computer Vision und Echtzeit-Inferenz
- 🎮 **WebGPU-Integration**: Chrome-optimierte GPU-Beschleunigung mit CPU-Fallback für Cross-Browser-Kompatibilität
- 📁 **Modellformat-Support**: Safetensors, ONNX-Inferenz, PyTorch state dict Kompatibilität
- ✅ **Produktionsbereit**: 1128 Tests bestanden, einheitliches Fehlerbehandlungssystem
- 🎯 **Phase 8 Tensor-Utilities**: Bedingte Operationen (where, masked_select, masked_fill), Indizierungs-Operationen (gather, scatter, index_select), statistische Operationen (topk, kthvalue) und erweiterte Utilities (unique, histogram)

## 🚀 Schnellstart

**📓 Für die vollständige Jupyter-Setup-Anleitung siehe [README_JUPYTER.md](../../README_JUPYTER.md)**

### Python Jupyter Lab Demo

📓 **[Vollständige Jupyter-Setup-Anleitung](../../README_JUPYTER.md)** | **[Jupyter Leitfaden](jupyter-guide.md)**

#### Standard CPU Demo
RusTorch mit Jupyter Lab in einem Befehl starten:

```bash
./start_jupyter.sh
```

#### WebGPU-beschleunigte Demo
RusTorch mit WebGPU-Support für browser-basierte GPU-Beschleunigung starten:

```bash
./start_jupyter_webgpu.sh
```

### Rust-Verwendung

```rust
use rustorch::tensor::Tensor;
use rustorch::nn::{Linear, ReLU};
use rustorch::optim::Adam;

// Tensor-Erstellung
let x = Tensor::randn(vec![32, 784]); // Batch-Größe 32, Features 784
let y = Tensor::randn(vec![32, 10]);  // 10 Klassen

// Neurales Netzwerk definieren
let linear1 = Linear::new(784, 256)?;
let relu = ReLU::new();
let linear2 = Linear::new(256, 10)?;

// Vorwärtsdurchgang
let z1 = linear1.forward(&x)?;
let a1 = relu.forward(&z1)?;
let output = linear2.forward(&a1)?;

// Optimierer
let mut optimizer = Adam::new(
    vec![linear1.weight(), linear2.weight()], 
    0.001, 0.9, 0.999, 1e-8
)?;
```

## 🧪 Tests

### Alle Tests ausführen
```bash
cargo test --lib
```

### Feature-spezifische Tests
```bash
cargo test tensor::     # Tensor-Operationen Tests
cargo test nn::         # Neurale Netzwerk Tests
cargo test autograd::   # Automatische Differenzierung Tests
cargo test optim::      # Optimierer Tests
cargo test gpu::        # GPU-Operationen Tests
```

## 🔧 Installation

### Cargo.toml
```toml
[dependencies]
rustorch = "0.5.15"

# GPU Features
rustorch = { version = "0.5.15", features = ["cuda"] }      # CUDA
rustorch = { version = "0.5.15", features = ["metal"] }     # Metal (macOS)
rustorch = { version = "0.5.15", features = ["opencl"] }    # OpenCL

# WebAssembly
rustorch = { version = "0.5.15", features = ["wasm"] }      # WASM Basic
rustorch = { version = "0.5.15", features = ["webgpu"] }    # WebGPU
```

## ⚠️ Bekannte Einschränkungen

1. **GPU-Speicher-Beschränkung**: Explizite Speicherverwaltung erforderlich für große Tensoren (>8GB)
2. **WebAssembly-Beschränkung**: Einige BLAS-Operationen nicht verfügbar in WASM-Umgebung
3. **Verteiltes Lernen**: NCCL-Backend nur unter Linux unterstützt
4. **Metal-Beschränkung**: Einige erweiterte Operationen nur mit CUDA-Backend verfügbar

## 🤝 Beitrag

Pull Requests und Issues sind willkommen! Siehe [CONTRIBUTING.md](../../CONTRIBUTING.md) für Details.

## 📄 Lizenz

MIT Lizenz - siehe [LICENSE](../../LICENSE) für Details.

---

**Entwickelt von**: Jun Suzuki | **Version**: v0.5.15 | **Letzte Aktualisierung**: 2025