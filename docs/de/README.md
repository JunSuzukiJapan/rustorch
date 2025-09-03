# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Eine produktionsbereite Deep Learning Bibliothek in Rust mit PyTorch-ähnlicher API, GPU-Beschleunigung und Unternehmens-Performance**

RusTorch ist eine vollständig funktionale Deep Learning Bibliothek, die Rusts Sicherheit und Performance nutzt und umfassende Tensoroperationen, automatische Differentiation, neuronale Netzwerkschichten, Transformer-Architekturen, Multi-Backend GPU-Beschleunigung (CUDA/Metal/OpenCL), fortgeschrittene SIMD-Optimierungen, Unternehmens-Memory-Management, Datenvalidierung & Qualitätssicherung sowie umfassende Debug- und Logging-Systeme bietet.

## ✨ Funktionen

- 🔥 **Umfassende Tensoroperationen**: Mathematische Operationen, Broadcasting, Indizierung und Statistiken
- 🤖 **Transformer-Architektur**: Vollständige Transformer-Implementierung mit Multi-Head-Attention
- 🧮 **Matrixzerlegung**: SVD, QR, Eigenwertzerlegung mit PyTorch-Kompatibilität
- 🧠 **Automatische Differentiation**: Bandbasierter Berechnungsgraph für Gradientenberechnung
- 🚀 **Dynamische Ausführungs-Engine**: JIT-Kompilierung und Laufzeitoptimierung
- 🏗️ **Neuronale Netzwerkschichten**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout und mehr
- ⚡ **Plattformübergreifende Optimierungen**: SIMD (AVX2/SSE/NEON), plattformspezifische und hardwarebewusste Optimierungen
- 🎮 **GPU-Integration**: CUDA/Metal/OpenCL-Unterstützung mit automatischer Geräteauswahl
- 🌐 **WebAssembly-Unterstützung**: Vollständiges Browser-ML mit neuronalen Netzwerkschichten, Computer Vision und Echtzeit-Inferenz
- 🎮 **WebGPU-Integration**: Chrome-optimierte GPU-Beschleunigung mit CPU-Fallback für browserübergreifende Kompatibilität
- 📁 **Modellformat-Unterstützung**: Safetensors, ONNX-Inferenz, PyTorch-State-Dict-Kompatibilität
- ✅ **Produktionsbereit**: 968 Tests bestanden, vereinheitlichtes Fehlerbehandlungssystem
- 📐 **Erweiterte mathematische Funktionen**: Vollständiger Satz mathematischer Funktionen (exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **Fortgeschrittene Operatorüberladungen**: Vollständige Operatorunterstützung für Tensoren mit skalaren Operationen und In-Place-Zuweisungen
- 📈 **Fortgeschrittene Optimierer**: SGD, Adam, AdamW, RMSprop, AdaGrad mit Lernraten-Schedulern
- 🔍 **Datenvalidierung & Qualitätssicherung**: Statistische Analyse, Anomalieerkennung, Konsistenzprüfung, Echtzeit-Überwachung
- 🐛 **Umfassendes Debug & Logging**: Strukturiertes Logging, Performance-Profiling, Memory-Tracking, automatisierte Alerts

## 🚀 Schnellstart

**📓 Für die vollständige Jupyter-Setup-Anleitung, siehe [README_JUPYTER.md](../../README_JUPYTER.md)**

### Python Jupyter Lab Demo

📓 **[Vollständige Jupyter-Setup-Anleitung](../../README_JUPYTER.md)** | **[Jupyter-Anleitung](jupyter-guide.md)**

#### Standard-CPU-Demo
RusTorch mit Jupyter Lab in einem Befehl starten:

```bash
./start_jupyter.sh
```

#### WebGPU-beschleunigte Demo
RusTorch mit WebGPU-Unterstützung für browserbasierte GPU-Beschleunigung starten:

```bash
./start_jupyter_webgpu.sh
```

### Grundlegende Installation

```bash
cargo add rustorch
```

### Einfaches Beispiel

```rust
use rustorch::{
    tensor::Tensor,
    nn::{Linear, Module},
    optim::SGD,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Erstelle Tensor
    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let y = Tensor::<f32>::new(vec![0.0, 1.0], vec![2, 1])?;
    
    // Erstelle Modell
    let mut model = Linear::<f32>::new(2, 1)?;
    let mut optimizer = SGD::new(model.parameters(), 0.01)?;
    
    // Training
    for epoch in 0..100 {
        // Forward-Pass
        let pred = model.forward(&x)?;
        let loss = (pred - &y)?.pow(2.0)?.mean()?;
        
        // Backward-Pass
        loss.backward()?;
        optimizer.step()?;
        optimizer.zero_grad()?;
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.data()[0]);
        }
    }
    
    Ok(())
}
```

## 📋 Anforderungen

- **Rust**: Version 1.75.0 oder höher
- **Plattformen**: Windows, macOS, Linux
- **Optional**: CUDA 12.0+, OpenCL 2.0+, Metal (für GPU-Beschleunigung)

## 🎯 Anwendungsfälle

### 🏭 Unternehmensbereit
- ✅ **968 Tests bestanden**: Umfassende Test-Suite gewährleistet Zuverlässigkeit
- 🔒 **Memory-sicher**: Rust verhindert Segfaults und Memory-Leaks
- 📊 **Performance-profiled**: Optimiert für Produktionslasten
- 🌍 **Plattformübergreifend**: Läuft auf Windows, macOS, Linux

### 🧠 Machine Learning
- 🔥 **Transformer-Modelle**: State-of-the-art NLP mit Multi-Head-Attention
- 🖼️ **Computer Vision**: CNN, ResNet, Bildklassifizierung
- 📊 **Datenanalyse**: Statistik, Anomalieerkennung, Zeitreihen
- 🎯 **Reinforcement Learning**: Richtlinien-Gradienten, Q-Learning

### ⚡ High Performance Computing
- 🚀 **SIMD-Optimierungen**: AVX2/SSE/NEON-Vektorisierung
- 🎮 **Multi-GPU**: CUDA/Metal/OpenCL-Parallelisierung
- 🧵 **Multithreading**: Rayon-basierte Parallelverarbeitung
- 💾 **Memory-optimiert**: Efficient Memory-Pooling und Caching

### 🌐 Web & Edge Deployment
- 📱 **WebAssembly**: Vollständige ML-Pipeline im Browser
- ⚡ **WebGPU**: Chrome-optimierte GPU-Beschleunigung
- 🔧 **Edge-Computing**: Eingebettete Systeme und IoT
- 🌍 **Plattformunabhängig**: Ein Code-base, überall lauffähig

## 🏗️ Architektur

RusTorch folgt einer modularen Architektur mit klaren Verantwortlichkeiten:

```
rustorch/
├── tensor/          # Kern-Tensoroperationen
├── autograd/        # Automatische Differentiation
├── nn/              # Neuronale Netzwerkschichten
├── optim/           # Optimierungsalgorithmen
├── data/            # Datenladen und -vorverarbeitung  
├── gpu/             # GPU-Backend (CUDA/Metal/OpenCL)
├── wasm/            # WebAssembly-Bindings
└── examples/        # Beispiele und Demos
```

## 🔧 Features

### Tensor-Operationen
```rust
let a = Tensor::<f32>::ones(vec![3, 3])?;
let b = Tensor::<f32>::eye(3)?;
let c = a.matmul(&b)?;  // Matrix-Multiplikation
let d = c.transpose(0, 1)?;  // Transponierung
```

### GPU-Beschleunigung
```rust
#[cfg(feature = "cuda")]
{
    let device = Device::cuda(0)?;
    let tensor = Tensor::<f32>::ones(vec![1000, 1000])?.to_device(&device)?;
    let result = tensor.matmul(&tensor)?;  // GPU-beschleunigt
}
```

### Transformer-Architektur
```rust
use rustorch::nn::transformer::TransformerEncoder;

let encoder = TransformerEncoder::<f32>::new(
    512,    // d_model
    8,      // num_heads
    2048,   // d_ff
    6,      // num_layers
    0.1,    // dropout
)?;
```

## 📚 Dokumentation

- 📖 **[Vollständige API-Dokumentation](https://docs.rs/rustorch)**
- 📓 **[Jupyter-Setup-Anleitung](../../README_JUPYTER.md)**
- 🎯 **[Beispiele](../../examples/)**
- 🧪 **[Tests](../../tests/)**

## 🤝 Beitragen

Beiträge sind willkommen! Siehe [CONTRIBUTING.md](../../CONTRIBUTING.md) für Details.

## 📄 Lizenz

Dieses Projekt steht unter MIT ODER Apache-2.0 Lizenz - siehe die [LICENSE](../../LICENSE) Dateien für Details.

## 🙏 Danksagungen

- Inspiriert von PyTorch für API-Design
- Rust-Community für außergewöhnliche Werkzeuge
- Candle für Rust-ML-Inspiration
- Alle Mitwirkenden und Tester

---

**🚀 Bereit loszulegen? Schauen Sie sich unsere [Jupyter-Demo](../../README_JUPYTER.md) an!**