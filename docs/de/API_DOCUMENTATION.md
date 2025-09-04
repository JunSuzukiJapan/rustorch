# RusTorch API Dokumentation

## 📚 Vollständige API-Referenz

Dieses Dokument bietet umfassende API-Dokumentation für RusTorch v0.5.15, organisiert nach Modulen und Funktionalität. Es umfasst einheitliche Fehlerbehandlung mit `RusTorchError` und `RusTorchResult<T>` für konsistente Fehlerverwaltung über alle 1060+ Tests. **Phase 8 ABGESCHLOSSEN** fügt erweiterte Tensor-Utilities hinzu, einschließlich bedingter Operationen, Indizierung und statistischer Funktionen. **Phase 9 ABGESCHLOSSEN** führt umfassendes Serialisierungssystem ein mit Modell-Speichern/Laden, JIT-Kompilation und Unterstützung mehrerer Formate einschließlich PyTorch-Kompatibilität.

## 🏗️ Kern-Architektur

### Modul-Struktur

```
rustorch/
├── tensor/              # Kern-Tensor-Operationen und Datenstrukturen
├── nn/                  # Neuronale Netzwerk-Layer und Funktionen
├── autograd/            # Automatische Differenzierungsengine
├── optim/               # Optimierer und Lernraten-Scheduler
├── special/             # Spezielle mathematische Funktionen
├── distributions/       # Statistische Verteilungen
├── vision/              # Computer Vision Transformationen
├── linalg/              # Lineare Algebra Operationen (BLAS/LAPACK)
├── gpu/                 # GPU-Beschleunigung (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Sparse-Tensor-Operationen und Pruning (Phase 12)
├── serialization/       # Modell-Serialisierung und JIT-Kompilation (Phase 9)
└── wasm/                # WebAssembly-Bindings (siehe [WASM API Dokumentation](WASM_API_DOCUMENTATION.md))
```

## 📊 Tensor-Modul

### Grundlegende Tensor-Erstellung

```rust
use rustorch::tensor::Tensor;

// Grundlegende Erstellung
let tensor = Tensor::new(vec![2, 3]);               // Formbasierte Erstellung
let tensor = Tensor::from_vec(data, vec![2, 3]);    // Aus Daten-Vektor
let tensor = Tensor::zeros(vec![10, 10]);           // Mit Nullen gefüllter Tensor
let tensor = Tensor::ones(vec![5, 5]);              // Mit Einsen gefüllter Tensor
let tensor = Tensor::randn(vec![3, 3]);             // Zufällige Normalverteilung
let tensor = Tensor::rand(vec![3, 3]);              // Zufällige Gleichverteilung [0,1)
let tensor = Tensor::eye(5);                        // Einheitsmatrix
let tensor = Tensor::full(vec![2, 2], 3.14);       // Mit spezifischem Wert füllen
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Bereichs-Tensor
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Lineare Abstände
```

### Tensor-Operationen

```rust
// Arithmetische Operationen
let result = a.add(&b);                             // Elementweise Addition
let result = a.sub(&b);                             // Elementweise Subtraktion
let result = a.mul(&b);                             // Elementweise Multiplikation
let result = a.div(&b);                             // Elementweise Division
let result = a.pow(&b);                             // Elementweise Potenzierung
let result = a.rem(&b);                             // Elementweiser Rest

// Matrix-Operationen
let result = a.matmul(&b);                          // Matrix-Multiplikation
let result = a.transpose();                         // Matrix-Transposition
let result = a.dot(&b);                             // Skalarprodukt

// Mathematische Funktionen
let result = tensor.exp();                          // Exponential
let result = tensor.ln();                           // Natürlicher Logarithmus
let result = tensor.log10();                        // Logarithmus zur Basis 10
let result = tensor.sqrt();                         // Quadratwurzel
let result = tensor.abs();                          // Absolutwert
let result = tensor.sin();                          // Sinus-Funktion
let result = tensor.cos();                          // Kosinus-Funktion
let result = tensor.tan();                          // Tangens-Funktion
let result = tensor.asin();                         // Arkussinus
let result = tensor.acos();                         // Arkuskosinus
let result = tensor.atan();                         // Arkustangens
let result = tensor.sinh();                         // Hyperbolischer Sinus
let result = tensor.cosh();                         // Hyperbolischer Kosinus
let result = tensor.tanh();                         // Hyperbolischer Tangens
let result = tensor.floor();                        // Boden-Funktion
let result = tensor.ceil();                         // Decken-Funktion
let result = tensor.round();                        // Rundungs-Funktion
let result = tensor.sign();                         // Vorzeichen-Funktion
let result = tensor.max();                          // Maximalwert
let result = tensor.min();                          // Minimalwert
let result = tensor.sum();                          // Summe aller Elemente
let result = tensor.mean();                         // Mittelwert
let result = tensor.std();                          // Standardabweichung
let result = tensor.var();                          // Varianz

// Form-Manipulation
let result = tensor.reshape(vec![6, 4]);            // Tensor umformen
let result = tensor.squeeze();                      // Größe-1-Dimensionen entfernen
let result = tensor.unsqueeze(1);                   // Dimension am Index hinzufügen
let result = tensor.permute(vec![1, 0, 2]);         // Dimensionen permutieren
let result = tensor.expand(vec![10, 10, 5]);        // Tensor-Dimensionen erweitern
```

## 🧠 Neural Network (nn) Modul

### Basis-Layer

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// Linearer Layer
let linear = Linear::new(784, 256)?;                // Eingabe 784, Ausgabe 256
let output = linear.forward(&input)?;

// Faltungs-Layer
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// Batch-Normalisierung
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### Aktivierungsfunktionen

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// Grundlegende Aktivierungsfunktionen
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Parametrisierte Aktivierungsfunktionen
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// Anwendungsbeispiel
let activated = relu.forward(&input)?;
```

## 🚀 GPU-Beschleunigungs-Modul

### Geräteverwaltung

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// Verfügbare Geräte prüfen
let device_count = get_device_count()?;
let device = Device::best_available()?;            // Bestes Gerät auswählen

// Gerätekonfiguration
set_device(&device)?;

// Tensor zu GPU bewegen
let gpu_tensor = tensor.to_device(&device)?;
```

### CUDA-Operationen

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// CUDA-Geräteoperationen
let cuda_device = CudaDevice::new(0)?;              // GPU 0 verwenden
let stats = memory_stats(0)?;                      // Speicher-Statistiken
println!("Verwendeter Speicher: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 Optimierer (Optim) Modul

### Basis-Optimierer

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Adam-Optimierer
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// SGD-Optimierer
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// Optimierungsschritt
optimizer.zero_grad()?;
// ... Vorwärtsberechnung und Rückpropagation ...
optimizer.step()?;
```

## 📖 Anwendungsbeispiel

### Lineare Regression

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// Datenvorbereitung
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// Modelldefinition
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// Trainingsschleife
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("Epoche {}: Verlust = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ Bekannte Einschränkungen

1. **GPU-Speicher-Beschränkung**: Explizite Speicherverwaltung erforderlich für große Tensoren (>8GB)
2. **WebAssembly-Beschränkung**: Einige BLAS-Operationen nicht verfügbar in WASM-Umgebung
3. **Verteiltes Lernen**: NCCL-Backend nur unter Linux unterstützt
4. **Metal-Beschränkung**: Einige erweiterte Operationen nur mit CUDA-Backend verfügbar

## 🔗 Verwandte Links

- [Haupt-README](../README.md)
- [WASM API Dokumentation](WASM_API_DOCUMENTATION.md)
- [Jupyter-Leitfaden](jupyter-guide.md)
- [GitHub Repository](https://github.com/JunSuzukiJapan/RusTorch)
- [Crates.io Paket](https://crates.io/crates/rustorch)

---

**Letztes Update**: v0.5.15 | **Lizenz**: MIT | **Autor**: Jun Suzuki