# RusTorch Jupyter Vollständiger Leitfaden

Dieser Leitfaden zeigt, wie Sie RusTorch in Jupyter-Notebooks für interaktive Machine Learning- und Deep Learning-Entwicklung verwenden können.

## 📋 Inhaltsverzeichnis

- [Konfiguration](#konfiguration)
- [Installation](#installation)
- [Erstes Notebook](#erstes-notebook)
- [Praktische Beispiele](#praktische-beispiele)
- [Visualisierungen](#visualisierungen)
- [Python-Integration](#python-integration)
- [Tipps und Tricks](#tipps-und-tricks)

## 🛠️ Konfiguration

### Voraussetzungen

- **Rust** (Version 1.70 oder höher)
- **Python** (Version 3.8 oder höher)  
- **Jupyter Lab** oder **Jupyter Notebook**

### Rust-Kernel für Jupyter

Installieren Sie den Rust-Kernel für Jupyter:

```bash
# evcxr_jupyter installieren
cargo install evcxr_jupyter

# Kernel installieren
evcxr_jupyter --install
```

Installation überprüfen:
```bash
jupyter kernelspec list
```

## 📦 Installation

### Via Cargo

Erstellen Sie ein neues Rust-Projekt:

```bash
mkdir rustorch-notebook
cd rustorch-notebook
cargo init
```

Fügen Sie zu `Cargo.toml` hinzu:

```toml
[package]
name = "rustorch-notebook"
version = "0.1.0"
edition = "2021"

[dependencies]
rustorch = { version = "0.6.7", features = ["python", "cuda"] }
ndarray = "0.16"
plotters = "0.3"
serde_json = "1.0"
```

### Mit spezifischen Features

Für verschiedene Anwendungsfälle:

```toml
# Grundlegendes Machine Learning
rustorch = { version = "0.6.7", features = ["linalg"] }

# Mit GPU-Beschleunigung
rustorch = { version = "0.6.7", features = ["cuda", "metal"] }  

# Für Webentwicklung
rustorch = { version = "0.6.7", features = ["wasm", "webgpu"] }

# Vollständige Features
rustorch = { version = "0.6.7", features = ["python", "cuda", "metal", "linalg", "model-hub"] }
```

## 🚀 Erstes Notebook

### Initiale Konfiguration

```rust
// Abhängigkeiten importieren
:dep rustorch = { version = "0.6.7", features = ["python"] }
:dep plotters = "0.3"

use rustorch::prelude::*;
use std::error::Error;

// Display konfigurieren
println!("RusTorch Version: {}", env!("CARGO_PKG_VERSION"));
```

### Grundlegende Tensor-Operationen

```rust
// Tensoren erstellen
let tensor_a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let tensor_b = Tensor::ones(&[2, 2], ScalarType::Float32);

println!("Tensor A:\n{:?}", tensor_a);
println!("Tensor B:\n{:?}", tensor_b);

// Operationen
let ergebnis = tensor_a.add(&tensor_b)?;
println!("A + B:\n{:?}", ergebnis);
```

## 🔬 Praktische Beispiele

### 1. Einfache lineare Regression

```rust
use rustorch::{nn::*, optim::*, autograd::*};

// Synthetische Daten generieren
let n_samples = 100;
let true_w = 2.0;
let true_b = 1.0;

// X: Features, y: Targets
let x_data: Vec<f32> = (0..n_samples).map(|i| i as f32 / 10.0).collect();
let y_data: Vec<f32> = x_data.iter()
    .map(|x| true_w * x + true_b + (rand::random::<f32>() - 0.5) * 0.1)
    .collect();

let x = Tensor::from_slice(&x_data, &[n_samples, 1])?;
let y = Tensor::from_slice(&y_data, &[n_samples, 1])?;

// Modell
let mut model = Linear::new(1, 1)?;

// Optimierer
let mut optimizer = Adam::new(model.parameters(), 0.01)?;

// Training
for epoch in 0..1000 {
    let prediction = model.forward(&x)?;
    let loss = mse_loss(&prediction, &y)?;
    
    if epoch % 100 == 0 {
        println!("Epoche {}: Verlust = {:.6}", epoch, loss.item::<f32>());
    }
    
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
}

println!("Training abgeschlossen!");
```

### 2. Neuronales Netz für Klassifikation

```rust
// Iris-Datensatz simulieren
fn generate_iris_data() -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    // Klasse 0: Setosa
    for _ in 0..50 {
        features.extend_from_slice(&[
            4.0 + rand::random::<f32>() * 2.0,  // sepal_length
            3.0 + rand::random::<f32>() * 1.0,  // sepal_width  
            1.0 + rand::random::<f32>() * 0.5,  // petal_length
            0.2 + rand::random::<f32>() * 0.3,  // petal_width
        ]);
        labels.push(0.0);
    }
    
    // Weitere Klassen...
    // (Ähnlicher Code für Klassen 1 und 2)
    
    let x = Tensor::from_slice(&features, &[150, 4])?;
    let y = Tensor::from_slice(&labels, &[150])?;
    
    Ok((x, y))
}

// Klassifikationsmodell
let mut model = Sequential::new()
    .add(Linear::new(4, 16)?)
    .add(ReLU::new())
    .add(Linear::new(16, 8)?)
    .add(ReLU::new())
    .add(Linear::new(8, 3)?);  // 3 Klassen

// Training
let mut optimizer = Adam::new(model.parameters(), 0.001)?;

for epoch in 0..500 {
    let prediction = model.forward(&x)?;
    let loss = cross_entropy(&prediction, &y.to_dtype(ScalarType::Int64)?)?;
    
    if epoch % 50 == 0 {
        // Genauigkeit berechnen
        let pred_classes = prediction.argmax(1, false)?;
        let y_int = y.to_dtype(ScalarType::Int64)?;
        let correct = pred_classes.eq(&y_int)?.to_dtype(ScalarType::Float32)?.sum()?;
        let accuracy = correct.item::<f32>() / 150.0 * 100.0;
        
        println!("Epoche {}: Verlust = {:.4}, Genauigkeit = {:.2}%", 
                epoch, loss.item::<f32>(), accuracy);
    }
    
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
}
```

## 📊 Visualisierungen

### Training-Verlust plotten

```rust
use plotters::prelude::*;

fn plot_training_loss(losses: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut buffer = Vec::new();
    {
        let root = SVGBackend::with_buffer(&mut buffer, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Training-Verlust", ("Arial", 50))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f32..losses.len() as f32,
                *losses.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                ...*losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            )?;
            
        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(i, &loss)| (i as f32, loss)),
            &RED,
        ))?.label("Verlust").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));
        
        chart.configure_series_labels().draw()?;
    }
    
    println!("Diagramm gespeichert als: training_loss.svg");
    Ok(())
}
```

## 🐍 Python-Integration

### RusTorch in Python verwenden

```python
# Python-Bindings installieren
# cargo build --features python
# maturin develop

import numpy as np
import matplotlib.pyplot as plt
import rustorch_py

# Tensor erstellen
tensor = rustorch_py.create_tensor([1, 2, 3, 4], [2, 2])
print(f"Erstellter Tensor: {tensor}")
```

## 💡 Tipps und Tricks

### 1. Effizientes Debugging

```rust
fn debug_tensor(tensor: &Tensor, name: &str) {
    println!("{}: Form={:?}, Datentyp={:?}", 
             name, tensor.size(), tensor.dtype());
}
```

### 2. Speicher-Management

```rust
// GPU-Speicher explizit freigeben
{
    let big_tensor = Tensor::randn(&[10000, 10000], ScalarType::Float32).to(cuda_device)?;
    // Tensor verwenden...
} // Tensor wird hier automatisch freigegeben

// CUDA-Cache leeren
if cfg!(feature = "cuda") {
    rustorch::cuda::empty_cache();
}
```

### 3. Performance-Profiling

```rust
use std::time::Instant;

let start = Instant::now();
let result = model.forward(&batch)?;
let duration = start.elapsed();

println!("Forward-Pass: {:?}", duration);
```

## 🚀 Nächste Schritte

1. **Experimentieren Sie mit verschiedenen Architekturen**: CNNs, RNNs, Transformers
2. **Performance optimieren**: GPU-Beschleunigung, Quantisierung
3. **Modelle deployen**: WebAssembly, ONNX, Mobile
4. **Integration mit Tools**: TensorBoard, MLflow

Für weitere Beispiele und vollständige Dokumentation besuchen Sie:
- [GitHub Repository](https://github.com/JunSuzukiJapan/rustorch)
- [API-Dokumentation](https://docs.rs/rustorch)
- [Vollständige Beispiele](../examples/)