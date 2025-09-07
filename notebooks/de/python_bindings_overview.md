# RusTorch Python Bindings Übersicht

## Übersicht

RusTorch ist ein hochperformantes Deep Learning Framework, implementiert in Rust und bietet PyTorch-ähnliche APIs während es Rusts Sicherheit und Performance-Vorteile nutzt. Durch Python-Bindings können Sie RusTorch-Funktionalitäten direkt aus Python heraus nutzen.

## Hauptmerkmale

### 🚀 **Hohe Performance**
- **Rust-Kern**: Erreicht C++-Level Performance während Speichersicherheit garantiert wird
- **SIMD-Unterstützung**: Automatische Vektorisierung für optimierte numerische Berechnungen
- **Parallelverarbeitung**: Effiziente parallele Berechnung mit rayon
- **Null-Kopie**: Minimale Datenkopierung zwischen NumPy und RusTorch

### 🛡️ **Sicherheit**
- **Speichersicherheit**: Verhindert Speicherlecks und Data Races durch Rusts Ownership-System
- **Typsicherheit**: Compile-Time Type-Checking reduziert Laufzeitfehler
- **Fehlerbehandlung**: Umfassende Fehlerbehandlung mit automatischer Konvertierung zu Python-Exceptions

### 🎯 **Benutzerfreundlichkeit**
- **PyTorch-kompatible API**: Einfache Migration von existierendem PyTorch-Code
- **Keras-artige High-Level API**: Intuitive Schnittstellen wie model.fit()
- **NumPy-Integration**: Bidirektionale Konvertierung mit NumPy-Arrays

## Architektur

RusTorchs Python-Bindings bestehen aus 10 Modulen:

### 1. **tensor** - Tensor-Operationen
```python
import rustorch

# Tensor-Erstellung
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# NumPy-Integration
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - Automatische Differenzierung
```python
# Gradientenberechnung
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # Gradienten ausgeben
```

### 3. **nn** - Neuronale Netzwerke
```python
# Schicht-Erstellung
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# Verlustfunktionen
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - Optimierer
```python
# Optimierer
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Lernraten-Scheduler
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - Datenladen
```python
# Dataset-Erstellung
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Datentransformationen
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - High-Level Training-API
```python
# Keras-artige API
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Training-Ausführung
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - Verteiltes Training
```python
# Verteilte Training-Konfiguration
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# Datenparallel
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - Visualisierung
```python
# Training-Verlauf plotten
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# Tensor-Visualisierung
plotter.plot_tensor_as_image(tensor, title="Feature Map")
```

### 9. **utils** - Hilfsprogramme
```python
# Modell speichern/laden
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# Profiling
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## Installation

### Voraussetzungen
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (für GPU-Nutzung)

### Erstellen und Installieren
```bash
# Repository klonen
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install maturin numpy

# Erstellen und installieren
maturin develop --release

# Oder von PyPI installieren (für die Zukunft geplant)
# pip install rustorch
```

## Schnellstart

### 1. Grundlegende Tensor-Operationen
```python
import rustorch
import numpy as np

# Tensor-Erstellung
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Shape: {x.shape()}")  # Shape: [2, 2]

# Mathematische Operationen
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Ergebnis: {z.to_numpy()}")
```

### 2. Lineare Regression Beispiel
```python
import rustorch
import numpy as np

# Daten generieren
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Zu Tensoren konvertieren
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Modell definieren
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# Dataset erstellen
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Trainieren
history = model.fit(dataloader, epochs=100, verbose=True)

# Ergebnisse anzeigen
print(f"Finaler Verlust: {history.train_loss()[-1]:.4f}")
```

### 3. Neuronales Netzwerk Klassifikation
```python
import rustorch

# Daten vorbereiten
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Modell erstellen
model = rustorch.Model("KlassifikationsNetz")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# Modell kompilieren
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training-Konfiguration
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# Trainieren
history = trainer.train(model, train_loader, val_loader)

# Evaluieren
metrics = model.evaluate(test_loader)
print(f"Test-Genauigkeit: {metrics['accuracy']:.4f}")
```

## Performance-Optimierung

### SIMD-Nutzung
```python
# SIMD-Optimierung während des Builds aktivieren
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # SIMD-optimierte Berechnung
```

### GPU-Nutzung
```python
# CUDA-Nutzung (für die Zukunft geplant)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # GPU-Berechnung
```

### Paralleles Datenladen
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # Anzahl paralleler Worker
)
```

## Best Practices

### 1. Speichereffizienz
```python
# Null-Kopie-Konvertierung nutzen
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # Kein Kopieren

# In-Place-Operationen verwenden
tensor.add_(1.0)  # Speichereffizient
```

### 2. Fehlerbehandlung
```python
try:
    result = model(ungueltige_eingabe)
except rustorch.RusTorchError as e:
    print(f"RusTorch-Fehler: {e}")
except Exception as e:
    print(f"Unerwarteter Fehler: {e}")
```

### 3. Debugging und Profiling
```python
# Profiler verwenden
profiler = rustorch.utils.Profiler()
profiler.start()

# Berechnung ausführen
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## Einschränkungen

### Aktuelle Einschränkungen
- **GPU-Unterstützung**: CUDA/ROCm-Unterstützung in Entwicklung
- **Dynamische Graphen**: Unterstützt derzeit nur statische Graphen
- **Verteiltes Training**: Nur grundlegende Funktionalität implementiert

### Zukünftige Erweiterungen
- GPU-Beschleunigung (CUDA, Metal, ROCm)
- Unterstützung für dynamische Berechnungsgraphen
- Mehr neuronale Netzwerk-Schichten
- Modellquantisierung und Pruning
- ONNX-Export-Funktionalität

## Mitwirken

### Entwicklungsteilnahme
```bash
# Entwicklungsumgebung einrichten
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# Tests ausführen
cargo test
python -m pytest tests/

# Code-Qualitätsprüfungen
cargo clippy
cargo fmt
```

### Community
- GitHub Issues: Fehlermeldungen und Feature-Requests
- Discussions: Fragen und Diskussionen
- Discord: Echtzeit-Support

## Lizenz

RusTorch wird unter der MIT-Lizenz veröffentlicht. Frei verwendbar für sowohl kommerzielle als auch nicht-kommerzielle Zwecke.

## Verwandte Links

- [GitHub Repository](https://github.com/JunSuzukiJapan/RusTorch)
- [API Dokumentation](./python_api_reference.md)
- [Beispiele und Tutorials](../examples/)
- [Performance-Benchmarks](./benchmarks.md)