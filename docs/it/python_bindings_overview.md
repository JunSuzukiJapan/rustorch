# Panoramica dei Bindings Python di RusTorch

Una panoramica completa dell'integrazione Python in RusTorch per un'interoperabilità senza soluzione di continuità tra Rust e Python.

## 🌉 Panoramica

I bindings Python di RusTorch consentono di utilizzare la potente libreria di deep learning basata su Rust direttamente da Python. Questi bindings combinano le prestazioni e la sicurezza di Rust con la facilità d'uso di Python.

## 📋 Indice

- [Architettura](#architettura)
- [Installazione e Configurazione](#installazione-e-configurazione)
- [Funzionalità Principale](#funzionalità-principale)
- [Panoramica dei Moduli](#panoramica-dei-moduli)
- [Funzionalità Avanzate](#funzionalità-avanzate)
- [Ottimizzazioni delle Prestazioni](#ottimizzazioni-delle-prestazioni)
- [Interoperabilità](#interoperabilità)
- [Linee Guida per lo Sviluppo](#linee-guida-per-lo-sviluppo)

## 🏗️ Architettura

### Integrazione PyO3

RusTorch utilizza PyO3 per l'interoperabilità Python-Rust:

```rust
use pyo3::prelude::*;

#[pymodule]
fn rustorch_py(_py: Python, m: &PyModule) -> PyResult<()> {
    // Registrare moduli tensor
    m.add_class::<PyTensor>()?;
    
    // API funzionale
    m.add_function(wrap_pyfunction!(create_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_operations, m)?)?;
    
    Ok(())
}
```

### Struttura Modulare

```
rustorch_py/
├── tensor/          # Operazioni tensor di base
├── autograd/        # Differenziazione automatica
├── nn/              # Layer di reti neurali
├── optim/           # Algoritmi di ottimizzazione
├── data/            # Elaborazione e caricamento dati
├── training/        # Loop e utilità di training
├── utils/           # Funzioni ausiliarie
├── distributed/     # Training distribuito
└── visualization/   # Grafici e visualizzazione
```

## 🛠️ Installazione e Configurazione

### Prerequisiti

- **Rust** (versione 1.70+)
- **Python** (versione 3.8+)
- **PyO3** (versione 0.24+)
- **Maturin** per la build

### Processo di Build

```bash
# Compilare bindings Python
cargo build --features python

# Sviluppare con Maturin (modalità sviluppo)
maturin develop --features python

# Build di release
maturin build --release --features python
```

### Installazione lato Python

```python
# Dopo la build
pip install target/wheels/rustorch_py-*.whl

# O direttamente con Maturin
pip install maturin
maturin develop
```

## ⚡ Funzionalità Principale

### 1. Operazioni Tensor

```python
import rustorch_py

# Creare tensor
tensor = rustorch_py.create_tensor([1, 2, 3, 4], shape=[2, 2])
print(f"Tensor: {tensor}")

# Operazioni di base
result = rustorch_py.tensor_add(tensor, tensor)
matrix_result = rustorch_py.tensor_matmul(tensor, tensor)
```

### 2. Differenziazione Automatica

```python
# Tensor capaci di gradiente
x = rustorch_py.create_variable([2.0, 3.0], requires_grad=True)
y = rustorch_py.create_variable([1.0, 4.0], requires_grad=True)

# Passaggio in avanti
z = rustorch_py.operations.mul(x, y)
loss = rustorch_py.operations.sum(z)

# Passaggio all'indietro
rustorch_py.backward(loss)

print(f"Gradiente di x: {x.grad}")
print(f"Gradiente di y: {y.grad}")
```

### 3. Reti Neurali

```python
# Definire layer
linear = rustorch_py.nn.Linear(input_size=784, output_size=128)
relu = rustorch_py.nn.ReLU()
dropout = rustorch_py.nn.Dropout(p=0.2)

# Modello sequenziale
model = rustorch_py.nn.Sequential([
    linear,
    relu,
    dropout,
    rustorch_py.nn.Linear(128, 10)
])

# Passaggio in avanti
input_data = rustorch_py.create_tensor(data, shape=[batch_size, 784])
output = model.forward(input_data)
```

## 📦 Panoramica dei Moduli

### Modulo Tensor

```python
import rustorch_py.tensor as tensor

# Creazione tensor
zeros = tensor.zeros([3, 4])
ones = tensor.ones([2, 2])
randn = tensor.randn([5, 5])

# Operazioni
result = tensor.add(a, b)
transposed = tensor.transpose(matrix, 0, 1)
reshaped = tensor.reshape(tensor_input, [6, -1])
```

### Modulo Autograd

```python
import rustorch_py.autograd as autograd

# Variabile con calcolo gradiente
var = autograd.Variable(data, requires_grad=True)

# Calcolare gradienti
loss = compute_loss(var)
autograd.backward(loss)

# Attivare/disattivare raccolta gradienti
with autograd.no_grad():
    prediction = model.forward(input_data)
```

### Modulo Neural Network

```python
import rustorch_py.nn as nn

# Layer di base
linear = nn.Linear(in_features, out_features)
conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
lstm = nn.LSTM(input_size, hidden_size, num_layers)

# Funzioni di attivazione
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
gelu = nn.GELU()

# Funzioni di loss
mse_loss = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()
```

### Modulo di Ottimizzazione

```python
import rustorch_py.optim as optim

# Ottimizzatori
adam = optim.Adam(model.parameters(), lr=0.001)
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Loop di training
for epoch in range(num_epochs):
    prediction = model.forward(input_data)
    loss = criterion(prediction, target)
    
    # Calcolo gradienti
    loss.backward()
    
    # Aggiornare parametri
    optimizer.step()
    optimizer.zero_grad()
```

## 🚀 Funzionalità Avanzate

### Accelerazione GPU

```python
# Supporto CUDA
if rustorch_py.cuda.is_available():
    device = rustorch_py.device("cuda:0")
    tensor_gpu = tensor.to(device)
    
    # Operazioni GPU
    result = rustorch_py.cuda.matmul(tensor_gpu, tensor_gpu)

# Supporto Metal (macOS)
if rustorch_py.metal.is_available():
    metal_device = rustorch_py.device("metal:0")
    tensor_metal = tensor.to(metal_device)
```

### Training Distribuito

```python
import rustorch_py.distributed as dist

# Inizializzazione
dist.init_process_group("nccl", rank=0, world_size=4)

# Training Multi-GPU
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# All-Reduce per sincronizzazione gradienti
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

### Elaborazione Dati

```python
import rustorch_py.data as data

# Classe Dataset
class CustomDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# DataLoader
dataset = CustomDataset(train_data, train_targets)
dataloader = data.DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4
)
```

## ⚡ Ottimizzazioni delle Prestazioni

### Ottimizzazioni SIMD

```python
# Abilitare ottimizzazioni SIMD
rustorch_py.set_simd_enabled(True)

# Abilitare parallelizzazione
rustorch_py.set_num_threads(8)  # Per parallelizzazione CPU
```

### Gestione Memoria

```python
# Pool di memoria per allocazione efficiente
rustorch_py.memory.enable_memory_pool()

# Pulire cache memoria GPU
if rustorch_py.cuda.is_available():
    rustorch_py.cuda.empty_cache()
```

### Compilazione Just-in-Time

```python
# Compilazione JIT per funzioni critiche
@rustorch_py.jit.script
def optimized_function(x, y):
    return rustorch_py.operations.mul(x, y) + rustorch_py.operations.sin(x)

result = optimized_function(tensor1, tensor2)
```

## 🔄 Interoperabilità

### Integrazione NumPy

```python
import numpy as np
import rustorch_py

# NumPy → RusTorch
numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
rust_tensor = rustorch_py.from_numpy(numpy_array)

# RusTorch → NumPy
numpy_result = rust_tensor.numpy()
```

### Compatibilità PyTorch

```python
# Conversione tensor PyTorch
import torch

# PyTorch → RusTorch
torch_tensor = torch.randn(3, 4)
rust_tensor = rustorch_py.from_torch(torch_tensor)

# RusTorch → PyTorch
pytorch_tensor = rust_tensor.to_torch()
```

### Sistema Callback

```python
# Callback Python per training
def training_callback(epoch, loss, accuracy):
    print(f"Epoca {epoch}: Loss={loss:.4f}, Accuratezza={accuracy:.4f}")

# Registrare callback
rustorch_py.callbacks.register_training_callback(training_callback)

# Training con callback
trainer = rustorch_py.training.Trainer(model, optimizer, criterion)
trainer.train(dataloader, epochs=100)
```

## 📊 Visualizzazione

```python
import rustorch_py.visualization as viz

# Graficare storia training
viz.plot_training_history(losses, accuracies)

# Visualizzazione tensor
viz.visualize_tensor(tensor, title="Distribuzione Pesi")

# Grafico architettura rete
viz.plot_model_graph(model)
```

## 🧪 Linee Guida per lo Sviluppo

### Testing

```python
# Test unitari
import rustorch_py.testing as testing

def test_tensor_operations():
    a = rustorch_py.create_tensor([1, 2, 3])
    b = rustorch_py.create_tensor([4, 5, 6])
    
    result = rustorch_py.tensor_add(a, b)
    expected = [5, 7, 9]
    
    testing.assert_tensor_equal(result, expected)
```

### Debugging

```python
# Abilitare modalità debug
rustorch_py.set_debug_mode(True)

# Profiling
with rustorch_py.profiler.profile() as prof:
    result = model.forward(input_data)

prof.print_stats()
```

### Gestione Errori

```python
try:
    tensor = rustorch_py.create_tensor(data, shape)
except rustorch_py.TensorError as e:
    print(f"Errore tensor: {e}")
except rustorch_py.DeviceError as e:
    print(f"Errore dispositivo: {e}")
```

## 🔧 Configurazione Avanzata

### Variabili d'Ambiente

```bash
# Configurazione specifica Rust
export RUSTORCH_NUM_THREADS=8
export RUSTORCH_CUDA_DEVICE=0
export RUSTORCH_LOG_LEVEL=info

# Integrazione Python
export PYTHONPATH=$PYTHONPATH:./target/debug
```

### Configurazione Runtime

```python
# Impostazioni globali
rustorch_py.config.set_default_device("cuda:0")
rustorch_py.config.set_default_dtype(rustorch_py.float32)
rustorch_py.config.enable_fast_math(True)

# Configurazione pool thread
rustorch_py.config.set_thread_pool_size(16)
```

## 🚀 Prospettive Future

### Funzionalità Pianificate

- **Integrazione WebAssembly**: Deploy browser via WASM
- **Supporto Mobile**: Ottimizzazioni iOS/Android
- **Strategie di Distribuzione Avanzate**: Parallelismo pipeline
- **Quantizzazione**: Ottimizzazione inferenza INT8/FP16
- **Integrazione AutoML**: Ottimizzazione automatica iperparametri

### Contributi della Comunità

- **Sistema Plugin**: Architettura estensibile per operazioni personalizzate
- **Suite di Benchmarking**: Confronti prestazioni con altri framework
- **Collezione Tutorial**: Risorse di apprendimento complete

Per ulteriori informazioni e riferimento API completo, consultare la [Documentazione API Python](python_api_reference.md) e la [Guida Jupyter](jupyter-guide.md).