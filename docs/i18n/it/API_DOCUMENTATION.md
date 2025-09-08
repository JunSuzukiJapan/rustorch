# Documentazione API RusTorch

## 📚 Riferimento API Completo

Questo documento fornisce documentazione API completa per RusTorch v0.5.15, organizzata per modulo e funzionalità. Include gestione degli errori unificata con `RusTorchError` e `RusTorchResult<T>` per gestione coerente degli errori attraverso tutti i 1060+ test. **Fase 8 COMPLETATA** aggiunge utilità di tensori avanzate incluse operazioni condizionali, indicizzazione e funzioni statistiche. **Fase 9 COMPLETATA** introduce sistema di serializzazione completo con salvataggio/caricamento modelli, compilazione JIT e supporto formati multipli inclusa compatibilità PyTorch.

## 🏗️ Architettura Core

### Struttura Moduli

```
rustorch/
├── tensor/              # Operazioni tensori core e strutture dati
├── nn/                  # Layer reti neurali e funzioni
├── autograd/            # Motore differenziazione automatica
├── optim/               # Ottimizzatori e schedulatori tasso apprendimento
├── special/             # Funzioni matematiche speciali
├── distributions/       # Distribuzioni statistiche
├── vision/              # Trasformazioni computer vision
├── linalg/              # Operazioni algebra lineare (BLAS/LAPACK)
├── gpu/                 # Accelerazione GPU (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Operazioni tensori sparsi e pruning (Fase 12)
├── serialization/       # Serializzazione modelli e compilazione JIT (Fase 9)
└── wasm/                # Binding WebAssembly (vedi [Documentazione API WASM](WASM_API_DOCUMENTATION.md))
```

## 📊 Modulo Tensor

### Creazione Tensori Base

```rust
use rustorch::tensor::Tensor;

// Creazione base
let tensor = Tensor::new(vec![2, 3]);               // Creazione basata su forma
let tensor = Tensor::from_vec(data, vec![2, 3]);    // Da vettore dati
let tensor = Tensor::zeros(vec![10, 10]);           // Tensore riempito zeri
let tensor = Tensor::ones(vec![5, 5]);              // Tensore riempito uni
let tensor = Tensor::randn(vec![3, 3]);             // Distribuzione normale casuale
let tensor = Tensor::rand(vec![3, 3]);              // Distribuzione uniforme casuale [0,1)
let tensor = Tensor::eye(5);                        // Matrice identità
let tensor = Tensor::full(vec![2, 2], 3.14);       // Riempi con valore specifico
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Tensore intervallo
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Spaziatura lineare
```

### Operazioni Tensori

```rust
// Operazioni aritmetiche
let result = a.add(&b);                             // Addizione elemento per elemento
let result = a.sub(&b);                             // Sottrazione elemento per elemento
let result = a.mul(&b);                             // Moltiplicazione elemento per elemento
let result = a.div(&b);                             // Divisione elemento per elemento
let result = a.pow(&b);                             // Potenza elemento per elemento
let result = a.rem(&b);                             // Resto elemento per elemento

// Operazioni matriciali
let result = a.matmul(&b);                          // Moltiplicazione matriciale
let result = a.transpose();                         // Trasposizione matriciale
let result = a.dot(&b);                             // Prodotto scalare

// Funzioni matematiche
let result = tensor.exp();                          // Esponenziale
let result = tensor.ln();                           // Logaritmo naturale
let result = tensor.log10();                        // Logaritmo base 10
let result = tensor.sqrt();                         // Radice quadrata
let result = tensor.abs();                          // Valore assoluto
let result = tensor.sin();                          // Funzione seno
let result = tensor.cos();                          // Funzione coseno
let result = tensor.tan();                          // Funzione tangente
let result = tensor.asin();                         // Arcoseno
let result = tensor.acos();                         // Arcocoseno
let result = tensor.atan();                         // Arcotangente
let result = tensor.sinh();                         // Seno iperbolico
let result = tensor.cosh();                         // Coseno iperbolico
let result = tensor.tanh();                         // Tangente iperbolica
let result = tensor.floor();                        // Funzione pavimento
let result = tensor.ceil();                         // Funzione soffitto
let result = tensor.round();                        // Funzione arrotonda
let result = tensor.sign();                         // Funzione segno
let result = tensor.max();                          // Valore massimo
let result = tensor.min();                          // Valore minimo
let result = tensor.sum();                          // Somma tutti elementi
let result = tensor.mean();                         // Valore medio
let result = tensor.std();                          // Deviazione standard
let result = tensor.var();                          // Varianza

// Manipolazione forma
let result = tensor.reshape(vec![6, 4]);            // Ridimensiona tensore
let result = tensor.squeeze();                      // Rimuovi dimensioni taglia-1
let result = tensor.unsqueeze(1);                   // Aggiungi dimensione all'indice
let result = tensor.permute(vec![1, 0, 2]);         // Permuta dimensioni
let result = tensor.expand(vec![10, 10, 5]);        // Espandi dimensioni tensore
```

## 🧠 Modulo Neural Network (nn)

### Layer Base

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// Layer lineare
let linear = Linear::new(784, 256)?;                // input 784, output 256
let output = linear.forward(&input)?;

// Layer convoluzionale
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// Normalizzazione batch
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### Funzioni Attivazione

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// Funzioni attivazione base
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Funzioni attivazione parametrizzate
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// Esempio utilizzo
let activated = relu.forward(&input)?;
```

## 🚀 Modulo Accelerazione GPU

### Gestione Dispositivi

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// Verifica dispositivi disponibili
let device_count = get_device_count()?;
let device = Device::best_available()?;            // Selezione miglior dispositivo

// Configurazione dispositivo
set_device(&device)?;

// Sposta tensore su GPU
let gpu_tensor = tensor.to_device(&device)?;
```

### Operazioni CUDA

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// Operazioni dispositivo CUDA
let cuda_device = CudaDevice::new(0)?;              // Usa GPU 0
let stats = memory_stats(0)?;                      // Statistiche memoria
println!("Memoria utilizzata: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 Modulo Ottimizzatore (Optim)

### Ottimizzatori Base

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Ottimizzatore Adam
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// Ottimizzatore SGD
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// Passo ottimizzazione
optimizer.zero_grad()?;
// ... calcolo in avanti e retropropagazione ...
optimizer.step()?;
```

## 📖 Esempio Utilizzo

### Regressione Lineare

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// Preparazione dati
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// Definizione modello
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// Loop addestramento
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("Epoca {}: Perdita = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ Limitazioni Conosciute

1. **Limitazione memoria GPU**: Gestione esplicita memoria richiesta per tensori grandi (>8GB)
2. **Limitazione WebAssembly**: Alcune operazioni BLAS non disponibili in ambiente WASM
3. **Addestramento distribuito**: Backend NCCL supportato solo su Linux
4. **Limitazione Metal**: Alcune operazioni avanzate disponibili solo con backend CUDA

## 🔗 Collegamenti Correlati

- [README Principale](../README.md)
- [Documentazione API WASM](WASM_API_DOCUMENTATION.md)
- [Guida Jupyter](jupyter-guide.md)
- [Repository GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Pacchetto Crates.io](https://crates.io/crates/rustorch)

---

**Ultimo Aggiornamento**: v0.5.15 | **Licenza**: MIT | **Autore**: Jun Suzuki