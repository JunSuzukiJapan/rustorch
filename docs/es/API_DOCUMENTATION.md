# Documentación API RusTorch

## 📚 Referencia API Completa

Este documento proporciona documentación API completa para RusTorch v0.5.15, organizada por módulo y funcionalidad. Incluye manejo de errores unificado con `RusTorchError` y `RusTorchResult<T>` para gestión consistente de errores a través de todos los 1060+ tests. **Fase 8 COMPLETADA** añade utilidades de tensores avanzadas incluyendo operaciones condicionales, indexación y funciones estadísticas. **Fase 9 COMPLETADA** introduce sistema de serialización completo con guardado/carga de modelos, compilación JIT y soporte de múltiples formatos incluyendo compatibilidad PyTorch.

## 🏗️ Arquitectura Core

### Estructura de Módulos

```
rustorch/
├── tensor/              # Operaciones tensores core y estructuras datos
├── nn/                  # Capas redes neurales y funciones
├── autograd/            # Motor diferenciación automática
├── optim/               # Optimizadores y programadores tasa aprendizaje
├── special/             # Funciones matemáticas especiales
├── distributions/       # Distribuciones estadísticas
├── vision/              # Transformaciones computer vision
├── linalg/              # Operaciones álgebra lineal (BLAS/LAPACK)
├── gpu/                 # Aceleración GPU (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Operaciones tensores dispersos y poda (Fase 12)
├── serialization/       # Serialización modelos y compilación JIT (Fase 9)
└── wasm/                # Bindings WebAssembly (ver [Documentación API WASM](WASM_API_DOCUMENTATION.md))
```

## 📊 Módulo Tensor

### Creación Tensores Base

```rust
use rustorch::tensor::Tensor;

// Creación base
let tensor = Tensor::new(vec![2, 3]);               // Creación basada en forma
let tensor = Tensor::from_vec(data, vec![2, 3]);    // Desde vector datos
let tensor = Tensor::zeros(vec![10, 10]);           // Tensor lleno ceros
let tensor = Tensor::ones(vec![5, 5]);              // Tensor lleno unos
let tensor = Tensor::randn(vec![3, 3]);             // Distribución normal aleatoria
let tensor = Tensor::rand(vec![3, 3]);              // Distribución uniforme aleatoria [0,1)
let tensor = Tensor::eye(5);                        // Matriz identidad
let tensor = Tensor::full(vec![2, 2], 3.14);       // Llenar con valor específico
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Tensor rango
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Espaciado lineal
```

### Operaciones Tensores

```rust
// Operaciones aritméticas
let result = a.add(&b);                             // Suma elemento por elemento
let result = a.sub(&b);                             // Resta elemento por elemento
let result = a.mul(&b);                             // Multiplicación elemento por elemento
let result = a.div(&b);                             // División elemento por elemento
let result = a.pow(&b);                             // Potencia elemento por elemento
let result = a.rem(&b);                             // Resto elemento por elemento

// Operaciones matriciales
let result = a.matmul(&b);                          // Multiplicación matricial
let result = a.transpose();                         // Transposición matricial
let result = a.dot(&b);                             // Producto escalar

// Funciones matemáticas
let result = tensor.exp();                          // Exponencial
let result = tensor.ln();                           // Logaritmo natural
let result = tensor.log10();                        // Logaritmo base 10
let result = tensor.sqrt();                         // Raíz cuadrada
let result = tensor.abs();                          // Valor absoluto
let result = tensor.sin();                          // Función seno
let result = tensor.cos();                          // Función coseno
let result = tensor.tan();                          // Función tangente
let result = tensor.asin();                         // Arcoseno
let result = tensor.acos();                         // Arcocoseno
let result = tensor.atan();                         // Arcotangente
let result = tensor.sinh();                         // Seno hiperbólico
let result = tensor.cosh();                         // Coseno hiperbólico
let result = tensor.tanh();                         // Tangente hiperbólica
let result = tensor.floor();                        // Función suelo
let result = tensor.ceil();                         // Función techo
let result = tensor.round();                        // Función redondear
let result = tensor.sign();                         // Función signo
let result = tensor.max();                          // Valor máximo
let result = tensor.min();                          // Valor mínimo
let result = tensor.sum();                          // Suma todos elementos
let result = tensor.mean();                         // Valor medio
let result = tensor.std();                          // Desviación estándar
let result = tensor.var();                          // Varianza

// Manipulación forma
let result = tensor.reshape(vec![6, 4]);            // Redimensionar tensor
let result = tensor.squeeze();                      // Quitar dimensiones tamaño-1
let result = tensor.unsqueeze(1);                   // Añadir dimensión en índice
let result = tensor.permute(vec![1, 0, 2]);         // Permutar dimensiones
let result = tensor.expand(vec![10, 10, 5]);        // Expandir dimensiones tensor
```

## 🧠 Módulo Neural Network (nn)

### Capas Base

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// Capa lineal
let linear = Linear::new(784, 256)?;                // entrada 784, salida 256
let output = linear.forward(&input)?;

// Capa convolucional
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// Normalización por lotes
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### Funciones Activación

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// Funciones activación base
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Funciones activación parametrizadas
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// Ejemplo uso
let activated = relu.forward(&input)?;
```

## 🚀 Módulo Aceleración GPU

### Gestión Dispositivos

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// Verificar dispositivos disponibles
let device_count = get_device_count()?;
let device = Device::best_available()?;            // Selección mejor dispositivo

// Configuración dispositivo
set_device(&device)?;

// Mover tensor a GPU
let gpu_tensor = tensor.to_device(&device)?;
```

### Operaciones CUDA

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// Operaciones dispositivo CUDA
let cuda_device = CudaDevice::new(0)?;              // Usar GPU 0
let stats = memory_stats(0)?;                      // Estadísticas memoria
println!("Memoria usada: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 Módulo Optimizador (Optim)

### Optimizadores Base

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Optimizador Adam
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// Optimizador SGD
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// Paso optimización
optimizer.zero_grad()?;
// ... cálculo hacia adelante y retropropagación ...
optimizer.step()?;
```

## 📖 Ejemplo Uso

### Regresión Lineal

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// Preparación datos
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// Definición modelo
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// Bucle entrenamiento
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("Época {}: Pérdida = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ Limitaciones Conocidas

1. **Limitación memoria GPU**: Gestión explícita memoria requerida para tensores grandes (>8GB)
2. **Limitación WebAssembly**: Algunas operaciones BLAS no disponibles en entorno WASM
3. **Entrenamiento distribuido**: Backend NCCL solo soportado en Linux
4. **Limitación Metal**: Algunas operaciones avanzadas solo disponibles con backend CUDA

## 🔗 Enlaces Relacionados

- [README Principal](../README.md)
- [Documentación API WASM](WASM_API_DOCUMENTATION.md)
- [Guía Jupyter](jupyter-guide.md)
- [Repositorio GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Paquete Crates.io](https://crates.io/crates/rustorch)

---

**Última Actualización**: v0.5.15 | **Licencia**: MIT | **Autor**: Jun Suzuki