# RusTorch - Biblioteca de Deep Learning em Rust

[![Crates.io](https://img.shields.io/crates/v/rustorch.svg)](https://crates.io/crates/rustorch)
[![Documentação](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![Licença](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)

**RusTorch** é uma biblioteca de deep learning pronta para produção, compatível com PyTorch, implementada em Rust puro com funções matemáticas especiais, distribuições estatísticas, transformadas de Fourier (FFT/RFFT), decomposição de matrizes (SVD/QR/LU/autovalores), diferenciação automática, redes neurais, transformações de visão computacional, aceleração GPU completa (CUDA/Metal/OpenCL), otimizações SIMD, processamento paralelo, suporte para WebAssembly em navegadores, suporte abrangente para aprendizado distribuído e validação de desempenho.

## ✨ Principais Recursos

### 🚀 Operações de Tensor de Alta Performance
- **Operações de Tensor N-dimensional** com interface similar ao PyTorch
- **Otimizações SIMD** para máxima performance da CPU
- **Processamento Paralelo** usando Rayon para operações em lote
- **Gerenciamento Inteligente de Memória** com alocação otimizada

### 🧮 Matemática Avançada
- **Funções Especiais**: Gamma, Bessel, funções de erro, integrais elípticas
- **Distribuições Estatísticas**: Normal, Exponencial, Gamma, Beta, e mais
- **Transformadas de Fourier**: FFT/RFFT implementadas com algoritmos otimizados
- **Decomposição de Matrizes**: SVD, QR, LU, decomposição de autovalores

### 🤖 Deep Learning Completo
- **Diferenciação Automática** com grafo de computação dinâmico
- **Camadas de Redes Neurais**: Linear, Convolucional, LSTM, Attention
- **Funções de Ativação**: ReLU, Sigmoid, Tanh, GELU, Swish, e mais
- **Algoritmos de Otimização**: Adam, SGD, AdaGrad, RMSprop

### 🎨 Visão Computacional
- **Transformações de Imagem**: Redimensionamento, rotação, normalização
- **Filtros**: Convolução, detecção de bordas, desfoque
- **Processamento de Pipeline** para fluxos de trabalho de visão

### ⚡ Aceleração GPU
- **CUDA**: Suporte completo para GPUs NVIDIA (CUDA 12.x)
- **Metal**: Aceleração nativa para GPUs Apple Silicon
- **OpenCL**: Suporte multiplataforma para vários fornecedores de GPU
- **Seleção Automática de Dispositivo** baseada em disponibilidade

### 🌐 Compatibilidade com WebAssembly
- **Execução em Navegador** com otimizações específicas para WASM
- **WebGPU**: Aceleração GPU em navegadores Chrome/Edge
- **Detecção de Recursos** para capacidades do navegador
- **Otimização de Tamanho de Bundle** para aplicações web

### 🔗 Integração Python
- **Bindings PyO3** para interoperabilidade perfeita com Python
- **Compatibilidade com NumPy** para troca de arrays
- **API Familiar do PyTorch** para fácil migração
- **Sistema de Callbacks** para comunicação Rust ↔ Python

### 📊 Aprendizado Distribuído
- **Processamento Multi-GPU** com NCCL
- **Sincronização de Gradientes** para treinamento em larga escala
- **Estratégias de Paralelismo** (dados, modelo, pipeline)
- **Tolerância a Falhas** com checkpointing automático

## 🚀 Início Rápido

### Instalação

Adicione ao seu `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.6.7"

# Recursos opcionais
[dependencies.rustorch]
version = "0.6.7"
features = [
    "cuda",           # Aceleração CUDA
    "metal",          # Aceleração Metal (macOS)
    "linalg",         # Álgebra linear avançada
    "python",         # Bindings Python
    "wasm",           # Suporte WebAssembly
    "model-hub",      # Download de modelos
]
```

### Uso Básico

```rust
use rustorch::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Criar tensores
    let x = Tensor::randn(&[2, 3], ScalarType::Float32);
    let y = Tensor::ones(&[3, 4], ScalarType::Float32);
    
    // Multiplicação de matrizes
    let z = x.matmul(&y)?;
    println!("Resultado: {:?}", z);
    
    // Construir uma rede neural simples
    let mut model = Sequential::new()
        .add(Linear::new(784, 256)?)
        .add(ReLU::new())
        .add(Linear::new(256, 10)?);
    
    // Forward pass
    let input = Tensor::randn(&[32, 784], ScalarType::Float32);
    let output = model.forward(&input)?;
    
    Ok(())
}
```

### Treinamento com Autograd

```rust
use rustorch::{nn::*, optim::*, autograd::*};

// Definir modelo
let mut model = Sequential::new()
    .add(Linear::new(784, 128)?)
    .add(ReLU::new())
    .add(Linear::new(128, 10)?);

// Configurar otimizador
let mut optimizer = Adam::new(model.parameters(), 0.001)?;

// Loop de treinamento
for epoch in 0..100 {
    let prediction = model.forward(&input)?;
    let loss = mse_loss(&prediction, &target)?;
    
    // Backpropagation
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
    
    println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>());
}
```

## 📚 Documentação

- **[Guia de Início Rápido]**: Tutorial para começar
- **[Documentação da API]**: Referência completa da API
- **[Guia WebAssembly]**: Integração com navegadores
- **[Guia CUDA]**: Aceleração GPU
- **[Exemplos]**: Projetos de exemplo

## 🌟 Recursos Avançados

### Processamento GPU
```rust
// Detecção automática de GPU
let device = Device::cuda_if_available();
let tensor = Tensor::randn(&[1024, 1024], ScalarType::Float32).to(device);

// Operações aceleradas por GPU
let result = tensor.matmul(&tensor.t())?;
```

### Integração Python
```python
import rustorch

# Usar diretamente do Python
tensor = rustorch.tensor([1, 2, 3, 4])
result = tensor.sum()
print(f"Soma: {result}")
```

### Deployment WebAssembly
```javascript
import init, { RusTorchModel } from './pkg/rustorch.js';

async function runModel() {
    await init();
    const model = new RusTorchModel();
    const result = model.forward([1, 2, 3, 4]);
    console.log('Resultado:', result);
}
```

## 🏗️ Arquitetura

RusTorch é construído com arquitetura modular:

```
rustorch/
├── tensor/          # Operações de tensor fundamentais
├── autograd/        # Sistema de diferenciação automática
├── nn/             # Camadas de redes neurais
├── optim/          # Algoritmos de otimização
├── vision/         # Utilitários de visão computacional
├── distributed/    # Computação distribuída
├── gpu/            # Kernels de aceleração GPU
└── python/         # Bindings Python
```

## 🚀 Performance

Benchmarks mostram performance competitiva:

| Operação | RusTorch | PyTorch | NumPy |
|----------|----------|---------|--------|
| Multiplicação de Matrix (1024×1024) | 1.2ms | 1.8ms | 3.2ms |
| Convolução 2D | 0.8ms | 1.1ms | 2.4ms |
| Forward Pass (ResNet) | 12ms | 15ms | N/A |

*Medido em Intel i7-12700K com RTX 3080*

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja o [GUIA DE CONTRIBUIÇÃO] para detalhes.

## 📄 Licença

Este projeto está licenciado sob MIT OU Apache-2.0 - veja os arquivos [LICENSE-MIT] e [LICENSE-APACHE] para detalhes.

## 🙏 Agradecimentos

- Inspirado pelo PyTorch e pelo ecossistema Rust
- Agradecimentos especiais aos contribuidores e mantenedores
- Construído com ❤️ para a comunidade de machine learning

---

**RusTorch** - Liberando o poder do deep learning com segurança e performance do Rust 🦀✨