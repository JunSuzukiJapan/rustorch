# Guia Completo do RusTorch com Jupyter

Este guia mostra como usar RusTorch em notebooks Jupyter para desenvolvimento interativo de machine learning e deep learning.

## 📋 Sumário

- [Configuração](#configuração)
- [Instalação](#instalação)
- [Primeiro Notebook](#primeiro-notebook)
- [Exemplos Práticos](#exemplos-práticos)
- [Visualizações](#visualizações)
- [Integração com Python](#integração-com-python)
- [Dicas e Truques](#dicas-e-truques)

## 🛠️ Configuração

### Pré-requisitos

- **Rust** (versão 1.70 ou superior)
- **Python** (versão 3.8 ou superior)  
- **Jupyter Lab** ou **Jupyter Notebook**

### Kernel Rust para Jupyter

Instale o kernel Rust para Jupyter:

```bash
# Instalar evcxr_jupyter
cargo install evcxr_jupyter

# Instalar kernel
evcxr_jupyter --install
```

Verificar instalação:
```bash
jupyter kernelspec list
```

## 📦 Instalação

### Via Cargo

Crie um novo projeto Rust:

```bash
mkdir rustorch-notebook
cd rustorch-notebook
cargo init
```

Adicione ao `Cargo.toml`:

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

### Com Recursos Específicos

Para diferentes casos de uso:

```toml
# Machine Learning básico
rustorch = { version = "0.6.7", features = ["linalg"] }

# Com aceleração GPU
rustorch = { version = "0.6.7", features = ["cuda", "metal"] }  

# Para desenvolvimento web
rustorch = { version = "0.6.7", features = ["wasm", "webgpu"] }

# Recursos completos
rustorch = { version = "0.6.7", features = ["python", "cuda", "metal", "linalg", "model-hub"] }
```

## 🚀 Primeiro Notebook

### Configuração Inicial

```rust
// Importar dependências
:dep rustorch = { version = "0.6.7", features = ["python"] }
:dep plotters = "0.3"

use rustorch::prelude::*;
use std::error::Error;

// Configurar display
println!("RusTorch versão: {}", env!("CARGO_PKG_VERSION"));
```

### Operações Básicas com Tensores

```rust
// Criar tensores
let tensor_a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let tensor_b = Tensor::ones(&[2, 2], ScalarType::Float32);

println!("Tensor A:\n{:?}", tensor_a);
println!("Tensor B:\n{:?}", tensor_b);

// Operações
let resultado = tensor_a.add(&tensor_b)?;
println!("A + B:\n{:?}", resultado);
```

### Visualização de Dados

```rust
// Usando plotters para visualização
use plotters::prelude::*;

// Gerar dados
let x_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
let y_data: Vec<f32> = x_data.iter().map(|x| x.sin()).collect();

// Criar tensor
let x_tensor = Tensor::from_slice(&x_data, &[100, 1])?;
let y_tensor = Tensor::from_slice(&y_data, &[100, 1])?;

println!("Dados criados: {} pontos", x_data.len());
```

## 🔬 Exemplos Práticos

### 1. Regressão Linear Simples

```rust
use rustorch::{nn::*, optim::*, autograd::*};

// Gerar dados sintéticos
let n_samples = 100;
let true_w = 2.0;
let true_b = 1.0;

// X: features, y: targets
let x_data: Vec<f32> = (0..n_samples).map(|i| i as f32 / 10.0).collect();
let y_data: Vec<f32> = x_data.iter()
    .map(|x| true_w * x + true_b + (rand::random::<f32>() - 0.5) * 0.1)
    .collect();

let x = Tensor::from_slice(&x_data, &[n_samples, 1])?;
let y = Tensor::from_slice(&y_data, &[n_samples, 1])?;

// Modelo
let mut model = Linear::new(1, 1)?;

// Otimizador
let mut optimizer = Adam::new(model.parameters(), 0.01)?;

// Treinamento
for epoch in 0..1000 {
    let prediction = model.forward(&x)?;
    let loss = mse_loss(&prediction, &y)?;
    
    if epoch % 100 == 0 {
        println!("Epoch {}: Loss = {:.6}", epoch, loss.item::<f32>());
    }
    
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
}

println!("Treinamento completo!");
```

### 2. Classificação com Rede Neural

```rust
// Dataset Iris sintético
fn generate_iris_data() -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    // Classe 0: Setosa
    for _ in 0..50 {
        features.extend_from_slice(&[
            4.0 + rand::random::<f32>() * 2.0,  // sepal_length
            3.0 + rand::random::<f32>() * 1.0,  // sepal_width  
            1.0 + rand::random::<f32>() * 0.5,  // petal_length
            0.2 + rand::random::<f32>() * 0.3,  // petal_width
        ]);
        labels.push(0.0);
    }
    
    // Classe 1: Versicolor  
    for _ in 0..50 {
        features.extend_from_slice(&[
            6.0 + rand::random::<f32>() * 1.0,
            2.8 + rand::random::<f32>() * 0.8,
            4.0 + rand::random::<f32>() * 1.0,
            1.3 + rand::random::<f32>() * 0.5,
        ]);
        labels.push(1.0);
    }
    
    // Classe 2: Virginica
    for _ in 0..50 {
        features.extend_from_slice(&[
            7.0 + rand::random::<f32>() * 1.0,
            3.0 + rand::random::<f32>() * 0.8,
            5.5 + rand::random::<f32>() * 1.0,
            2.0 + rand::random::<f32>() * 0.5,
        ]);
        labels.push(2.0);
    }
    
    let x = Tensor::from_slice(&features, &[150, 4])?;
    let y = Tensor::from_slice(&labels, &[150])?;
    
    Ok((x, y))
}

// Criar dados
let (x, y) = generate_iris_data()?;

// Modelo de classificação
let mut model = Sequential::new()
    .add(Linear::new(4, 16)?)
    .add(ReLU::new())
    .add(Linear::new(16, 8)?)
    .add(ReLU::new())
    .add(Linear::new(8, 3)?);  // 3 classes

// Treinamento
let mut optimizer = Adam::new(model.parameters(), 0.001)?;

for epoch in 0..500 {
    let prediction = model.forward(&x)?;
    let loss = cross_entropy(&prediction, &y.to_dtype(ScalarType::Int64)?)?;
    
    if epoch % 50 == 0 {
        // Calcular acurácia
        let pred_classes = prediction.argmax(1, false)?;
        let y_int = y.to_dtype(ScalarType::Int64)?;
        let correct = pred_classes.eq(&y_int)?.to_dtype(ScalarType::Float32)?.sum()?;
        let accuracy = correct.item::<f32>() / 150.0 * 100.0;
        
        println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", 
                epoch, loss.item::<f32>(), accuracy);
    }
    
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
}
```

### 3. Processamento de Imagens

```rust
use rustorch::vision::transforms::*;

// Simular imagem 28x28 (MNIST-like)
let image_data: Vec<f32> = (0..784).map(|_| rand::random::<f32>()).collect();
let image = Tensor::from_slice(&image_data, &[1, 1, 28, 28])?;

// Aplicar transformações
let transform = Compose::new(vec![
    Box::new(Resize::new(32, 32)),
    Box::new(Normalize::new(vec![0.5], vec![0.5])),
]);

let transformed = transform.forward(&image)?;
println!("Imagem original: {:?}", image.size());
println!("Imagem transformada: {:?}", transformed.size());

// CNN simples
let mut cnn = Sequential::new()
    .add(Conv2d::new(1, 16, 3, 1, 1)?)    // 1->16 canais, kernel 3x3
    .add(ReLU::new())
    .add(MaxPool2d::new(2, 2)?)           // pooling 2x2
    .add(Conv2d::new(16, 32, 3, 1, 1)?)   // 16->32 canais
    .add(ReLU::new())
    .add(MaxPool2d::new(2, 2)?)
    .add(Flatten::new())                   // flatten para linear
    .add(Linear::new(32 * 8 * 8, 128)?)   // ajustar tamanho
    .add(ReLU::new())
    .add(Linear::new(128, 10)?);          // 10 classes

let output = cnn.forward(&transformed)?;
println!("Saída da CNN: {:?}", output.size());
```

## 📊 Visualizações

### Plotar Loss de Treinamento

```rust
// Usando plotters para gráficos
use plotters::prelude::*;
use std::io::Write;

fn plot_training_loss(losses: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut buffer = Vec::new();
    {
        let root = SVGBackend::with_buffer(&mut buffer, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Loss de Treinamento", ("Arial", 50))
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
        ))?.label("Loss").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));
        
        chart.configure_series_labels().draw()?;
    }
    
    // Salvar como arquivo
    let mut file = std::fs::File::create("training_loss.svg")?;
    file.write_all(&buffer)?;
    
    println!("Gráfico salvo em: training_loss.svg");
    Ok(())
}

// Exemplo de uso
let exemplo_losses = vec![2.3, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2];
plot_training_loss(&exemplo_losses)?;
```

## 🐍 Integração com Python

### Usar RusTorch em Python

```python
# Primeiro, instalar os bindings Python
# cargo build --features python
# maturin develop

import numpy as np
import matplotlib.pyplot as plt

# Importar RusTorch Python bindings
import rustorch_py

# Criar tensor
tensor = rustorch_py.create_tensor([1, 2, 3, 4], [2, 2])
print(f"Tensor criado: {tensor}")

# Converter para NumPy
numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(f"Array NumPy: {numpy_array}")
```

### Callback Python de Rust

```rust
// No Rust
use rustorch::python_bindings::*;

// Registrar callback Python
let mut registry = CallbackRegistry::new();
registry.register_callback(
    "progress".to_string(),
    python_callback_function,
    py
)?;

// Chamar função Python do Rust
let result = call_python_from_rust(
    &registry,
    "progress".to_string(),
    "Treinamento em progresso...".to_string(),
    py
)?;
```

## 💡 Dicas e Truques

### 1. Debugging Eficiente

```rust
// Imprimir forma e tipo de tensor
fn debug_tensor(tensor: &Tensor, name: &str) {
    println!("{}: shape={:?}, dtype={:?}", 
             name, tensor.size(), tensor.dtype());
}

debug_tensor(&x, "input");
debug_tensor(&output, "prediction");
```

### 2. Gerenciamento de Memória

```rust
// Liberar memória GPU explicitamente
{
    let big_tensor = Tensor::randn(&[10000, 10000], ScalarType::Float32).to(cuda_device)?;
    // usar tensor...
} // tensor é automaticamente liberado aqui

// Limpar cache CUDA
if cfg!(feature = "cuda") {
    rustorch::cuda::empty_cache();
}
```

### 3. Checkpoint e Restauração

```rust
use rustorch::serialize::*;

// Salvar estado do treinamento
#[derive(Serialize, Deserialize)]
struct TrainingState {
    epoch: usize,
    loss: f32,
    model_state: Vec<u8>,
    optimizer_state: Vec<u8>,
}

let state = TrainingState {
    epoch: current_epoch,
    loss: current_loss,
    model_state: model.state_dict(),
    optimizer_state: optimizer.state_dict(),
};

save_checkpoint(&state, "checkpoint.pth")?;
```

### 4. Profiling de Performance

```rust
use std::time::Instant;

// Medir tempo de execução
let start = Instant::now();
let result = model.forward(&batch)?;
let duration = start.elapsed();

println!("Forward pass: {:?}", duration);

// Medir uso de memória GPU
if cfg!(feature = "cuda") {
    let memory_used = rustorch::cuda::memory_allocated();
    println!("Memória GPU: {} MB", memory_used / 1024 / 1024);
}
```

### 5. Validação de Modelo

```rust
// Modo de avaliação
model.eval();

let mut total_correct = 0;
let mut total_samples = 0;

for batch in validation_loader {
    // Sem gradientes para validação
    with_no_grad(|| {
        let prediction = model.forward(&batch.input)?;
        let predicted = prediction.argmax(1, false)?;
        
        total_correct += predicted.eq(&batch.target)?.sum().item::<i64>();
        total_samples += batch.target.size()[0];
        
        Ok(())
    })?;
}

let accuracy = total_correct as f32 / total_samples as f32 * 100.0;
println!("Acurácia de validação: {:.2}%", accuracy);

// Voltar ao modo de treinamento
model.train();
```

## 🚀 Próximos Passos

1. **Experimente com diferentes arquiteturas**: CNNs, RNNs, Transformers
2. **Otimize performance**: Use aceleração GPU, quantização
3. **Deploy modelos**: WebAssembly, ONNX, mobile
4. **Integre com ferramentas**: TensorBoard, MLflow, Weights & Biases

Para mais exemplos avançados e documentação completa, visite:
- [Repositório GitHub](https://github.com/JunSuzukiJapan/rustorch)
- [Documentação da API](https://docs.rs/rustorch)
- [Exemplos Completos](../examples/)