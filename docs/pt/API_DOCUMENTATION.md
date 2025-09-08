# Documentação da API RusTorch

Esta documentação fornece uma referência abrangente da API RusTorch para desenvolvedores brasileiros e de língua portuguesa.

## Índice

- [Módulo Tensor](#módulo-tensor)
- [Diferenciação Automática](#diferenciação-automática)
- [Redes Neurais](#redes-neurais)
- [Otimização](#otimização)
- [Visão Computacional](#visão-computacional)
- [GPU e Dispositivos](#gpu-e-dispositivos)
- [Utilitários](#utilitários)

## Módulo Tensor

### `Tensor`

Estrutura fundamental para operações de tensor N-dimensional.

#### Construtores

```rust
// Criar tensor a partir de dados
let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;

// Tensor com zeros
let zeros = Tensor::zeros(&[2, 3], ScalarType::Float32);

// Tensor com uns
let ones = Tensor::ones(&[2, 3], ScalarType::Float32);

// Tensor com números aleatórios (distribuição normal)
let randn = Tensor::randn(&[2, 3], ScalarType::Float32);

// Tensor com números aleatórios (distribuição uniforme)
let rand = Tensor::rand(&[2, 3], ScalarType::Float32);
```

#### Operações Básicas

```rust
// Operações aritméticas
let result = tensor1.add(&tensor2)?;
let result = tensor1.sub(&tensor2)?;
let result = tensor1.mul(&tensor2)?;
let result = tensor1.div(&tensor2)?;

// Multiplicação de matrizes
let result = tensor1.matmul(&tensor2)?;

// Transposição
let transposed = tensor.t()?;

// Redimensionamento
let reshaped = tensor.reshape(&[6, 1])?;
```

#### Operações de Redução

```rust
// Soma
let sum_all = tensor.sum()?;
let sum_dim = tensor.sum_dim(0, false)?;

// Média
let mean_all = tensor.mean()?;
let mean_dim = tensor.mean_dim(0, false)?;

// Máximo e mínimo
let max_val = tensor.max()?;
let min_val = tensor.min()?;
```

#### Indexação e Seleção

```rust
// Seleção por índice
let slice = tensor.slice(0, 0, 2, 1)?;

// Seleção por condição
let mask = tensor.gt(&threshold)?;
let selected = tensor.masked_select(&mask)?;
```

## Diferenciação Automática

### `Variable`

Wrapper para tensores que permite diferenciação automática.

```rust
use rustorch::autograd::*;

// Criar variável com requires_grad=true
let x = Variable::new(Tensor::randn(&[2, 2], ScalarType::Float32), true);
let y = Variable::new(Tensor::randn(&[2, 2], ScalarType::Float32), true);

// Operações que constroem o grafo de computação
let z = x.matmul(&y)?;
let loss = z.sum()?;

// Backpropagation
loss.backward()?;

// Acessar gradientes
let x_grad = x.grad()?;
println!("Gradiente de x: {:?}", x_grad);
```

### Funções de Diferenciação

```rust
// Função personalizada com gradiente
fn custom_function(input: &Variable) -> Result<Variable, TensorError> {
    // Forward pass
    let output = input.pow(&Variable::new(
        Tensor::from_f32(2.0), false
    ))?;
    
    // O gradiente será calculado automaticamente
    Ok(output)
}
```

## Redes Neurais

### Camadas Básicas

#### `Linear`

Transformação linear (camada completamente conectada).

```rust
use rustorch::nn::*;

let linear = Linear::new(784, 256)?; // entrada: 784, saída: 256
let input = Tensor::randn(&[32, 784], ScalarType::Float32);
let output = linear.forward(&input)?;
```

#### Camadas de Ativação

```rust
// ReLU
let relu = ReLU::new();
let output = relu.forward(&input)?;

// Sigmoid
let sigmoid = Sigmoid::new();
let output = sigmoid.forward(&input)?;

// Tanh
let tanh = Tanh::new();
let output = tanh.forward(&input)?;

// GELU
let gelu = GELU::new();
let output = gelu.forward(&input)?;
```

### Camadas Convolucionais

```rust
// Convolução 2D
let conv2d = Conv2d::new(
    3,    // canais de entrada
    64,   // canais de saída
    3,    // tamanho do kernel
    1,    // stride
    1     // padding
)?;

let input = Tensor::randn(&[1, 3, 224, 224], ScalarType::Float32);
let output = conv2d.forward(&input)?;
```

### Modelos Sequenciais

```rust
let model = Sequential::new()
    .add(Linear::new(784, 512)?)
    .add(ReLU::new())
    .add(Dropout::new(0.2))
    .add(Linear::new(512, 256)?)
    .add(ReLU::new())
    .add(Linear::new(256, 10)?);

// Forward pass
let input = Tensor::randn(&[32, 784], ScalarType::Float32);
let output = model.forward(&input)?;
```

## Otimização

### Otimizadores

#### `Adam`

```rust
use rustorch::optim::*;

let mut optimizer = Adam::new(
    model.parameters(),  // parâmetros do modelo
    0.001,              // taxa de aprendizado
    (0.9, 0.999),       // betas
    1e-8                // epsilon
)?;

// Loop de treinamento
for batch in data_loader {
    let prediction = model.forward(&batch.input)?;
    let loss = criterion(&prediction, &batch.target)?;
    
    loss.backward()?;
    optimizer.step()?;
    optimizer.zero_grad()?;
}
```

#### `SGD`

```rust
let mut optimizer = SGD::new(
    model.parameters(),
    0.01,               // taxa de aprendizado
    0.9                 // momentum
)?;
```

### Funções de Loss

```rust
use rustorch::nn::functional::*;

// Mean Squared Error
let mse_loss = mse_loss(&prediction, &target)?;

// Cross Entropy
let ce_loss = cross_entropy(&prediction, &target)?;

// Binary Cross Entropy
let bce_loss = binary_cross_entropy(&prediction, &target)?;
```

## Visão Computacional

### Transformações de Imagem

```rust
use rustorch::vision::transforms::*;

// Redimensionar imagem
let resize = Resize::new(224, 224);
let resized = resize.forward(&image)?;

// Normalização
let normalize = Normalize::new(
    vec![0.485, 0.456, 0.406],  // média
    vec![0.229, 0.224, 0.225]   // desvio padrão
);
let normalized = normalize.forward(&image)?;

// Transformações aleatórias
let random_crop = RandomCrop::new(32, 4);  // tamanho, padding
let cropped = random_crop.forward(&image)?;
```

### Modelos Pré-treinados

```rust
use rustorch::vision::models::*;

// ResNet
let resnet18 = resnet18(true)?;  // pré-treinado
let output = resnet18.forward(&input)?;

// VGG
let vgg16 = vgg16(true)?;
let features = vgg16.features(&input)?;
```

## GPU e Dispositivos

### Gerenciamento de Dispositivos

```rust
use rustorch::{Device, DeviceType};

// CPU
let cpu = Device::cpu();

// CUDA
let cuda = Device::cuda(0)?;  // GPU 0
let cuda_available = Device::cuda_if_available();

// Metal (macOS)
let metal = Device::metal(0)?;

// Mover tensor para dispositivo
let tensor_gpu = tensor.to(cuda)?;
```

### Operações Multi-GPU

```rust
use rustorch::distributed::*;

// Inicializar processamento distribuído
init_process_group("nccl", "tcp://localhost:23456", 0, 2)?;

// AllReduce para sincronização de gradientes
all_reduce(&mut tensor, ReduceOp::Sum)?;
```

## Utilitários

### Serialização

```rust
use rustorch::serialize::*;

// Salvar modelo
save(&model, "model.pth")?;

// Carregar modelo
let loaded_model: Sequential = load("model.pth")?;
```

### Métricas

```rust
use rustorch::metrics::*;

// Acurácia
let accuracy = accuracy(&predictions, &targets)?;

// F1 Score
let f1 = f1_score(&predictions, &targets, "macro")?;

// Matriz de confusão
let confusion_matrix = confusion_matrix(&predictions, &targets)?;
```

### Utilitários de Dados

```rust
use rustorch::data::*;

// DataLoader
let dataset = TensorDataset::new(inputs, targets);
let data_loader = DataLoader::new(dataset, 32, true)?;  // batch_size=32, shuffle=true

for batch in data_loader {
    let loss = train_step(&batch)?;
}
```

## Exemplos Completos

### Classificação com CNN

```rust
use rustorch::prelude::*;

struct CNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl CNN {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(CNN {
            conv1: Conv2d::new(1, 32, 3, 1, 1)?,
            conv2: Conv2d::new(32, 64, 3, 1, 1)?,
            fc1: Linear::new(64 * 7 * 7, 128)?,
            fc2: Linear::new(128, 10)?,
            dropout: Dropout::new(0.5),
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let x = self.conv1.forward(x)?;
        let x = relu(&x)?;
        let x = max_pool2d(&x, 2)?;
        
        let x = self.conv2.forward(&x)?;
        let x = relu(&x)?;
        let x = max_pool2d(&x, 2)?;
        
        let x = x.view(&[-1, 64 * 7 * 7])?;
        let x = self.fc1.forward(&x)?;
        let x = relu(&x)?;
        let x = self.dropout.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        
        Ok(x)
    }
}
```

Este é um guia abrangente das principais funcionalidades da API RusTorch. Para exemplos mais detalhados e casos de uso avançados, consulte os exemplos no repositório oficial.