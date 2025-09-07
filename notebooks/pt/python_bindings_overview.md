# Visão Geral dos Bindings Python RusTorch

## Visão Geral

RusTorch é um framework de aprendizado profundo de alta performance implementado em Rust, fornecendo APIs similares ao PyTorch enquanto aproveita os benefícios de segurança e performance do Rust. Através dos bindings Python, você pode acessar a funcionalidade do RusTorch diretamente do Python.

## Características Principais

### 🚀 **Alta Performance**
- **Núcleo Rust**: Alcança performance de nível C++ enquanto garante segurança de memória
- **Suporte SIMD**: Vetorização automática para computações numéricas otimizadas
- **Processamento Paralelo**: Computação paralela eficiente usando rayon
- **Cópia Zero**: Cópia mínima de dados entre NumPy e RusTorch

### 🛡️ **Segurança**
- **Segurança de Memória**: Previne vazamentos de memória e condições de corrida através do sistema de propriedade do Rust
- **Segurança de Tipo**: Verificação de tipo em tempo de compilação reduz erros de tempo de execução
- **Tratamento de Erro**: Tratamento abrangente de erros com conversão automática para exceções Python

### 🎯 **Facilidade de Uso**
- **API Compatível com PyTorch**: Migração fácil do código PyTorch existente
- **API de Alto Nível estilo Keras**: Interfaces intuitivas como model.fit()
- **Integração NumPy**: Conversão bidirecional com arrays NumPy

## Arquitetura

Os bindings Python do RusTorch consistem em 10 módulos:

### 1. **tensor** - Operações de Tensor
```python
import rustorch

# Criação de tensores
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# Integração NumPy
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - Diferenciação Automática
```python
# Cálculo de gradientes
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # Obter gradientes
```

### 3. **nn** - Redes Neurais
```python
# Criação de camadas
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# Funções de perda
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - Otimizadores
```python
# Otimizadores
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Programadores de taxa de aprendizado
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - Carregamento de Dados
```python
# Criação de conjunto de dados
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Transformações de dados
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - API de Treinamento de Alto Nível
```python
# API estilo Keras
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Execução de treinamento
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - Treinamento Distribuído
```python
# Configuração de treinamento distribuído
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# Paralelismo de dados
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - Visualização
```python
# Plotar histórico de treinamento
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# Visualização de tensor
plotter.plot_tensor_as_image(tensor, title="Mapa de Características")
```

### 9. **utils** - Utilitários
```python
# Salvar/carregar modelo
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# Profiling
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## Instalação

### Pré-requisitos
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (para uso de GPU)

### Construir e Instalar
```bash
# Clonar repositório
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Criar ambiente virtual Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependências
pip install maturin numpy

# Construir e instalar
maturin develop --release

# Ou instalar do PyPI (planejado para o futuro)
# pip install rustorch
```

## Início Rápido

### 1. Operações Básicas de Tensor
```python
import rustorch
import numpy as np

# Criação de tensor
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Forma: {x.shape()}")  # Forma: [2, 2]

# Operações matemáticas
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Resultado: {z.to_numpy()}")
```

### 2. Exemplo de Regressão Linear
```python
import rustorch
import numpy as np

# Gerar dados
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Converter para tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Definir modelo
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# Criar conjunto de dados
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Treinar
history = model.fit(dataloader, epochs=100, verbose=True)

# Exibir resultados
print(f"Perda final: {history.train_loss()[-1]:.4f}")
```

### 3. Classificação com Rede Neural
```python
import rustorch

# Preparar dados
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Construir modelo
model = rustorch.Model("RedeClassificação")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# Compilar modelo
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Configuração de treinamento
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# Treinar
history = trainer.train(model, train_loader, val_loader)

# Avaliar
metrics = model.evaluate(test_loader)
print(f"Precisão de teste: {metrics['accuracy']:.4f}")
```

## Otimização de Performance

### Utilização SIMD
```python
# Habilitar otimização SIMD durante a construção
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # Computação otimizada com SIMD
```

### Uso de GPU
```python
# Uso de CUDA (planejado para o futuro)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # Computação em GPU
```

### Carregamento Paralelo de Dados
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # Número de workers paralelos
)
```

## Melhores Práticas

### 1. Eficiência de Memória
```python
# Utilizar conversão de cópia zero
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # Sem cópia

# Usar operações in-place
tensor.add_(1.0)  # Eficiente em memória
```

### 2. Tratamento de Erro
```python
try:
    result = model(entrada_inválida)
except rustorch.RusTorchError as e:
    print(f"Erro RusTorch: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")
```

### 3. Debug e Profiling
```python
# Usar profiler
profiler = rustorch.utils.Profiler()
profiler.start()

# Executar computação
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## Limitações

### Limitações Atuais
- **Suporte GPU**: Suporte CUDA/ROCm em desenvolvimento
- **Grafos Dinâmicos**: Atualmente suporta apenas grafos estáticos
- **Treinamento Distribuído**: Apenas funcionalidade básica implementada

### Extensões Futuras
- Aceleração GPU (CUDA, Metal, ROCm)
- Suporte para grafos de computação dinâmicos
- Mais camadas de rede neural
- Quantização e poda de modelo
- Funcionalidade de exportação ONNX

## Contribuindo

### Participação no Desenvolvimento
```bash
# Configurar ambiente de desenvolvimento
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# Executar testes
cargo test
python -m pytest tests/

# Verificações de qualidade de código
cargo clippy
cargo fmt
```

### Comunidade
- GitHub Issues: Relatórios de bugs e solicitações de recursos
- Discussions: Perguntas e discussões
- Discord: Suporte em tempo real

## Licença

RusTorch é lançado sob a Licença MIT. Livre para uso tanto para fins comerciais quanto não comerciais.

## Links Relacionados

- [Repositório GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Documentação da API](./python_api_reference.md)
- [Exemplos e Tutoriais](../examples/)
- [Benchmarks de Performance](./benchmarks.md)