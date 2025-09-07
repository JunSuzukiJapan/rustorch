# Guia de Início Rápido do RusTorch

## Instalação

### 1. Pré-requisitos
```bash
# Rust 1.70 ou superior
rustc --version

# Python 3.8 ou superior
python --version

# Instalar dependências necessárias
pip install maturin numpy matplotlib
```

### 2. Compilar e Instalar RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Criar ambiente virtual Python (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Compilar e instalar em modo de desenvolvimento
maturin develop --release
```

## Exemplos Básicos de Uso

### 1. Criação de Tensores e Operações Básicas

```python
import rustorch
import numpy as np

# Criação de tensor
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tensor x:\n{x}")
print(f"Forma: {x.shape()}")  # [2, 2]

# Matrizes zero e matrizes identidade
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Matriz zero:\n{zeros}")
print(f"Matriz de uns:\n{ones}")
print(f"Matriz identidade:\n{identity}")

# Tensores aleatórios
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Aleatório normal:\n{random_normal}")
print(f"Aleatório uniforme:\n{random_uniform}")

# Integração com NumPy
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"Do NumPy:\n{tensor_from_numpy}")

# Converter de volta para NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"De volta para NumPy:\n{back_to_numpy}")
```

### 2. Operações Aritméticas

```python
# Operações aritméticas básicas
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Operações elemento por elemento
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (elemento por elemento)
div_result = a.div(b)  # a / b (elemento por elemento)

print(f"Adição:\n{add_result}")
print(f"Subtração:\n{sub_result}")
print(f"Multiplicação:\n{mul_result}")
print(f"Divisão:\n{div_result}")

# Operações escalares
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Adição escalar (+2):\n{scalar_add}")
print(f"Multiplicação escalar (*3):\n{scalar_mul}")

# Multiplicação de matrizes
matmul_result = a.matmul(b)
print(f"Multiplicação matricial:\n{matmul_result}")

# Funções matemáticas
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Raiz quadrada:\n{sqrt_result}")
print(f"Exponencial:\n{exp_result}")
print(f"Logaritmo natural:\n{log_result}")
```

### 3. Manipulação da Forma de Tensores

```python
# Exemplos de manipulação de forma
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Forma original: {original.shape()}")  # [2, 4]

# Remodelagem
reshaped = original.reshape([4, 2])
print(f"Remodelado [4, 2]:\n{reshaped}")

# Transposição
transposed = original.transpose(0, 1)
print(f"Transposto:\n{transposed}")

# Adição/remoção de dimensões
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Antes do squeeze: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"Após squeeze: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"Após unsqueeze: {expanded.shape()}")  # [1, 3]
```

### 4. Operações Estatísticas

```python
# Funções estatísticas
data = rustorch.randn([3, 4])
print(f"Dados:\n{data}")

# Estatísticas básicas
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Média: {mean_val.item():.4f}")
print(f"Soma: {sum_val.item():.4f}")
print(f"Desvio padrão: {std_val.item():.4f}")
print(f"Variância: {var_val.item():.4f}")
print(f"Máximo: {max_val.item():.4f}")
print(f"Mínimo: {min_val.item():.4f}")

# Estatísticas ao longo de dimensões específicas
row_mean = data.mean(dim=1)  # Média de cada linha
col_sum = data.sum(dim=0)    # Soma de cada coluna

print(f"Médias das linhas: {row_mean}")
print(f"Somas das colunas: {col_sum}")
```

## Fundamentos de Diferenciação Automática

### 1. Cálculo de Gradientes

```python
# Exemplo de diferenciação automática
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Tensor de entrada: {x}")

# Criar variável
var_x = rustorch.autograd.Variable(x)

# Construir grafo computacional
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Saída: {y.data().item()}")

# Propagação reversa
y.backward()

# Obter gradiente
grad = var_x.grad()
print(f"Gradiente: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Grafos Computacionais Complexos

```python
# Exemplo mais complexo
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Função complexa: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), depois soma

print(f"Resultado: {z.data().item():.4f}")

# Propagação reversa
z.backward()
grad = var_x.grad()
print(f"Gradiente: {grad}")
```

## Fundamentos de Redes Neurais

### 1. Camada Linear Simples

```python
# Criar camada linear
linear_layer = rustorch.nn.Linear(3, 1)  # 3 entradas -> 1 saída

# Entrada aleatória
input_data = rustorch.randn([2, 3])  # Tamanho do lote 2, 3 características
print(f"Entrada: {input_data}")

# Passada para frente
output = linear_layer.forward(input_data)
print(f"Saída: {output}")

# Verificar parâmetros
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Forma do peso: {weight.shape()}")
print(f"Peso: {weight}")
if bias is not None:
    print(f"Viés: {bias}")
```

### 2. Funções de Ativação

```python
# Várias funções de ativação
x = rustorch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# ReLU
relu = rustorch.nn.ReLU()
relu_output = relu.forward(x)
print(f"ReLU: {relu_output}")

# Sigmoid
sigmoid = rustorch.nn.Sigmoid()
sigmoid_output = sigmoid.forward(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh
tanh = rustorch.nn.Tanh()
tanh_output = tanh.forward(x)
print(f"Tanh: {tanh_output}")
```

### 3. Funções de Perda

```python
# Exemplos de uso de funções de perda
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Erro quadrático médio
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"Perda MSE: {loss_value.item():.6f}")

# Entropia cruzada (para classificação)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Índices das classes

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Perda de entropia cruzada: {ce_loss_value.item():.6f}")
```

## Processamento de Dados

### 1. Conjuntos de Dados e DataLoaders

```python
# Criar conjunto de dados
import numpy as np

# Gerar dados de amostra
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 amostras, 4 características
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # Classificação de 3 classes

# Converter para tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Criar conjunto de dados
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Tamanho do conjunto de dados: {len(dataset)}")

# Criar dataloader
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Obter lotes do dataloader
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Mostrar apenas os primeiros 3 lotes
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Lote {batch_idx}: Forma da entrada {inputs.shape()}, Forma do alvo {targets.shape()}")
```

### 2. Transformações de Dados

```python
# Exemplos de transformação de dados
data = rustorch.randn([10, 10])
print(f"Média original dos dados: {data.mean().item():.4f}")
print(f"Desvio padrão original dos dados: {data.std().item():.4f}")

# Transformação de normalização
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Média dos dados normalizados: {normalized_data.mean().item():.4f}")
print(f"Desvio padrão dos dados normalizados: {normalized_data.std().item():.4f}")
```

## Exemplo Completo de Treinamento

### Regressão Linear

```python
# Exemplo completo de regressão linear
import numpy as np

# Gerar dados
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Converter para tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Criar conjunto de dados e dataloader
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Definir modelo
model = rustorch.nn.Linear(1, 1)  # 1 entrada -> 1 saída

# Função de perda e otimizador
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Loop de treinamento
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    dataloader.reset()
    while True:
        batch = dataloader.next_batch()
        if batch is None:
            break
        
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Passada para frente
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Propagação reversa (simplificada)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Época {epoch}: Perda = {avg_loss:.6f}")

print("Treinamento concluído!")

# Parâmetros finais
final_weight = model.weight()
final_bias = model.bias()
print(f"Peso aprendido: {final_weight.item():.4f} (verdadeiro: 2.0)")
if final_bias is not None:
    print(f"Viés aprendido: {final_bias.item():.4f} (verdadeiro: 1.0)")
```

## Solução de Problemas

### Problemas Comuns e Soluções

1. **Problemas de Instalação**
```bash
# Se maturin não for encontrado
pip install --upgrade maturin

# Se Rust estiver desatualizado
rustup update

# Problemas de ambiente Python
python -m pip install --upgrade pip
```

2. **Erros de Tempo de Execução**
```python
# Verificar formas de tensores
print(f"Forma do tensor: {tensor.shape()}")
print(f"Tipo de dados do tensor: {tensor.dtype()}")

# Cuidado com tipos de dados na conversão NumPy
np_array = np.array(data, dtype=np.float32)  # float32 explícito
```

3. **Otimização de Performance**
```python
# Compilar em modo release
# maturin develop --release

# Ajustar tamanho do lote
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Lote maior
```

## Próximos Passos

1. **Experimente Exemplos Avançados**: Confira os exemplos em `docs/examples/neural_networks/`
2. **Use API estilo Keras**: `rustorch.training.Model` para construção mais fácil de modelos
3. **Recursos de Visualização**: `rustorch.visualization` para visualização do progresso de treinamento
4. **Treinamento Distribuído**: `rustorch.distributed` para processamento paralelo

Documentação Detalhada:
- [Referência da API Python](../en/python_api_reference.md)
- [Documentação de Visão Geral](../en/python_bindings_overview.md)
- [Coleção de Exemplos](../examples/)