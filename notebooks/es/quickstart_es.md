# Guía de Inicio Rápido de RusTorch

## Instalación

### 1. Requisitos Previos
```bash
# Rust 1.70 o superior
rustc --version

# Python 3.8 o superior
python --version

# Instalar dependencias requeridas
pip install maturin numpy matplotlib
```

### 2. Compilar e Instalar RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Crear entorno virtual de Python (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Compilar e instalar en modo desarrollo
maturin develop --release
```

## Ejemplos de Uso Básico

### 1. Creación de Tensores y Operaciones Básicas

```python
import rustorch
import numpy as np

# Creación de tensores
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tensor x:\n{x}")
print(f"Forma: {x.shape()}")  # [2, 2]

# Matrices de ceros y unos
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Ceros:\n{zeros}")
print(f"Unos:\n{ones}")
print(f"Identidad:\n{identity}")

# Tensores aleatorios
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Aleatorio normal:\n{random_normal}")
print(f"Aleatorio uniforme:\n{random_uniform}")

# Integración con NumPy
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"Desde NumPy:\n{tensor_from_numpy}")

# Convertir de vuelta a NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"De vuelta a NumPy:\n{back_to_numpy}")
```

### 2. Operaciones Aritméticas

```python
# Operaciones aritméticas básicas
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Operaciones elemento por elemento
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (elemento por elemento)
div_result = a.div(b)  # a / b (elemento por elemento)

print(f"Suma:\n{add_result}")
print(f"Resta:\n{sub_result}")
print(f"Multiplicación:\n{mul_result}")
print(f"División:\n{div_result}")

# Operaciones escalares
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Suma escalar (+2):\n{scalar_add}")
print(f"Multiplicación escalar (*3):\n{scalar_mul}")

# Multiplicación de matrices
matmul_result = a.matmul(b)
print(f"Multiplicación de matrices:\n{matmul_result}")

# Funciones matemáticas
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Raíz cuadrada:\n{sqrt_result}")
print(f"Exponencial:\n{exp_result}")
print(f"Logaritmo natural:\n{log_result}")
```

### 3. Manipulación de Formas de Tensores

```python
# Ejemplos de manipulación de formas
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Forma original: {original.shape()}")  # [2, 4]

# Redimensionar
reshaped = original.reshape([4, 2])
print(f"Redimensionado [4, 2]:\n{reshaped}")

# Transponer
transposed = original.transpose(0, 1)
print(f"Transpuesto:\n{transposed}")

# Adición/eliminación de dimensiones
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Antes de comprimir: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"Después de comprimir: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"Después de expandir: {expanded.shape()}")  # [1, 3]
```

### 4. Operaciones Estadísticas

```python
# Funciones estadísticas
data = rustorch.randn([3, 4])
print(f"Datos:\n{data}")

# Estadísticas básicas
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Media: {mean_val.item():.4f}")
print(f"Suma: {sum_val.item():.4f}")
print(f"Desviación estándar: {std_val.item():.4f}")
print(f"Varianza: {var_val.item():.4f}")
print(f"Máximo: {max_val.item():.4f}")
print(f"Mínimo: {min_val.item():.4f}")

# Estadísticas por dimensión específica
row_mean = data.mean(dim=1)  # Media de cada fila
col_sum = data.sum(dim=0)    # Suma de cada columna

print(f"Medias por fila: {row_mean}")
print(f"Sumas por columna: {col_sum}")
```

## Fundamentos de Diferenciación Automática

### 1. Cálculo de Gradientes

```python
# Ejemplo de diferenciación automática
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Tensor de entrada: {x}")

# Crear Variable
var_x = rustorch.autograd.Variable(x)

# Construir grafo computacional
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Salida: {y.data().item()}")

# Propagación hacia atrás
y.backward()

# Obtener gradiente
grad = var_x.grad()
print(f"Gradiente: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Grafos Computacionales Complejos

```python
# Ejemplo más complejo
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Función compleja: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), luego suma

print(f"Resultado: {z.data().item():.4f}")

# Propagación hacia atrás
z.backward()
grad = var_x.grad()
print(f"Gradiente: {grad}")
```

## Fundamentos de Redes Neuronales

### 1. Capa Lineal Simple

```python
# Crear capa lineal
linear_layer = rustorch.nn.Linear(3, 1)  # 3 entradas -> 1 salida

# Entrada aleatoria
input_data = rustorch.randn([2, 3])  # Tamaño de lote 2, 3 características
print(f"Entrada: {input_data}")

# Paso hacia adelante
output = linear_layer.forward(input_data)
print(f"Salida: {output}")

# Verificar parámetros
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Forma del peso: {weight.shape()}")
print(f"Peso: {weight}")
if bias is not None:
    print(f"Sesgo: {bias}")
```

### 2. Funciones de Activación

```python
# Varias funciones de activación
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

### 3. Funciones de Pérdida

```python
# Ejemplos de uso de funciones de pérdida
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Error cuadrático medio
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"Pérdida MSE: {loss_value.item():.6f}")

# Entropía cruzada (para clasificación)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Índices de clase

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Pérdida de Entropía Cruzada: {ce_loss_value.item():.6f}")
```

## Procesamiento de Datos

### 1. Conjuntos de Datos y Cargadores de Datos

```python
# Crear conjunto de datos
import numpy as np

# Generar datos de muestra
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 muestras, 4 características
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # Clasificación de 3 clases

# Convertir a tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Crear conjunto de datos
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Tamaño del conjunto de datos: {len(dataset)}")

# Crear cargador de datos
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Obtener lotes del cargador de datos
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Mostrar solo los primeros 3 lotes
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Lote {batch_idx}: Forma de entrada {inputs.shape()}, Forma del objetivo {targets.shape()}")
```

### 2. Transformaciones de Datos

```python
# Ejemplos de transformación de datos
data = rustorch.randn([10, 10])
print(f"Media de datos originales: {data.mean().item():.4f}")
print(f"Desviación estándar de datos originales: {data.std().item():.4f}")

# Transformación de normalización
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Media de datos normalizados: {normalized_data.mean().item():.4f}")
print(f"Desviación estándar de datos normalizados: {normalized_data.std().item():.4f}")
```

## Ejemplo de Entrenamiento Completo

### Regresión Lineal

```python
# Ejemplo completo de regresión lineal
import numpy as np

# Generar datos
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Convertir a tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Crear conjunto de datos y cargador
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Definir modelo
model = rustorch.nn.Linear(1, 1)  # 1 entrada -> 1 salida

# Función de pérdida y optimizador
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Bucle de entrenamiento
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
            
            # Poner gradientes a cero
            optimizer.zero_grad()
            
            # Paso hacia adelante
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Retropropagación (simplificada)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Época {epoch}: Pérdida = {avg_loss:.6f}")

print("¡Entrenamiento completado!")

# Parámetros finales
final_weight = model.weight()
final_bias = model.bias()
print(f"Peso aprendido: {final_weight.item():.4f} (verdadero: 2.0)")
if final_bias is not None:
    print(f"Sesgo aprendido: {final_bias.item():.4f} (verdadero: 1.0)")
```

## Solución de Problemas

### Problemas Comunes y Soluciones

1. **Problemas de Instalación**
```bash
# Si no se encuentra maturin
pip install --upgrade maturin

# Si Rust está desactualizado
rustup update

# Problemas del entorno de Python
python -m pip install --upgrade pip
```

2. **Errores de Tiempo de Ejecución**
```python
# Verificar formas de tensor
print(f"Forma del tensor: {tensor.shape()}")
print(f"Tipo de dato del tensor: {tensor.dtype()}")

# Ten cuidado con los tipos de datos en la conversión de NumPy
np_array = np.array(data, dtype=np.float32)  # float32 explícito
```

3. **Optimización del Rendimiento**
```python
# Compilar en modo release
# maturin develop --release

# Ajustar el tamaño del lote
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Lote más grande
```

## Próximos Pasos

1. **Prueba Ejemplos Avanzados**: Consulta ejemplos en `docs/examples/neural_networks/`
2. **Usa la API estilo Keras**: `rustorch.training.Model` para construcción de modelos más fácil
3. **Características de Visualización**: `rustorch.visualization` para visualizar progreso de entrenamiento
4. **Entrenamiento Distribuido**: `rustorch.distributed` para procesamiento paralelo

Documentación Detallada:
- [Referencia de la API de Python](../es/python_api_reference.md)
- [Documentación General](../es/python_bindings_overview.md)
- [Colección de Ejemplos](../examples/)