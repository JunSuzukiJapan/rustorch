# Referencia de API Python para RusTorch

Referencia completa de la API de Python para RusTorch dirigida a desarrolladores de aprendizaje automático y aprendizaje profundo.

## Índice

- [Módulo Tensor](#módulo-tensor)
- [Diferenciación Automática](#diferenciación-automática)
- [Redes Neuronales](#redes-neuronales)
- [Optimización](#optimización)
- [Visión por Computador](#visión-por-computador)
- [GPU y Dispositivos](#gpu-y-dispositivos)
- [Utilidades](#utilidades)

## Módulo Tensor

### `Tensor`

Estructura fundamental para operaciones de tensor N-dimensional.

#### Constructores

```python
import rustorch_py as torch

# Crear tensor desde datos
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# Tensor con ceros
zeros = torch.zeros([2, 3], dtype=torch.float32)

# Tensor con unos
ones = torch.ones([2, 3], dtype=torch.float32)

# Tensores aleatorios (distribución normal)
randn = torch.randn([2, 3], dtype=torch.float32)

# Tensores aleatorios (distribución uniforme)
rand = torch.rand([2, 3], dtype=torch.float32)
```

#### Operaciones Básicas

```python
# Operaciones aritméticas
result = tensor1.add(tensor2)
result = tensor1.sub(tensor2)
result = tensor1.mul(tensor2)
result = tensor1.div(tensor2)

# Multiplicación de matrices
result = tensor1.matmul(tensor2)

# Transposición
transposed = tensor.t()

# Cambio de forma
reshaped = tensor.reshape([6, 1])
```

#### Operaciones de Reducción

```python
# Suma
sum_all = tensor.sum()
sum_dim = tensor.sum(dim=0, keepdim=False)

# Media
mean_all = tensor.mean()
mean_dim = tensor.mean(dim=0, keepdim=False)

# Máximo y mínimo
max_val, max_indices = tensor.max(dim=0)
min_val, min_indices = tensor.min(dim=0)
```

#### Indexación y Selección

```python
# Selección por índices
slice_result = tensor.slice(dim=0, start=0, end=2, step=1)

# Selección por condición
mask = tensor.gt(threshold)
selected = tensor.masked_select(mask)
```

## Diferenciación Automática

### `Variable`

Envoltorio para tensores que permite diferenciación automática.

```python
import rustorch_py.autograd as autograd

# Crear variable con requires_grad=True
x = autograd.Variable(torch.randn([2, 2]), requires_grad=True)
y = autograd.Variable(torch.randn([2, 2]), requires_grad=True)

# Operaciones que construyen el grafo de cómputo
z = x.matmul(y)
loss = z.sum()

# Propagación hacia atrás
loss.backward()

# Acceder a gradientes
x_grad = x.grad
print(f"Gradiente de x: {x_grad}")
```

### Funciones de Diferenciación

```python
# Función personalizada con gradiente
def custom_function(input_var):
    # Pase hacia adelante
    output = input_var.pow(2.0)
    
    # El gradiente se calculará automáticamente
    return output

# Contexto sin cálculo de gradientes
with torch.no_grad():
    result = model.forward(input_data)
```

## Redes Neuronales

### Capas Básicas

#### `Linear`

Transformación lineal (capa completamente conectada).

```python
import rustorch_py.nn as nn

linear = nn.Linear(784, 256)  # entrada: 784, salida: 256
input_tensor = torch.randn([32, 784])
output = linear.forward(input_tensor)
```

#### Funciones de Activación

```python
# ReLU
relu = nn.ReLU()
output = relu.forward(input_tensor)

# Sigmoid
sigmoid = nn.Sigmoid()
output = sigmoid.forward(input_tensor)

# Tanh
tanh = nn.Tanh()
output = tanh.forward(input_tensor)

# GELU
gelu = nn.GELU()
output = gelu.forward(input_tensor)
```

### Capas Convolucionales

```python
# Convolución 2D
conv2d = nn.Conv2d(
    in_channels=3,     # canales de entrada
    out_channels=64,   # canales de salida
    kernel_size=3,     # tamaño del kernel
    stride=1,          # paso
    padding=1          # relleno
)

input_tensor = torch.randn([1, 3, 224, 224])
output = conv2d.forward(input_tensor)
```

### Modelos Secuenciales

```python
model = nn.Sequential([
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
])

# Pase hacia adelante
input_data = torch.randn([32, 784])
output = model.forward(input_data)
```

## Optimización

### Optimizadores

#### `Adam`

```python
import rustorch_py.optim as optim

optimizer = optim.Adam(
    params=model.parameters(),  # parámetros del modelo
    lr=0.001,                   # tasa de aprendizaje
    betas=(0.9, 0.999),        # coeficientes beta
    eps=1e-8                   # epsilon
)

# Bucle de entrenamiento
for batch in data_loader:
    prediction = model.forward(batch.input)
    loss = criterion(prediction, batch.target)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### `SGD`

```python
optimizer = optim.SGD(
    params=model.parameters(),
    lr=0.01,                   # tasa de aprendizaje
    momentum=0.9               # momento
)
```

### Funciones de Pérdida

```python
import rustorch_py.nn.functional as F

# Error cuadrático medio
mse_loss = F.mse_loss(prediction, target)

# Entropía cruzada
ce_loss = F.cross_entropy(prediction, target)

# Entropía cruzada binaria
bce_loss = F.binary_cross_entropy(prediction, target)
```

## Visión por Computador

### Transformaciones de Imagen

```python
import rustorch_py.vision.transforms as transforms

# Redimensionar imagen
resize = transforms.Resize((224, 224))
resized = resize.forward(image)

# Normalización
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # media
    std=[0.229, 0.224, 0.225]    # desviación estándar
)
normalized = normalize.forward(image)

# Transformaciones aleatorias
random_crop = transforms.RandomCrop(32, padding=4)
cropped = random_crop.forward(image)
```

### Modelos Preentrenados

```python
import rustorch_py.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
output = resnet18.forward(input_tensor)

# VGG
vgg16 = models.vgg16(pretrained=True)
features = vgg16.features(input_tensor)
```

## GPU y Dispositivos

### Gestión de Dispositivos

```python
import rustorch_py as torch

# CPU
cpu = torch.device('cpu')

# CUDA
cuda = torch.device('cuda:0')  # GPU 0
cuda_available = torch.cuda.is_available()

# Metal (macOS)
metal = torch.device('metal:0')

# Mover tensor a dispositivo
tensor_gpu = tensor.to(cuda)
```

### Operaciones Multi-GPU

```python
import rustorch_py.distributed as dist

# Inicialización de procesamiento distribuido
dist.init_process_group("nccl", rank=0, world_size=2)

# AllReduce para sincronización de gradientes
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## Utilidades

### Serialización

```python
import rustorch_py.serialize as serialize

# Guardar modelo
serialize.save(model, "model.pth")

# Cargar modelo
loaded_model = serialize.load("model.pth")
```

### Métricas

```python
import rustorch_py.metrics as metrics

# Precisión
accuracy = metrics.accuracy(predictions, targets)

# F1-Score
f1 = metrics.f1_score(predictions, targets, average="macro")

# Matriz de confusión
confusion_matrix = metrics.confusion_matrix(predictions, targets)
```

### Utilidades de Datos

```python
import rustorch_py.data as data

# DataLoader
dataset = data.TensorDataset(inputs, targets)
data_loader = data.DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

for batch in data_loader:
    loss = train_step(batch)
```

## Ejemplos Completos

### Clasificación con CNN

```python
import rustorch_py as torch
import rustorch_py.nn as nn
import rustorch_py.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Crear modelo
model = CNN()

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función de pérdida
criterion = nn.CrossEntropyLoss()

# Bucle de entrenamiento
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Pase hacia adelante
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Pase hacia atrás
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Época {epoch}: Pérdida = {loss.item():.4f}")
```

Esta referencia cubre las funcionalidades principales de la API de Python para RusTorch. Para ejemplos más detallados y casos de uso avanzados, consulte la [Guía Completa de Enlaces de Python](python_bindings_overview.md) y la [Guía de Jupyter](jupyter-guide.md).