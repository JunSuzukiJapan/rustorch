# Справочник Python API для RusTorch

Полный справочник по Python API RusTorch для разработчиков машинного обучения и глубокого обучения.

## Содержание

- [Модуль Tensor](#модуль-tensor)
- [Автоматическое дифференцирование](#автоматическое-дифференцирование)
- [Нейронные сети](#нейронные-сети)
- [Оптимизация](#оптимизация)
- [Компьютерное зрение](#компьютерное-зрение)
- [GPU и устройства](#gpu-и-устройства)
- [Утилиты](#утилиты)

## Модуль Tensor

### `Tensor`

Основная структура для операций с N-мерными тензорами.

#### Конструкторы

```python
import rustorch_py as torch

# Создание тензора из данных
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# Тензор с нулями
zeros = torch.zeros([2, 3], dtype=torch.float32)

# Тензор с единицами
ones = torch.ones([2, 3], dtype=torch.float32)

# Случайные тензоры (нормальное распределение)
randn = torch.randn([2, 3], dtype=torch.float32)

# Случайные тензоры (равномерное распределение)
rand = torch.rand([2, 3], dtype=torch.float32)
```

#### Базовые операции

```python
# Арифметические операции
result = tensor1.add(tensor2)
result = tensor1.sub(tensor2) 
result = tensor1.mul(tensor2)
result = tensor1.div(tensor2)

# Матричное умножение
result = tensor1.matmul(tensor2)

# Транспонирование
transposed = tensor.t()

# Изменение размерности
reshaped = tensor.reshape([6, 1])
```

#### Операции свёртки

```python
# Сумма
sum_all = tensor.sum()
sum_dim = tensor.sum(dim=0, keepdim=False)

# Среднее значение
mean_all = tensor.mean()
mean_dim = tensor.mean(dim=0, keepdim=False)

# Максимум и минимум
max_val, max_indices = tensor.max(dim=0)
min_val, min_indices = tensor.min(dim=0)
```

#### Индексирование и выбор

```python
# Выбор по индексам
slice_result = tensor.slice(dim=0, start=0, end=2, step=1)

# Выбор по условию
mask = tensor.gt(threshold)
selected = tensor.masked_select(mask)
```

## Автоматическое дифференцирование

### `Variable`

Обёртка для тензоров, позволяющая автоматическое дифференцирование.

```python
import rustorch_py.autograd as autograd

# Создание переменной с requires_grad=True
x = autograd.Variable(torch.randn([2, 2]), requires_grad=True)
y = autograd.Variable(torch.randn([2, 2]), requires_grad=True)

# Операции, строящие вычислительный граф
z = x.matmul(y)
loss = z.sum()

# Обратное распространение
loss.backward()

# Доступ к градиентам
x_grad = x.grad
print(f"Градиент x: {x_grad}")
```

### Функции дифференцирования

```python
# Пользовательская функция с градиентом
def custom_function(input_var):
    # Прямой проход
    output = input_var.pow(2.0)
    
    # Градиент будет вычислен автоматически
    return output

# Контекст без вычисления градиентов
with torch.no_grad():
    result = model.forward(input_data)
```

## Нейронные сети

### Базовые слои

#### `Linear`

Линейное преобразование (полносвязный слой).

```python
import rustorch_py.nn as nn

linear = nn.Linear(784, 256)  # вход: 784, выход: 256
input_tensor = torch.randn([32, 784])
output = linear.forward(input_tensor)
```

#### Функции активации

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

### Свёрточные слои

```python
# 2D свёртка
conv2d = nn.Conv2d(
    in_channels=3,     # входные каналы
    out_channels=64,   # выходные каналы
    kernel_size=3,     # размер ядра
    stride=1,          # шаг
    padding=1          # дополнение
)

input_tensor = torch.randn([1, 3, 224, 224])
output = conv2d.forward(input_tensor)
```

### Последовательные модели

```python
model = nn.Sequential([
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
])

# Прямой проход
input_data = torch.randn([32, 784])
output = model.forward(input_data)
```

## Оптимизация

### Оптимизаторы

#### `Adam`

```python
import rustorch_py.optim as optim

optimizer = optim.Adam(
    params=model.parameters(),  # параметры модели
    lr=0.001,                   # скорость обучения
    betas=(0.9, 0.999),        # коэффициенты beta
    eps=1e-8                   # epsilon
)

# Цикл обучения
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
    lr=0.01,                   # скорость обучения
    momentum=0.9               # импульс
)
```

### Функции потерь

```python
import rustorch_py.nn.functional as F

# Среднеквадратичная ошибка
mse_loss = F.mse_loss(prediction, target)

# Кросс-энтропия
ce_loss = F.cross_entropy(prediction, target)

# Бинарная кросс-энтропия
bce_loss = F.binary_cross_entropy(prediction, target)
```

## Компьютерное зрение

### Преобразования изображений

```python
import rustorch_py.vision.transforms as transforms

# Изменение размера изображения
resize = transforms.Resize((224, 224))
resized = resize.forward(image)

# Нормализация
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # среднее
    std=[0.229, 0.224, 0.225]    # стандартное отклонение
)
normalized = normalize.forward(image)

# Случайные преобразования
random_crop = transforms.RandomCrop(32, padding=4)
cropped = random_crop.forward(image)
```

### Предобученные модели

```python
import rustorch_py.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
output = resnet18.forward(input_tensor)

# VGG
vgg16 = models.vgg16(pretrained=True)
features = vgg16.features(input_tensor)
```

## GPU и устройства

### Управление устройствами

```python
import rustorch_py as torch

# CPU
cpu = torch.device('cpu')

# CUDA
cuda = torch.device('cuda:0')  # GPU 0
cuda_available = torch.cuda.is_available()

# Metal (macOS)
metal = torch.device('metal:0')

# Перемещение тензора на устройство
tensor_gpu = tensor.to(cuda)
```

### Многопроцессорные операции

```python
import rustorch_py.distributed as dist

# Инициализация распределённой обработки
dist.init_process_group("nccl", rank=0, world_size=2)

# AllReduce для синхронизации градиентов
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## Утилиты

### Сериализация

```python
import rustorch_py.serialize as serialize

# Сохранение модели
serialize.save(model, "model.pth")

# Загрузка модели
loaded_model = serialize.load("model.pth")
```

### Метрики

```python
import rustorch_py.metrics as metrics

# Точность
accuracy = metrics.accuracy(predictions, targets)

# F1-Score
f1 = metrics.f1_score(predictions, targets, average="macro")

# Матрица ошибок
confusion_matrix = metrics.confusion_matrix(predictions, targets)
```

### Утилиты для данных

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

## Полные примеры

### Классификация с CNN

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

# Создание модели
model = CNN()

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Функция потерь
criterion = nn.CrossEntropyLoss()

# Цикл обучения
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Эпоха {epoch}: Потери = {loss.item():.4f}")
```

Этот справочник представляет основные функции Python API RusTorch. For подробных примеров и продвинутых случаев использования см. [Полное руководство по Python-привязкам](python_bindings_overview.md) и [Руководство по Jupyter](jupyter-guide.md).