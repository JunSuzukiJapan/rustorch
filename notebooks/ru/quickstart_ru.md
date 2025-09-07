# Руководство по быстрому началу работы с RusTorch

## Установка

### 1. Предварительные требования
```bash
# Rust 1.70 или новее
rustc --version

# Python 3.8 или новее
python --version

# Установка необходимых зависимостей
pip install maturin numpy matplotlib
```

### 2. Сборка и установка RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Создание виртуальной среды Python (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Сборка и установка в режиме разработки
maturin develop --release
```

## Основные примеры использования

### 1. Создание тензоров и базовые операции

```python
import rustorch
import numpy as np

# Создание тензора
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Тензор x:\n{x}")
print(f"Форма: {x.shape()}")  # [2, 2]

# Нулевые матрицы и единичные матрицы
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Нулевая матрица:\n{zeros}")
print(f"Единичная матрица:\n{ones}")
print(f"Матрица идентичности:\n{identity}")

# Случайные тензоры
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Нормальное распределение:\n{random_normal}")
print(f"Равномерное распределение:\n{random_uniform}")

# Интеграция с NumPy
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"Из NumPy:\n{tensor_from_numpy}")

# Обратное преобразование в NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"Обратно в NumPy:\n{back_to_numpy}")
```

### 2. Арифметические операции

```python
# Базовые арифметические операции
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Поэлементные операции
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (поэлементное)
div_result = a.div(b)  # a / b (поэлементное)

print(f"Сложение:\n{add_result}")
print(f"Вычитание:\n{sub_result}")
print(f"Умножение:\n{mul_result}")
print(f"Деление:\n{div_result}")

# Скалярные операции
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Скалярное сложение (+2):\n{scalar_add}")
print(f"Скалярное умножение (*3):\n{scalar_mul}")

# Матричное умножение
matmul_result = a.matmul(b)
print(f"Матричное умножение:\n{matmul_result}")

# Математические функции
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Квадратный корень:\n{sqrt_result}")
print(f"Экспонента:\n{exp_result}")
print(f"Натуральный логарифм:\n{log_result}")
```

### 3. Манипуляции с формой тензоров

```python
# Примеры манипуляции формы
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Исходная форма: {original.shape()}")  # [2, 4]

# Изменение формы
reshaped = original.reshape([4, 2])
print(f"Измененная форма [4, 2]:\n{reshaped}")

# Транспонирование
transposed = original.transpose(0, 1)
print(f"Транспонированная:\n{transposed}")

# Добавление/удаление измерений
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"До squeeze: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"После squeeze: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"После unsqueeze: {expanded.shape()}")  # [1, 3]
```

### 4. Статистические операции

```python
# Статистические функции
data = rustorch.randn([3, 4])
print(f"Данные:\n{data}")

# Основная статистика
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Среднее: {mean_val.item():.4f}")
print(f"Сумма: {sum_val.item():.4f}")
print(f"Стандартное отклонение: {std_val.item():.4f}")
print(f"Дисперсия: {var_val.item():.4f}")
print(f"Максимум: {max_val.item():.4f}")
print(f"Минимум: {min_val.item():.4f}")

# Статистика по определенным измерениям
row_mean = data.mean(dim=1)  # Среднее каждой строки
col_sum = data.sum(dim=0)    # Сумма каждого столбца

print(f"Средние по строкам: {row_mean}")
print(f"Суммы по столбцам: {col_sum}")
```

## Основы автоматического дифференцирования

### 1. Вычисление градиентов

```python
# Пример автоматического дифференцирования
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Входной тензор: {x}")

# Создание переменной
var_x = rustorch.autograd.Variable(x)

# Построение вычислительного графа
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Результат: {y.data().item()}")

# Обратное распространение
y.backward()

# Получение градиента
grad = var_x.grad()
print(f"Градиент: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Сложные вычислительные графы

```python
# Более сложный пример
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Сложная функция: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), затем сумма

print(f"Результат: {z.data().item():.4f}")

# Обратное распространение
z.backward()
grad = var_x.grad()
print(f"Градиент: {grad}")
```

## Основы нейронных сетей

### 1. Простой линейный слой

```python
# Создание линейного слоя
linear_layer = rustorch.nn.Linear(3, 1)  # 3 входа -> 1 выход

# Случайные входные данные
input_data = rustorch.randn([2, 3])  # Размер пакета 2, 3 признака
print(f"Вход: {input_data}")

# Прямое распространение
output = linear_layer.forward(input_data)
print(f"Выход: {output}")

# Проверка параметров
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Форма весов: {weight.shape()}")
print(f"Веса: {weight}")
if bias is not None:
    print(f"Смещение: {bias}")
```

### 2. Функции активации

```python
# Различные функции активации
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

### 3. Функции потерь

```python
# Примеры использования функций потерь
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Средняя квадратическая ошибка
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE потеря: {loss_value.item():.6f}")

# Перекрестная энтропия (для классификации)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Индексы классов

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Потеря перекрестной энтропии: {ce_loss_value.item():.6f}")
```

## Обработка данных

### 1. Наборы данных и загрузчики данных

```python
# Создание набора данных
import numpy as np

# Генерация примерных данных
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 образцов, 4 признака
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3-классовая классификация

# Преобразование в тензоры
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Создание набора данных
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Размер набора данных: {len(dataset)}")

# Создание загрузчика данных
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Получение пакетов из загрузчика данных
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Показать только первые 3 пакета
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Пакет {batch_idx}: Форма входа {inputs.shape()}, Форма цели {targets.shape()}")
```

### 2. Преобразования данных

```python
# Примеры преобразования данных
data = rustorch.randn([10, 10])
print(f"Исходное среднее данных: {data.mean().item():.4f}")
print(f"Исходное стд. откл. данных: {data.std().item():.4f}")

# Преобразование нормализации
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Нормализованное среднее данных: {normalized_data.mean().item():.4f}")
print(f"Нормализованное стд. откл. данных: {normalized_data.std().item():.4f}")
```

## Полный пример обучения

### Линейная регрессия

```python
# Полный пример линейной регрессии
import numpy as np

# Генерация данных
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Преобразование в тензоры
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Создание набора данных и загрузчика данных
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Определение модели
model = rustorch.nn.Linear(1, 1)  # 1 вход -> 1 выход

# Функция потерь и оптимизатор
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Цикл обучения
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
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямое распространение
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Обратное распространение (упрощено)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Эпоха {epoch}: Потеря = {avg_loss:.6f}")

print("Обучение завершено!")

# Финальные параметры
final_weight = model.weight()
final_bias = model.bias()
print(f"Изученный вес: {final_weight.item():.4f} (истинный: 2.0)")
if final_bias is not None:
    print(f"Изученное смещение: {final_bias.item():.4f} (истинное: 1.0)")
```

## Устранение неполадок

### Общие проблемы и решения

1. **Проблемы с установкой**
```bash
# Если maturin не найден
pip install --upgrade maturin

# Если Rust устарел
rustup update

# Проблемы с Python окружением
python -m pip install --upgrade pip
```

2. **Ошибки времени выполнения**
```python
# Проверка формы тензоров
print(f"Форма тензора: {tensor.shape()}")
print(f"Тип данных тензора: {tensor.dtype()}")

# Будьте осторожны с типами данных при конверсии NumPy
np_array = np.array(data, dtype=np.float32)  # Явный float32
```

3. **Оптимизация производительности**
```python
# Сборка в релиз режиме
# maturin develop --release

# Настройка размера пакета
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Больший пакет
```

## Следующие шаги

1. **Попробуйте продвинутые примеры**: Проверьте примеры в `docs/examples/neural_networks/`
2. **Используйте API в стиле Keras**: `rustorch.training.Model` для более простого создания моделей
3. **Возможности визуализации**: `rustorch.visualization` для визуализации прогресса обучения
4. **Распределенное обучение**: `rustorch.distributed` для параллельной обработки

Подробная документация:
- [Справочник Python API](../en/python_api_reference.md)
- [Обзорная документация](../en/python_bindings_overview.md)
- [Коллекция примеров](../examples/)