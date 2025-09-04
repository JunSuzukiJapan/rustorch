# Документация API RusTorch

## 📚 Полная Справка API

Этот документ предоставляет комплексную документацию API для RusTorch v0.5.15, организованную по модулям и функциональности. Включает унифицированную обработку ошибок с `RusTorchError` и `RusTorchResult<T>` для последовательного управления ошибками во всех 1060+ тестах. **Фаза 8 ЗАВЕРШЕНА** добавляет продвинутые утилиты тензоров, включая условные операции, индексирование и статистические функции. **Фаза 9 ЗАВЕРШЕНА** вводит комплексную систему сериализации с сохранением/загрузкой моделей, JIT-компиляцией и поддержкой множественных форматов, включая совместимость с PyTorch.

## 🏗️ Основная Архитектура

### Структура Модулей

```
rustorch/
├── tensor/              # Основные операции тензоров и структуры данных
├── nn/                  # Слои нейронных сетей и функции
├── autograd/            # Движок автоматического дифференцирования
├── optim/               # Оптимизаторы и планировщики скорости обучения
├── special/             # Специальные математические функции
├── distributions/       # Статистические распределения
├── vision/              # Преобразования компьютерного зрения
├── linalg/              # Операции линейной алгебры (BLAS/LAPACK)
├── gpu/                 # GPU ускорение (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Операции разреженных тензоров и обрезка (Фаза 12)
├── serialization/       # Сериализация моделей и JIT-компиляция (Фаза 9)
└── wasm/                # WebAssembly привязки (см. [Документация WASM API](WASM_API_DOCUMENTATION.md))
```

## 📊 Модуль Тензор

### Основное Создание Тензоров

```rust
use rustorch::tensor::Tensor;

// Основное создание
let tensor = Tensor::new(vec![2, 3]);               // Создание на основе формы
let tensor = Tensor::from_vec(data, vec![2, 3]);    // Из вектора данных
let tensor = Tensor::zeros(vec![10, 10]);           // Тензор, заполненный нулями
let tensor = Tensor::ones(vec![5, 5]);              // Тензор, заполненный единицами
let tensor = Tensor::randn(vec![3, 3]);             // Случайное нормальное распределение
let tensor = Tensor::rand(vec![3, 3]);              // Случайное равномерное распределение [0,1)
let tensor = Tensor::eye(5);                        // Единичная матрица
let tensor = Tensor::full(vec![2, 2], 3.14);       // Заполнить конкретным значением
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Тензор диапазона
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Линейное разнесение
```

### Операции Тензоров

```rust
// Арифметические операции
let result = a.add(&b);                             // Поэлементное сложение
let result = a.sub(&b);                             // Поэлементное вычитание
let result = a.mul(&b);                             // Поэлементное умножение
let result = a.div(&b);                             // Поэлементное деление
let result = a.pow(&b);                             // Поэлементное возведение в степень
let result = a.rem(&b);                             // Поэлементный остаток

// Матричные операции
let result = a.matmul(&b);                          // Матричное умножение
let result = a.transpose();                         // Транспонирование матрицы
let result = a.dot(&b);                             // Скалярное произведение

// Математические функции
let result = tensor.exp();                          // Экспонента
let result = tensor.ln();                           // Натуральный логарифм
let result = tensor.log10();                        // Логарифм по основанию 10
let result = tensor.sqrt();                         // Квадратный корень
let result = tensor.abs();                          // Абсолютное значение
let result = tensor.sin();                          // Функция синус
let result = tensor.cos();                          // Функция косинус
let result = tensor.tan();                          // Функция тангенс
let result = tensor.asin();                         // Арксинус
let result = tensor.acos();                         // Арккосинус
let result = tensor.atan();                         // Арктангенс
let result = tensor.sinh();                         // Гиперболический синус
let result = tensor.cosh();                         // Гиперболический косинус
let result = tensor.tanh();                         // Гиперболический тангенс
let result = tensor.floor();                        // Функция пола
let result = tensor.ceil();                         // Функция потолка
let result = tensor.round();                        // Функция округления
let result = tensor.sign();                         // Функция знака
let result = tensor.max();                          // Максимальное значение
let result = tensor.min();                          // Минимальное значение
let result = tensor.sum();                          // Сумма всех элементов
let result = tensor.mean();                         // Среднее значение
let result = tensor.std();                          // Стандартное отклонение
let result = tensor.var();                          // Дисперсия

// Манипуляция формой
let result = tensor.reshape(vec![6, 4]);            // Изменить форму тензора
let result = tensor.squeeze();                      // Удалить размерности размера-1
let result = tensor.unsqueeze(1);                   // Добавить размерность в индексе
let result = tensor.permute(vec![1, 0, 2]);         // Переставить размерности
let result = tensor.expand(vec![10, 10, 5]);        // Расширить размерности тензора
```

## 🧠 Модуль Нейронная Сеть (nn)

### Основные Слои

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// Линейный слой
let linear = Linear::new(784, 256)?;                // вход 784, выход 256
let output = linear.forward(&input)?;

// Сверточный слой
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// Пакетная нормализация
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### Функции Активации

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// Основные функции активации
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Параметризованные функции активации
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// Пример использования
let activated = relu.forward(&input)?;
```

## 🚀 Модуль GPU Ускорения

### Управление Устройствами

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// Проверить доступные устройства
let device_count = get_device_count()?;
let device = Device::best_available()?;            // Выбор лучшего устройства

// Конфигурация устройства
set_device(&device)?;

// Переместить тензор на GPU
let gpu_tensor = tensor.to_device(&device)?;
```

### CUDA Операции

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// Операции CUDA устройства
let cuda_device = CudaDevice::new(0)?;              // Использовать GPU 0
let stats = memory_stats(0)?;                      // Статистика памяти
println!("Используемая память: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 Модуль Оптимизатор (Optim)

### Основные Оптимизаторы

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Оптимизатор Adam
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// Оптимизатор SGD
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// Шаг оптимизации
optimizer.zero_grad()?;
// ... прямой расчет и обратное распространение ...
optimizer.step()?;
```

## 📖 Пример Использования

### Линейная Регрессия

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// Подготовка данных
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// Определение модели
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// Цикл обучения
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("Эпоха {}: Потеря = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ Известные Ограничения

1. **Ограничение памяти GPU**: Требуется явное управление памятью для больших тензоров (>8GB)
2. **Ограничение WebAssembly**: Некоторые операции BLAS недоступны в среде WASM
3. **Распределенное обучение**: Бэкенд NCCL поддерживается только на Linux
4. **Ограничение Metal**: Некоторые продвинутые операции доступны только с бэкендом CUDA

## 🔗 Связанные Ссылки

- [Основной README](../README.md)
- [Документация WASM API](WASM_API_DOCUMENTATION.md)
- [Руководство Jupyter](jupyter-guide.md)
- [GitHub Репозиторий](https://github.com/JunSuzukiJapan/RusTorch)
- [Пакет Crates.io](https://crates.io/crates/rustorch)

---

**Последнее Обновление**: v0.5.15 | **Лицензия**: MIT | **Автор**: Jun Suzuki