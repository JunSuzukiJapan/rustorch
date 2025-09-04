# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-1128%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Готовая к продакшену библиотека глубокого обучения на Rust с PyTorch-подобным API, GPU ускорением и производительностью корпоративного уровня**

RusTorch — полнофункциональная библиотека глубокого обучения, использующая безопасность и производительность Rust, предоставляющая комплексные операции с тензорами, автоматическое дифференцирование, слои нейронных сетей, архитектуры трансформеров, мульти-бэкенд GPU ускорение (CUDA/Metal/OpenCL), продвинутые SIMD оптимизации, корпоративное управление памятью, валидацию данных и контроль качества, а также комплексные системы отладки и логирования.

## 📚 Документация

- **[Полная Справка API](API_DOCUMENTATION.md)** - Комплексная API документация для всех модулей
- **[Справка WASM API](WASM_API_DOCUMENTATION.md)** - WebAssembly-специфическая API документация
- **[Руководство Jupyter](jupyter-guide.md)** - Инструкция по использованию Jupyter Notebook

## ✨ Возможности

- 🔥 **Комплексные Операции с Тензорами**: Математические операции, broadcasting, индексирование и статистика, продвинутые утилиты Фазы 8
- 🤖 **Архитектура Трансформер**: Полная реализация трансформера с мульти-головым вниманием
- 🧮 **Матричное Разложение**: SVD, QR, разложение собственных значений с совместимостью PyTorch
- 🧠 **Автоматическое Дифференцирование**: Вычислительный граф на основе ленты для расчета градиентов
- 🚀 **Динамический Движок Выполнения**: JIT-компиляция и оптимизация времени выполнения
- 🏗️ **Слои Нейронных Сетей**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout и другие
- ⚡ **Кроссплатформенные Оптимизации**: SIMD (AVX2/SSE/NEON), платформо-специфичные и аппаратно-ориентированные оптимизации
- 🎮 **Интеграция с GPU**: Поддержка CUDA/Metal/OpenCL с автоматическим выбором устройства
- 🌐 **Поддержка WebAssembly**: Полное машинное обучение в браузере со слоями нейронных сетей, компьютерным зрением и инференсом в реальном времени
- 🎮 **Интеграция WebGPU**: GPU ускорение, оптимизированное для Chrome, с CPU fallback для кроссбраузерной совместимости
- 📁 **Поддержка Форматов Моделей**: Safetensors, ONNX инференс, совместимость PyTorch state dict
- ✅ **Готово к Продакшену**: 1128 пройденных тестов, унифицированная система обработки ошибок
- 🎯 **Утилиты Тензоров Фазы 8**: Условные операции (where, masked_select, masked_fill), операции индексирования (gather, scatter, index_select), статистические операции (topk, kthvalue) и продвинутые утилиты (unique, histogram)

## 🚀 Быстрый Старт

**📓 Для полного руководства по настройке Jupyter см. [README_JUPYTER.md](../../README_JUPYTER.md)**

### Демо Python Jupyter Lab

📓 **[Полное Руководство Настройки Jupyter](../../README_JUPYTER.md)** | **[Руководство Jupyter](jupyter-guide.md)**

#### Стандартная CPU Демо
Запуск RusTorch с Jupyter Lab одной командой:

```bash
./start_jupyter.sh
```

#### WebGPU Ускоренная Демо
Запуск RusTorch с поддержкой WebGPU для браузерного GPU ускорения:

```bash
./start_jupyter_webgpu.sh
```

### Использование Rust

```rust
use rustorch::tensor::Tensor;
use rustorch::nn::{Linear, ReLU};
use rustorch::optim::Adam;

// Создание тензора
let x = Tensor::randn(vec![32, 784]); // Размер батча 32, признаки 784
let y = Tensor::randn(vec![32, 10]);  // 10 классов

// Определение нейронной сети
let linear1 = Linear::new(784, 256)?;
let relu = ReLU::new();
let linear2 = Linear::new(256, 10)?;

// Прямой проход
let z1 = linear1.forward(&x)?;
let a1 = relu.forward(&z1)?;
let output = linear2.forward(&a1)?;

// Оптимизатор
let mut optimizer = Adam::new(
    vec![linear1.weight(), linear2.weight()], 
    0.001, 0.9, 0.999, 1e-8
)?;
```

## 🧪 Тестирование

### Запуск всех тестов
```bash
cargo test --lib
```

### Тесты по функциям
```bash
cargo test tensor::     # Тесты операций тензоров
cargo test nn::         # Тесты нейронных сетей
cargo test autograd::   # Тесты автоматического дифференцирования
cargo test optim::      # Тесты оптимизаторов
cargo test gpu::        # Тесты GPU операций
```

## 🔧 Установка

### Cargo.toml
```toml
[dependencies]
rustorch = "0.5.15"

# GPU функции
rustorch = { version = "0.5.15", features = ["cuda"] }      # CUDA
rustorch = { version = "0.5.15", features = ["metal"] }     # Metal (macOS)
rustorch = { version = "0.5.15", features = ["opencl"] }    # OpenCL

# WebAssembly
rustorch = { version = "0.5.15", features = ["wasm"] }      # WASM Basic
rustorch = { version = "0.5.15", features = ["webgpu"] }    # WebGPU
```

## ⚠️ Известные Ограничения

1. **Ограничение памяти GPU**: Требуется явное управление памятью для больших тензоров (>8GB)
2. **Ограничение WebAssembly**: Некоторые операции BLAS недоступны в среде WASM
3. **Распределенное обучение**: Бэкенд NCCL поддерживается только на Linux
4. **Ограничение Metal**: Некоторые продвинутые операции доступны только с бэкендом CUDA

## 🤝 Вклад

Pull requests и issues приветствуются! См. [CONTRIBUTING.md](../../CONTRIBUTING.md) для деталей.

## 📄 Лицензия

MIT Лицензия - см. [LICENSE](../../LICENSE) для деталей.

---

**Разработчик**: Jun Suzuki | **Версия**: v0.5.15 | **Последнее Обновление**: 2025