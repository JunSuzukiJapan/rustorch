# Руководство по RusTorch WASM Jupyter Notebook

Пошаговое руководство по простому использованию RusTorch WASM в Jupyter Notebook, разработанное для начинающих.

## 📚 Содержание

1. [Требования](#требования)
2. [Инструкции по настройке](#инструкции-по-настройке)
3. [Базовое использование](#базовое-использование)
4. [Практические примеры](#практические-примеры)
5. [Устранение неполадок](#устранение-неполадок)
6. [FAQ](#faq)

## Требования

### Минимальные требования
- **Python 3.8+**
- **Jupyter Notebook** или **Jupyter Lab**
- **Node.js 16+** (для WASM-сборок)
- **Rust** (последняя стабильная версия)
- **wasm-pack** (для конвертации Rust-кода в WASM)

### Рекомендуемая среда
- Память: 8GB или больше
- Браузер: Последние версии Chrome, Firefox, Safari
- ОС: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Инструкции по настройке

### 🚀 Быстрый старт (Рекомендуется)

**Самый простой способ**: Запустить Jupyter Lab одной командой
```bash
./start_jupyter.sh
```

Этот скрипт автоматически:
- Создаёт и активирует виртуальную среду
- Устанавливает зависимости (numpy, jupyter, matplotlib)
- Собирает Python-привязки RusTorch
- Запускает Jupyter Lab с открытым демо-notebook

### Ручная настройка

#### Шаг 1: Установить базовые инструменты

```bash
# Проверить версию Python
python --version

# Установить Rust (если не установлен)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Установить wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Проверить Node.js
node --version
npm --version
```

#### Шаг 2: Настроить Python-среду

```bash
# Создать виртуальную среду
python -m venv rustorch_env

# Активировать (Linux/macOS)
source rustorch_env/bin/activate

# Активировать (Windows)
rustorch_env\\Scripts\\activate

# Установить зависимости
pip install jupyter numpy matplotlib seaborn pandas
```

#### Шаг 3: Собрать RusTorch WASM

```bash
# Создать WASM-пакет
wasm-pack build --target web --features wasm

# Запустить Jupyter Lab
jupyter lab
```

## Базовое использование

### Загрузка WASM-модулей в Jupyter

```javascript
%%javascript
// Загрузить RusTorch WASM
import init, { WasmTensor, WasmAdvancedMath } from './pkg/rustorch.js';

async function setupRusTorch() {
    await init();
    
    // Базовый пример с тензором
    const data = [1.0, 2.0, 3.0, 4.0];
    const shape = [2, 2];
    const tensor = new WasmTensor(data, shape);
    
    console.log('Тензор создан:', tensor.data());
    console.log('Форма тензора:', tensor.shape());
    
    // Математические операции
    const math = new WasmAdvancedMath();
    const result = math.exp(tensor);
    console.log('Экспоненциальный результат:', result.data());
}

setupRusTorch();
```

### Интеграция с Python

```python
# Python-ячейка
import numpy as np
import matplotlib.pyplot as plt

# Подготовить данные для WASM
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(f"Входные данные: {data}")

# Визуализировать результаты
plt.figure(figsize=(10, 6))
plt.plot(data, 'o-', label='Оригинал')
plt.legend()
plt.title('Тензорные операции RusTorch WASM')
plt.show()
```

## Практические примеры

### 1. Арифметика тензоров

```javascript
%%javascript
// Демо тензорных операций
const a = new WasmTensor([1, 2, 3, 4], [2, 2]);
const b = new WasmTensor([2, 3, 4, 5], [2, 2]);

// Сложение
const sum = a.add(b);
console.log('Сложение:', sum.data());

// Поэлементное умножение
const product = a.multiply(b);
console.log('Умножение:', product.data());
```

### 2. Продвинутая математика

```javascript
%%javascript
const math = new WasmAdvancedMath();
const tensor = new WasmTensor([0.5, 1.0, 1.5, 2.0], [4]);

// Тригонометрические функции
const sin_result = math.sin(tensor);
const cos_result = math.cos(tensor);
const tan_result = math.tan(tensor);

console.log('Sin:', sin_result.data());
console.log('Cos:', cos_result.data());
console.log('Tan:', tan_result.data());
```

### 3. Метрики качества

```javascript
%%javascript
const quality = new WasmQualityMetrics(0.8);
const data_tensor = new WasmTensor([...Array(100)].map(() => Math.random()), [100]);

// Оценить качество данных
const quality_score = quality.overall_quality(data_tensor);
console.log('Показатель качества:', quality_score);

// Подробный отчёт
const report = quality.quality_report(data_tensor);
console.log('Отчёт о качестве:', report);
```

### 4. Обнаружение аномалий

```javascript
%%javascript
const detector = new WasmAnomalyDetector(2.0, 50);
const time_series = new WasmTimeSeriesDetector(30, 12);

// Обнаружить статистические аномалии
const anomalies = detector.detect_statistical(data_tensor);
console.log('Найденные аномалии:', anomalies.length());

// Обнаружение в реальном времени
for (let i = 0; i < data_tensor.data().length; i++) {
    const value = data_tensor.data()[i];
    const anomaly = detector.detect_realtime(value);
    if (anomaly) {
        console.log(`Аномалия в индексе ${i}: ${value}`);
    }
}
```

## Устранение неполадок

### Общие проблемы

#### Проблема: "Модуль не найден"
```bash
# Решение: Пересобрать WASM
wasm-pack build --target web --features wasm
```

#### Проблема: "Ошибки памяти"
```javascript
// Решение: Инициализировать Memory Manager
import { MemoryManager } from './pkg/rustorch.js';
MemoryManager.init_pool(200);  // Увеличить размер пула
```

#### Проблема: "Медленная производительность"
```javascript
// Решение: Включить кэш и использовать сборку мусора
const pipeline = new WasmTransformPipeline(true);  // Кэш включён
MemoryManager.gc();  // Освободить память
```

### Советы по отладке

1. **Проверить консоль браузера**: Открыть инструменты разработчика (F12)
2. **Мониторить использование памяти**: 
   ```javascript
   console.log('Статистика памяти:', MemoryManager.get_stats());
   ```
3. **Использовать обработчики ошибок**:
   ```javascript
   try {
       const result = tensor.operation();
   } catch (error) {
       console.error('Ошибка:', error.message);
   }
   ```

## FAQ

### В: Какие браузеры поддерживаются?
О: Chrome 90+, Firefox 89+, Safari 14+, Edge 90+

### В: Могу ли я использовать RusTorch WASM в Node.js?
О: Да, но это руководство фокусируется на использовании в браузере

### В: Каков размер WASM-bundle?
О: ~2-5MB сжатого, в зависимости от включённых функций

### В: Могу ли я использовать GPU-ускорение?
О: Да, с WebGPU в поддерживаемых браузерах (в основном Chrome)

### В: Готово ли это для продакшена?
О: Да, RusTorch полностью протестирован и оптимизирован для продакшн-сред

---

**🎯 Для продвинутых примеров см. директорию [examples/](../../examples/)**