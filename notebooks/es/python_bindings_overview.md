# Resumen de Bindings Python de RusTorch

## Resumen

RusTorch es un framework de aprendizaje profundo de alto rendimiento implementado en Rust, que proporciona APIs similares a PyTorch mientras aprovecha los beneficios de seguridad y rendimiento de Rust. A través de bindings de Python, puedes acceder a la funcionalidad de RusTorch directamente desde Python.

## Características Clave

### 🚀 **Alto Rendimiento**
- **Núcleo Rust**: Logra rendimiento de nivel C++ mientras garantiza la seguridad de memoria
- **Soporte SIMD**: Vectorización automática para cálculos numéricos optimizados
- **Procesamiento Paralelo**: Cálculo paralelo eficiente usando rayon
- **Copia Cero**: Copia mínima de datos entre NumPy y RusTorch

### 🛡️ **Seguridad**
- **Seguridad de Memoria**: Previene fugas de memoria y condiciones de carrera a través del sistema de propiedad de Rust
- **Seguridad de Tipos**: Verificación de tipos en tiempo de compilación reduce errores en tiempo de ejecución
- **Manejo de Errores**: Manejo completo de errores con conversión automática a excepciones de Python

### 🎯 **Facilidad de Uso**
- **API Compatible con PyTorch**: Migración fácil desde código PyTorch existente
- **API de Alto Nivel estilo Keras**: Interfaces intuitivas como model.fit()
- **Integración NumPy**: Conversión bidireccional con arrays NumPy

## Arquitectura

Los bindings de Python de RusTorch consisten en 10 módulos:

### 1. **tensor** - Operaciones de Tensor
```python
import rustorch

# Creación de tensores
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# Integración NumPy
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - Diferenciación Automática
```python
# Cálculo de gradientes
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # Obtener gradientes
```

### 3. **nn** - Redes Neuronales
```python
# Creación de capas
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# Funciones de pérdida
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - Optimizadores
```python
# Optimizadores
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Programadores de tasa de aprendizaje
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - Carga de Datos
```python
# Creación de conjunto de datos
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Transformaciones de datos
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - API de Entrenamiento de Alto Nivel
```python
# API estilo Keras
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Ejecución de entrenamiento
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - Entrenamiento Distribuido
```python
# Configuración de entrenamiento distribuido
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# Paralelismo de datos
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - Visualización
```python
# Graficar historial de entrenamiento
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# Visualización de tensores
plotter.plot_tensor_as_image(tensor, title="Mapa de Características")
```

### 9. **utils** - Utilidades
```python
# Guardar/cargar modelo
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# Perfilado
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## Instalación

### Prerrequisitos
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (para uso de GPU)

### Compilar e Instalar
```bash
# Clonar repositorio
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Crear entorno virtual de Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install maturin numpy

# Compilar e instalar
maturin develop --release

# O instalar desde PyPI (planeado para el futuro)
# pip install rustorch
```

## Inicio Rápido

### 1. Operaciones Básicas de Tensor
```python
import rustorch
import numpy as np

# Creación de tensores
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Forma: {x.shape()}")  # Forma: [2, 2]

# Operaciones matemáticas
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Resultado: {z.to_numpy()}")
```

### 2. Ejemplo de Regresión Lineal
```python
import rustorch
import numpy as np

# Generar datos
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convertir a tensores
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Definir modelo
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# Crear conjunto de datos
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Entrenar
history = model.fit(dataloader, epochs=100, verbose=True)

# Mostrar resultados
print(f"Pérdida final: {history.train_loss()[-1]:.4f}")
```

### 3. Clasificación con Redes Neuronales
```python
import rustorch

# Preparar datos
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Construir modelo
model = rustorch.Model("RedClasificacion")
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

# Configuración de entrenamiento
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# Entrenar
history = trainer.train(model, train_loader, val_loader)

# Evaluar
metrics = model.evaluate(test_loader)
print(f"Precisión de prueba: {metrics['accuracy']:.4f}")
```

## Optimización de Rendimiento

### Utilización SIMD
```python
# Habilitar optimización SIMD durante la compilación
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # Cálculo optimizado con SIMD
```

### Uso de GPU
```python
# Uso de CUDA (planeado para el futuro)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # Cálculo en GPU
```

### Carga de Datos Paralela
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # Número de trabajadores paralelos
)
```

## Mejores Prácticas

### 1. Eficiencia de Memoria
```python
# Utilizar conversión de copia cero
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # Sin copia

# Usar operaciones in-place
tensor.add_(1.0)  # Eficiente en memoria
```

### 2. Manejo de Errores
```python
try:
    result = model(entrada_invalida)
except rustorch.RusTorchError as e:
    print(f"Error de RusTorch: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
```

### 3. Depuración y Perfilado
```python
# Usar perfilador
profiler = rustorch.utils.Profiler()
profiler.start()

# Ejecutar cálculo
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## Limitaciones

### Limitaciones Actuales
- **Soporte GPU**: Soporte CUDA/ROCm en desarrollo
- **Grafos Dinámicos**: Actualmente soporta solo grafos estáticos
- **Entrenamiento Distribuido**: Solo funcionalidad básica implementada

### Extensiones Futuras
- Aceleración GPU (CUDA, Metal, ROCm)
- Soporte para grafos de cálculo dinámicos
- Más capas de redes neuronales
- Cuantización y poda de modelos
- Funcionalidad de exportación ONNX

## Contribuir

### Participación en el Desarrollo
```bash
# Configurar entorno de desarrollo
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# Ejecutar pruebas
cargo test
python -m pytest tests/

# Verificaciones de calidad de código
cargo clippy
cargo fmt
```

### Comunidad
- GitHub Issues: Reportes de errores y solicitudes de características
- Discussions: Preguntas y discusiones
- Discord: Soporte en tiempo real

## Licencia

RusTorch se publica bajo la Licencia MIT. Libre para usar tanto para propósitos comerciales como no comerciales.

## Enlaces Relacionados

- [Repositorio GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Documentación API](./python_api_reference.md)
- [Ejemplos y Tutoriales](../examples/)
- [Benchmarks de Rendimiento](./benchmarks.md)