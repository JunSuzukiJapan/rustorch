# Guía RusTorch WASM Jupyter Notebook

Una guía paso a paso para usar fácilmente RusTorch WASM en Jupyter Notebook, diseñada para principiantes.

## 📚 Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Instrucciones de Instalación](#instrucciones-de-instalación)
3. [Uso Básico](#uso-básico)
4. [Ejemplos Prácticos](#ejemplos-prácticos)
5. [Solución de Problemas](#solución-de-problemas)
6. [FAQ](#faq)

## Requisitos

### Requisitos Mínimos
- **Python 3.8+**
- **Jupyter Notebook** o **Jupyter Lab**
- **Node.js 16+** (para builds WASM)
- **Rust** (última versión estable)
- **wasm-pack** (para convertir código Rust a WASM)

### Entorno Recomendado
- Memoria: 8GB o más
- Navegador: Últimas versiones de Chrome, Firefox, Safari
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Instrucciones de Instalación

### 🚀 Inicio Rápido (Recomendado)

**Método más fácil**: Lanza Jupyter Lab con un comando
```bash
./start_jupyter.sh
```

Este script automáticamente:
- Crea y activa entorno virtual
- Instala dependencias (numpy, jupyter, matplotlib)
- Construye bindings Python de RusTorch
- Lanza Jupyter Lab con notebook demo abierto

### Instalación Manual

#### Paso 1: Instalar Herramientas Básicas

```bash
# Verificar versión Python
python --version

# Instalar Jupyter Lab
pip install jupyterlab

# Instalar Node.js (macOS con Homebrew)
brew install node

# Instalar Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Instalar wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### Paso 2: Construir RusTorch WASM

```bash
# Clonar proyecto
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Añadir target WASM
rustup target add wasm32-unknown-unknown

# Construir con wasm-pack
wasm-pack build --target web --out-dir pkg
```

#### Paso 3: Iniciar Jupyter

```bash
# Iniciar Jupyter Lab
jupyter lab
```

## Uso Básico

### Crear Tensores

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Tensor 1D
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('Tensor 1D:', vec.to_array());
    
    // Tensor 2D (matriz)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // forma: 2 filas, 3 columnas
    );
    console.log('Forma tensor 2D:', matrix.shape());
});
```

### Operaciones Básicas

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // Suma
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // Multiplicación matricial
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### Diferenciación Automática

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Crear tensor con seguimiento de gradiente
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // Cálculo: y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // Retropropagación
    y.backward();
    
    // Obtener gradiente (dy/dx = 2x + 3 = 7 cuando x=2)
    console.log('Gradiente:', x.grad().to_array());
});
```

## Ejemplos Prácticos

### Regresión Lineal

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Preparar datos
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // Inicializar parámetros
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // Bucle de entrenamiento
    for (let epoch = 0; epoch < 100; epoch++) {
        // Predicción: y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // Pérdida: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // Calcular gradientes
        loss.backward();
        
        // Actualizar parámetros
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // Reiniciar gradientes
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`Época ${epoch}: Pérdida = ${loss.item()}`);
        }
    }
    
    console.log(`w final: ${w.item()}, b final: ${b.item()}`);
});
```

## Solución de Problemas

### Errores Comunes

#### Error "RusTorch is not defined"
**Solución**: Siempre esperar RusTorchReady
```javascript
window.RusTorchReady.then((rt) => {
    // Usar RusTorch aquí
});
```

#### Error "Failed to load WASM module"
**Soluciones**:
1. Verificar que directorio `pkg` se generó correctamente
2. Revisar consola del navegador para mensajes de error
3. Asegurar que rutas de archivos WASM sean correctas

#### Error Falta de Memoria
**Soluciones**:
```javascript
// Liberar memoria explícitamente
tensor.free();

// Usar tamaños de lote más pequeños
const batchSize = 32;  // Usar 32 en lugar de 1000
```

### Consejos de Rendimiento

1. **Usar Procesamiento por Lotes**: Procesar datos en lotes en lugar de bucles
2. **Gestión de Memoria**: Liberar explícitamente tensores grandes
3. **Tipos de Datos Apropiados**: Usar f32 cuando alta precisión no es necesaria

## FAQ

### P: ¿Puedo usar esto en Google Colab?
**R**: Sí, sube los archivos WASM y usa cargadores JavaScript personalizados.

### P: ¿Puedo mezclar código Python y WASM?
**R**: Sí, usa IPython.display.Javascript para pasar datos entre Python y JavaScript.

### P: ¿Cómo hago depuración?
**R**: Usa herramientas de desarrollador del navegador (F12) y revisa la pestaña Console para errores.

### P: ¿Qué características avanzadas están disponibles?
**R**: Actualmente soporta operaciones tensoriales básicas, diferenciación automática y redes neuronales simples. Capas CNN y RNN están planificadas.

## Próximos Pasos

1. 📖 [API RusTorch WASM Detallada](../wasm.md)
2. 🔬 [Ejemplos Avanzados](../examples/)
3. 🚀 [Guía Optimización Rendimiento](../wasm-memory-optimization.md)

## Comunidad y Soporte

- GitHub: [Repositorio RusTorch](https://github.com/JunSuzukiJapan/rustorch)
- Issues: Reporta bugs y solicita características en GitHub

---

¡Feliz Aprendizaje con RusTorch WASM! 🦀🔥📓