# RusTorch Schnellstart-Anleitung

## Installation

### 1. Voraussetzungen
```bash
# Rust 1.70 oder neuer
rustc --version

# Python 3.8 oder neuer
python --version

# Installation der erforderlichen Abhängigkeiten
pip install maturin numpy matplotlib
```

### 2. RusTorch erstellen und installieren
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Im Entwicklungsmodus erstellen und installieren
maturin develop --release
```

## Grundlegende Verwendungsbeispiele

### 1. Tensor-Erstellung und grundlegende Operationen

```python
import rustorch
import numpy as np

# Tensor-Erstellung
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tensor x:\n{x}")
print(f"Form: {x.shape()}")  # [2, 2]

# Nullmatrizen und Einheitsmatrizen
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Nullmatrix:\n{zeros}")
print(f"Eins-Matrix:\n{ones}")
print(f"Einheitsmatrix:\n{identity}")

# Zufällige Tensoren
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Zufällig normal:\n{random_normal}")
print(f"Zufällig uniform:\n{random_uniform}")

# NumPy-Integration
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"Aus NumPy:\n{tensor_from_numpy}")

# Zurück zu NumPy konvertieren
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"Zurück zu NumPy:\n{back_to_numpy}")
```

### 2. Arithmetische Operationen

```python
# Grundlegende arithmetische Operationen
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Elementweise Operationen
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (elementweise)
div_result = a.div(b)  # a / b (elementweise)

print(f"Addition:\n{add_result}")
print(f"Subtraktion:\n{sub_result}")
print(f"Multiplikation:\n{mul_result}")
print(f"Division:\n{div_result}")

# Skalar-Operationen
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Skalar-Addition (+2):\n{scalar_add}")
print(f"Skalar-Multiplikation (*3):\n{scalar_mul}")

# Matrixmultiplikation
matmul_result = a.matmul(b)
print(f"Matrixmultiplikation:\n{matmul_result}")

# Mathematische Funktionen
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Quadratwurzel:\n{sqrt_result}")
print(f"Exponentialfunktion:\n{exp_result}")
print(f"Natürlicher Logarithmus:\n{log_result}")
```

### 3. Tensor-Formmanipulation

```python
# Beispiele für Formmanipulation
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Ursprüngliche Form: {original.shape()}")  # [2, 4]

# Umformen
reshaped = original.reshape([4, 2])
print(f"Umgeformt [4, 2]:\n{reshaped}")

# Transponieren
transposed = original.transpose(0, 1)
print(f"Transponiert:\n{transposed}")

# Dimensionen hinzufügen/entfernen
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Vor Squeeze: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"Nach Squeeze: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"Nach Unsqueeze: {expanded.shape()}")  # [1, 3]
```

### 4. Statistische Operationen

```python
# Statistische Funktionen
data = rustorch.randn([3, 4])
print(f"Daten:\n{data}")

# Grundlegende Statistiken
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Mittelwert: {mean_val.item():.4f}")
print(f"Summe: {sum_val.item():.4f}")
print(f"Standardabweichung: {std_val.item():.4f}")
print(f"Varianz: {var_val.item():.4f}")
print(f"Maximum: {max_val.item():.4f}")
print(f"Minimum: {min_val.item():.4f}")

# Statistiken entlang bestimmter Dimensionen
row_mean = data.mean(dim=1)  # Mittelwert jeder Zeile
col_sum = data.sum(dim=0)    # Summe jeder Spalte

print(f"Zeilenmittelwerte: {row_mean}")
print(f"Spaltensummen: {col_sum}")
```

## Grundlagen der automatischen Differentiation

### 1. Gradientenberechnung

```python
# Beispiel für automatische Differentiation
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Eingabetensor: {x}")

# Variable erstellen
var_x = rustorch.autograd.Variable(x)

# Berechnungsgraph aufbauen
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Ausgabe: {y.data().item()}")

# Rückwärtspropagation
y.backward()

# Gradient erhalten
grad = var_x.grad()
print(f"Gradient: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Komplexe Berechnungsgraphen

```python
# Komplexeres Beispiel
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Komplexe Funktion: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), dann summieren

print(f"Ergebnis: {z.data().item():.4f}")

# Rückwärtspropagation
z.backward()
grad = var_x.grad()
print(f"Gradient: {grad}")
```

## Grundlagen neuronaler Netze

### 1. Einfache lineare Schicht

```python
# Lineare Schicht erstellen
linear_layer = rustorch.nn.Linear(3, 1)  # 3 Eingänge -> 1 Ausgabe

# Zufällige Eingabe
input_data = rustorch.randn([2, 3])  # Stapelgröße 2, 3 Merkmale
print(f"Eingabe: {input_data}")

# Vorwärtsdurchgang
output = linear_layer.forward(input_data)
print(f"Ausgabe: {output}")

# Parameter überprüfen
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Gewichtsform: {weight.shape()}")
print(f"Gewicht: {weight}")
if bias is not None:
    print(f"Bias: {bias}")
```

### 2. Aktivierungsfunktionen

```python
# Verschiedene Aktivierungsfunktionen
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

### 3. Verlustfunktionen

```python
# Beispiele für die Verwendung von Verlustfunktionen
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Mittlerer quadratischer Fehler
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE-Verlust: {loss_value.item():.6f}")

# Kreuzentropie (für Klassifikation)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Klassenindizes

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Kreuzentropie-Verlust: {ce_loss_value.item():.6f}")
```

## Datenverarbeitung

### 1. Datensätze und DataLoader

```python
# Datensatz erstellen
import numpy as np

# Beispieldaten generieren
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 Stichproben, 4 Merkmale
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3-Klassen-Klassifikation

# Zu Tensoren konvertieren
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Datensatz erstellen
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Datensatzgröße: {len(dataset)}")

# DataLoader erstellen
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Stapel aus DataLoader erhalten
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Nur erste 3 Stapel zeigen
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Stapel {batch_idx}: Eingabeform {inputs.shape()}, Zielform {targets.shape()}")
```

### 2. Datentransformationen

```python
# Beispiele für Datentransformationen
data = rustorch.randn([10, 10])
print(f"Ursprünglicher Datenmittelwert: {data.mean().item():.4f}")
print(f"Ursprüngliche Datenstandardabweichung: {data.std().item():.4f}")

# Normalisierungstransformation
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Normalisierter Datenmittelwert: {normalized_data.mean().item():.4f}")
print(f"Normalisierte Datenstandardabweichung: {normalized_data.std().item():.4f}")
```

## Vollständiges Trainingsbeispiel

### Lineare Regression

```python
# Vollständiges Beispiel für lineare Regression
import numpy as np

# Daten generieren
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Zu Tensoren konvertieren
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Datensatz und DataLoader erstellen
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Modell definieren
model = rustorch.nn.Linear(1, 1)  # 1 Eingabe -> 1 Ausgabe

# Verlustfunktion und Optimierer
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Trainingsschleife
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
            
            # Gradienten nullsetzen
            optimizer.zero_grad()
            
            # Vorwärtsdurchgang
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Rückwärtspropagation (vereinfacht)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Epoche {epoch}: Verlust = {avg_loss:.6f}")

print("Training abgeschlossen!")

# Endgültige Parameter
final_weight = model.weight()
final_bias = model.bias()
print(f"Gelerntes Gewicht: {final_weight.item():.4f} (wahr: 2.0)")
if final_bias is not None:
    print(f"Gelernter Bias: {final_bias.item():.4f} (wahr: 1.0)")
```

## Fehlerbehebung

### Häufige Probleme und Lösungen

1. **Installationsprobleme**
```bash
# Wenn maturin nicht gefunden wird
pip install --upgrade maturin

# Wenn Rust veraltet ist
rustup update

# Python-Umgebungsprobleme
python -m pip install --upgrade pip
```

2. **Laufzeitfehler**
```python
# Tensorformen überprüfen
print(f"Tensorform: {tensor.shape()}")
print(f"Tensor-Datentyp: {tensor.dtype()}")

# Vorsicht bei Datentypen in der NumPy-Konvertierung
np_array = np.array(data, dtype=np.float32)  # Explizit float32
```

3. **Leistungsoptimierung**
```python
# Im Release-Modus erstellen
# maturin develop --release

# Stapelgröße anpassen
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Größerer Stapel
```

## Nächste Schritte

1. **Erweiterte Beispiele ausprobieren**: Schauen Sie sich die Beispiele in `docs/examples/neural_networks/` an
2. **Keras-ähnliche API verwenden**: `rustorch.training.Model` für einfachere Modellerstellung
3. **Visualisierungsfeatures**: `rustorch.visualization` für Trainingsfortschritt-Visualisierung
4. **Verteiltes Training**: `rustorch.distributed` für Parallelverarbeitung

Detaillierte Dokumentation:
- [Python API-Referenz](../en/python_api_reference.md)
- [Übersichtsdokumentation](../en/python_bindings_overview.md)
- [Beispielsammlung](../examples/)