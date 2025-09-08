# RusTorch Python API Referenz

Vollständige API-Referenz für RusTorch Python-Bindings für deutsche Entwickler.

## Überblick

RusTorch bietet nahtlose Python-Integration durch PyO3-Bindings, die eine PyTorch-ähnliche API bereitstellen und gleichzeitig die Leistung und Sicherheit von Rust nutzen.

## 🧮 Tensor-Operationen

### Tensor-Erstellung

```python
import rustorch as torch

# Aus Python-Liste erstellen
tensor = torch.tensor([1, 2, 3, 4])
print(tensor)  # [1, 2, 3, 4]

# Mit spezifischer Form
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(tensor_2d.shape)  # (2, 2)

# Nullen-Tensor
zeros = torch.zeros((3, 4))

# Einsen-Tensor
ones = torch.ones((2, 3))

# Zufalls-Tensor (Normalverteilung)
randn = torch.randn((2, 2))

# Zufalls-Tensor (Gleichverteilung)
rand = torch.rand((2, 2))

# Bereich
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Linearer Raum
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Tensor-Eigenschaften

```python
tensor = torch.randn((3, 4, 5))

# Form und Größe
print(tensor.shape)      # (3, 4, 5)
print(tensor.size())     # (3, 4, 5)
print(tensor.numel())    # 60 (Gesamtzahl der Elemente)

# Datentyp
print(tensor.dtype)      # torch.float32
tensor_int = tensor.int()
print(tensor_int.dtype)  # torch.int32

# Gerät
print(tensor.device)     # cpu
if torch.cuda.is_available():
    tensor_gpu = tensor.cuda()
    print(tensor_gpu.device)  # cuda:0

# Dimensionalität
print(tensor.dim())      # 3
print(tensor.ndim)       # 3
```

### Grundlegende Operationen

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Arithmetische Operationen
addition = a + b         # [5, 7, 9]
subtraktion = a - b      # [-3, -3, -3]
multiplikation = a * b   # [4, 10, 18]
division = a / b         # [0.25, 0.4, 0.5]

# Element-weise Funktionen
sqrt_a = torch.sqrt(a.float())
exp_a = torch.exp(a.float())
log_a = torch.log(a.float())
sin_a = torch.sin(a.float())

# Potenzierung
power = torch.pow(a, 2)  # [1, 4, 9]
```

### Matrix-Operationen

```python
# Matrix-Multiplikation
A = torch.randn((3, 4))
B = torch.randn((4, 5))
C = torch.matmul(A, B)    # oder A @ B
print(C.shape)            # (3, 5)

# Transposition
A_t = A.t()               # (4, 3)
A_transpose = A.transpose(0, 1)  # Dimensionen tauschen

# Determinante (nur quadratische Matrizen)
square = torch.randn((3, 3))
det = torch.det(square)

# Inverse
try:
    inverse = torch.inverse(square)
    print("Inverse berechnet")
except:
    print("Matrix ist singulär")

# Eigenvalues/Eigenvectors
eigenvals, eigenvecs = torch.eig(square, eigenvectors=True)
```

### Reshaping und Indexierung

```python
tensor = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
reshaped = tensor.view(3, 4)      # (3, 4)
reshaped2 = tensor.reshape(2, 6)  # (2, 6)

# Indexierung
print(reshaped[0, 1])      # Element an Position (0, 1)
print(reshaped[0, :])      # Erste Zeile
print(reshaped[:, 1])      # Zweite Spalte
print(reshaped[0:2, 1:3])  # Slice

# Erweiterte Indexierung
indices = torch.tensor([0, 2])
selected = reshaped[indices]  # Zeilen 0 und 2

# Boolesche Indexierung
mask = reshaped > 5
filtered = reshaped[mask]
```

### Reduktions-Operationen

```python
tensor = torch.randn((3, 4))

# Summe
total_sum = torch.sum(tensor)        # Skalare Summe
sum_dim0 = torch.sum(tensor, dim=0)  # Summe über Dimension 0
sum_dim1 = torch.sum(tensor, dim=1)  # Summe über Dimension 1

# Mittelwert
mean_all = torch.mean(tensor)
mean_dim0 = torch.mean(tensor, dim=0)

# Standardabweichung und Varianz
std = torch.std(tensor)
var = torch.var(tensor)

# Min/Max
min_val = torch.min(tensor)
max_val = torch.max(tensor)
min_indices = torch.argmin(tensor, dim=1)
max_indices = torch.argmax(tensor, dim=1)

# Sortierung
sorted_tensor, indices = torch.sort(tensor, dim=1)
```

## 🤖 Neuronale Netzwerke

### Module und Schichten

```python
import rustorch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Modell erstellen
model = MLP(784, 256, 10)
print(model)

# Forward-Pass
input_data = torch.randn((32, 784))
output = model(input_data)
print(output.shape)  # (32, 10)
```

### Schicht-Typen

```python
# Lineare Schicht
linear = nn.Linear(100, 50)

# Konvolutionale Schichten
conv2d = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
conv1d = nn.Conv1d(1, 32, kernel_size=5)

# Pooling-Schichten
maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool2d = nn.AvgPool2d(kernel_size=2)
adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))

# Normalisierung
batchnorm1d = nn.BatchNorm1d(256)
batchnorm2d = nn.BatchNorm2d(64)
layernorm = nn.LayerNorm(256)

# Dropout
dropout = nn.Dropout(p=0.5)
dropout2d = nn.Dropout2d(p=0.25)

# Aktivierungsfunktionen
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
gelu = nn.GELU()
swish = nn.SiLU()
leaky_relu = nn.LeakyReLU(0.01)
```

### Rekurrente Schichten

```python
# LSTM
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True)
input_seq = torch.randn((32, 10, 100))  # (batch, seq, features)
output, (hidden, cell) = lstm(input_seq)

# GRU
gru = nn.GRU(input_size=100, hidden_size=256, num_layers=1, batch_first=True)
output, hidden = gru(input_seq)

# RNN
rnn = nn.RNN(input_size=100, hidden_size=256, batch_first=True)
output, hidden = rnn(input_seq)
```

## ⚡ Automatische Differenzierung

```python
import rustorch.autograd as autograd

# Gradient-Verfolgung aktivieren
x = torch.randn((2, 2), requires_grad=True)
y = torch.randn((2, 2), requires_grad=True)

# Forward-Pass
z = x * y
loss = z.sum()

# Backward-Pass
loss.backward()

# Gradienten abrufen
print("Gradient von x:", x.grad)
print("Gradient von y:", y.grad)

# Gradienten zurücksetzen
x.grad.zero_()
y.grad.zero_()

# Gradient-freie Bereiche
with torch.no_grad():
    # Operationen hier verfolgen keine Gradienten
    result = x * y + 5
    
# Temporäre Gradient-Deaktivierung
x.requires_grad_(False)
result = x * 2  # Kein Gradient
x.requires_grad_(True)
```

### Benutzerdefinierte Funktionen

```python
class SquareFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

# Verwendung
square = SquareFunction.apply
x = torch.randn((2, 2), requires_grad=True)
y = square(x)
loss = y.sum()
loss.backward()
print(x.grad)
```

## 🎯 Optimierung

```python
import rustorch.optim as optim

# Optimierer erstellen
model = MLP(784, 256, 10)

# SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# AdaGrad
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.01)

# RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Training-Schleife
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer_adam.zero_grad()  # Gradienten zurücksetzen
        output = model(data)        # Forward-Pass
        loss = criterion(output, target)  # Loss berechnen
        loss.backward()             # Backward-Pass
        optimizer_adam.step()       # Parameter aktualisieren
        
        if batch_idx % 100 == 0:
            print(f'Epoche: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
```

### Learning Rate Scheduler

```python
import rustorch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step LR
scheduler_step = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential LR
scheduler_exp = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine Annealing
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# In der Training-Schleife
for epoch in range(100):
    # Training...
    train_one_epoch(model, optimizer, train_loader)
    
    # Learning Rate aktualisieren
    scheduler_step.step()
    print(f'Epoch {epoch}: LR = {scheduler_step.get_lr()[0]:.6f}')
```

## 📊 Verlustfunktionen

```python
import rustorch.nn.functional as F

# Klassifikation
predictions = torch.randn((32, 10))  # 32 Samples, 10 Klassen
targets = torch.randint(0, 10, (32,))  # Klassen-Indices

# Cross Entropy Loss
ce_loss = F.cross_entropy(predictions, targets)

# Negative Log Likelihood
log_probs = F.log_softmax(predictions, dim=1)
nll_loss = F.nll_loss(log_probs, targets)

# Regression
predictions_reg = torch.randn((32, 1))
targets_reg = torch.randn((32, 1))

# Mean Squared Error
mse_loss = F.mse_loss(predictions_reg, targets_reg)

# Mean Absolute Error
mae_loss = F.l1_loss(predictions_reg, targets_reg)

# Smooth L1 Loss
smooth_l1_loss = F.smooth_l1_loss(predictions_reg, targets_reg)

# Binäre Klassifikation
binary_predictions = torch.randn((32, 1))
binary_targets = torch.randint(0, 2, (32, 1)).float()

# Binary Cross Entropy
bce_loss = F.binary_cross_entropy_with_logits(binary_predictions, binary_targets)
```

## 🖼️ Computer Vision

```python
import rustorch.vision.transforms as transforms

# Transformationen definieren
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Vorgefertigte Modelle
import rustorch.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# VGG
vgg16 = models.vgg16(pretrained=True)

# DenseNet
densenet121 = models.densenet121(pretrained=True)

# MobileNet
mobilenet_v2 = models.mobilenet_v2(pretrained=True)

# Transfer Learning
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Für neue Anzahl von Klassen
```

## 💾 Serialisierung und Checkpoints

```python
# Modell speichern
torch.save(model.state_dict(), 'model_weights.pth')
torch.save(model, 'entire_model.pth')

# Checkpoint mit mehr Informationen
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'checkpoint.pth')

# Modell laden
model.load_state_dict(torch.load('model_weights.pth'))
model = torch.load('entire_model.pth')

# Checkpoint laden
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Für Inferenz vorbereiten
model.eval()
```

## 🔧 Utilities und Hilfsfunktionen

```python
# Gerät-Management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# Speicher-Information
if torch.cuda.is_available():
    print(f'GPU Speicher belegt: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
    print(f'GPU Speicher gecached: {torch.cuda.memory_cached() / 1024**2:.2f} MB')
    
    # Speicher freigeben
    torch.cuda.empty_cache()

# Zufalls-Seed setzen
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Modell-Informationen
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Anzahl trainierbare Parameter: {count_parameters(model):,}')

# Modell-Summary
def model_summary(model, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f'{class_name}-{module_idx+1}'
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1
                
            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
                
            summary[m_key]['nb_params'] = params
            
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Erstelle Summary
    summary = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    
    # Dummy Forward-Pass
    x = torch.randn(1, *input_size)
    model(x)
    
    # Hooks entfernen
    for h in hooks:
        h.remove()
    
    return summary

# Verwendung
summary = model_summary(model, (3, 224, 224))
print(summary)
```

Diese umfassende API-Referenz deckt die wichtigsten Funktionen der RusTorch Python-Bindings ab und bietet deutschen Entwicklern eine vertraute PyTorch-ähnliche Schnittstelle mit der Leistung von Rust.