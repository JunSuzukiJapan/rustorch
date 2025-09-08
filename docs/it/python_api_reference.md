# Riferimento API Python per RusTorch

Riferimento completo dell'API Python di RusTorch per sviluppatori di machine learning e deep learning.

## Indice

- [Modulo Tensor](#modulo-tensor)
- [Differenziazione Automatica](#differenziazione-automatica)
- [Reti Neurali](#reti-neurali)
- [Ottimizzazione](#ottimizzazione)
- [Computer Vision](#computer-vision)
- [GPU e Dispositivi](#gpu-e-dispositivi)
- [Utilità](#utilità)

## Modulo Tensor

### `Tensor`

Struttura fondamentale per operazioni di tensor N-dimensionale.

#### Costruttori

```python
import rustorch_py as torch

# Creare tensor da dati
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# Tensor di zeri
zeros = torch.zeros([2, 3], dtype=torch.float32)

# Tensor di uni
ones = torch.ones([2, 3], dtype=torch.float32)

# Tensor casuali (distribuzione normale)
randn = torch.randn([2, 3], dtype=torch.float32)

# Tensor casuali (distribuzione uniforme)
rand = torch.rand([2, 3], dtype=torch.float32)
```

#### Operazioni di Base

```python
# Operazioni aritmetiche
result = tensor1.add(tensor2)
result = tensor1.sub(tensor2)
result = tensor1.mul(tensor2)
result = tensor1.div(tensor2)

# Moltiplicazione di matrici
result = tensor1.matmul(tensor2)

# Trasposizione
transposed = tensor.t()

# Cambio di forma
reshaped = tensor.reshape([6, 1])
```

#### Operazioni di Riduzione

```python
# Somma
sum_all = tensor.sum()
sum_dim = tensor.sum(dim=0, keepdim=False)

# Media
mean_all = tensor.mean()
mean_dim = tensor.mean(dim=0, keepdim=False)

# Massimo e minimo
max_val, max_indices = tensor.max(dim=0)
min_val, min_indices = tensor.min(dim=0)
```

#### Indicizzazione e Selezione

```python
# Selezione per indici
slice_result = tensor.slice(dim=0, start=0, end=2, step=1)

# Selezione per condizione
mask = tensor.gt(threshold)
selected = tensor.masked_select(mask)
```

## Differenziazione Automatica

### `Variable`

Wrapper per tensor che consente la differenziazione automatica.

```python
import rustorch_py.autograd as autograd

# Creare variabile con requires_grad=True
x = autograd.Variable(torch.randn([2, 2]), requires_grad=True)
y = autograd.Variable(torch.randn([2, 2]), requires_grad=True)

# Operazioni che costruiscono il grafo computazionale
z = x.matmul(y)
loss = z.sum()

# Backpropagation
loss.backward()

# Accedere ai gradienti
x_grad = x.grad
print(f"Gradiente di x: {x_grad}")
```

### Funzioni di Differenziazione

```python
# Funzione personalizzata con gradiente
def custom_function(input_var):
    # Passaggio in avanti
    output = input_var.pow(2.0)
    
    # Il gradiente sarà calcolato automaticamente
    return output

# Contesto senza calcolo dei gradienti
with torch.no_grad():
    result = model.forward(input_data)
```

## Reti Neurali

### Layer di Base

#### `Linear`

Trasformazione lineare (layer completamente connesso).

```python
import rustorch_py.nn as nn

linear = nn.Linear(784, 256)  # input: 784, output: 256
input_tensor = torch.randn([32, 784])
output = linear.forward(input_tensor)
```

#### Funzioni di Attivazione

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

### Layer Convoluzionali

```python
# Convoluzione 2D
conv2d = nn.Conv2d(
    in_channels=3,     # canali di input
    out_channels=64,   # canali di output
    kernel_size=3,     # dimensione del kernel
    stride=1,          # passo
    padding=1          # padding
)

input_tensor = torch.randn([1, 3, 224, 224])
output = conv2d.forward(input_tensor)
```

### Modelli Sequenziali

```python
model = nn.Sequential([
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
])

# Passaggio in avanti
input_data = torch.randn([32, 784])
output = model.forward(input_data)
```

## Ottimizzazione

### Ottimizzatori

#### `Adam`

```python
import rustorch_py.optim as optim

optimizer = optim.Adam(
    params=model.parameters(),  # parametri del modello
    lr=0.001,                   # learning rate
    betas=(0.9, 0.999),        # coefficienti beta
    eps=1e-8                   # epsilon
)

# Loop di training
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
    lr=0.01,                   # learning rate
    momentum=0.9               # momentum
)
```

### Funzioni di Loss

```python
import rustorch_py.nn.functional as F

# Errore quadratico medio
mse_loss = F.mse_loss(prediction, target)

# Cross entropy
ce_loss = F.cross_entropy(prediction, target)

# Binary cross entropy
bce_loss = F.binary_cross_entropy(prediction, target)
```

## Computer Vision

### Trasformazioni di Immagine

```python
import rustorch_py.vision.transforms as transforms

# Ridimensionare immagine
resize = transforms.Resize((224, 224))
resized = resize.forward(image)

# Normalizzazione
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # media
    std=[0.229, 0.224, 0.225]    # deviazione standard
)
normalized = normalize.forward(image)

# Trasformazioni casuali
random_crop = transforms.RandomCrop(32, padding=4)
cropped = random_crop.forward(image)
```

### Modelli Pre-addestrati

```python
import rustorch_py.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
output = resnet18.forward(input_tensor)

# VGG
vgg16 = models.vgg16(pretrained=True)
features = vgg16.features(input_tensor)
```

## GPU e Dispositivi

### Gestione Dispositivi

```python
import rustorch_py as torch

# CPU
cpu = torch.device('cpu')

# CUDA
cuda = torch.device('cuda:0')  # GPU 0
cuda_available = torch.cuda.is_available()

# Metal (macOS)
metal = torch.device('metal:0')

# Spostare tensor su dispositivo
tensor_gpu = tensor.to(cuda)
```

### Operazioni Multi-GPU

```python
import rustorch_py.distributed as dist

# Inizializzazione elaborazione distribuita
dist.init_process_group("nccl", rank=0, world_size=2)

# AllReduce per sincronizzazione gradienti
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## Utilità

### Serializzazione

```python
import rustorch_py.serialize as serialize

# Salvare modello
serialize.save(model, "model.pth")

# Caricare modello
loaded_model = serialize.load("model.pth")
```

### Metriche

```python
import rustorch_py.metrics as metrics

# Accuratezza
accuracy = metrics.accuracy(predictions, targets)

# F1-Score
f1 = metrics.f1_score(predictions, targets, average="macro")

# Matrice di confusione
confusion_matrix = metrics.confusion_matrix(predictions, targets)
```

### Utilità per Dati

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

## Esempi Completi

### Classificazione con CNN

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

# Creare modello
model = CNN()

# Ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funzione di loss
criterion = nn.CrossEntropyLoss()

# Loop di training
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Passaggio in avanti
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Passaggio all'indietro
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoca {epoch}: Loss = {loss.item():.4f}")
```

Questo riferimento copre le funzionalità principali dell'API Python di RusTorch. Per esempi più dettagliati e casi d'uso avanzati, consultare la [Guida Completa ai Bindings Python](python_bindings_overview.md) e la [Guida Jupyter](jupyter-guide.md).