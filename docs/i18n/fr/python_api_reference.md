# Référence API Python pour RusTorch

Référence complète de l'API Python de RusTorch pour les développeurs en apprentissage automatique et apprentissage profond.

## Table des Matières

- [Module Tensor](#module-tensor)
- [Différenciation Automatique](#différenciation-automatique)
- [Réseaux de Neurones](#réseaux-de-neurones)
- [Optimisation](#optimisation)
- [Vision par Ordinateur](#vision-par-ordinateur)
- [GPU et Dispositifs](#gpu-et-dispositifs)
- [Utilitaires](#utilitaires)

## Module Tensor

### `Tensor`

Structure fondamentale pour les opérations de tenseur N-dimensionnel.

#### Constructeurs

```python
import rustorch_py as torch

# Créer un tenseur à partir de données
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# Tenseur de zéros
zeros = torch.zeros([2, 3], dtype=torch.float32)

# Tenseur d'uns
ones = torch.ones([2, 3], dtype=torch.float32)

# Tenseurs aléatoires (distribution normale)
randn = torch.randn([2, 3], dtype=torch.float32)

# Tenseurs aléatoires (distribution uniforme)
rand = torch.rand([2, 3], dtype=torch.float32)
```

#### Opérations de Base

```python
# Opérations arithmétiques
result = tensor1.add(tensor2)
result = tensor1.sub(tensor2)
result = tensor1.mul(tensor2)
result = tensor1.div(tensor2)

# Multiplication matricielle
result = tensor1.matmul(tensor2)

# Transposition
transposed = tensor.t()

# Changement de forme
reshaped = tensor.reshape([6, 1])
```

#### Opérations de Réduction

```python
# Somme
sum_all = tensor.sum()
sum_dim = tensor.sum(dim=0, keepdim=False)

# Moyenne
mean_all = tensor.mean()
mean_dim = tensor.mean(dim=0, keepdim=False)

# Maximum et minimum
max_val, max_indices = tensor.max(dim=0)
min_val, min_indices = tensor.min(dim=0)
```

#### Indexation et Sélection

```python
# Sélection par indices
slice_result = tensor.slice(dim=0, start=0, end=2, step=1)

# Sélection par condition
mask = tensor.gt(threshold)
selected = tensor.masked_select(mask)
```

## Différenciation Automatique

### `Variable`

Wrapper pour tenseurs permettant la différenciation automatique.

```python
import rustorch_py.autograd as autograd

# Créer une variable avec requires_grad=True
x = autograd.Variable(torch.randn([2, 2]), requires_grad=True)
y = autograd.Variable(torch.randn([2, 2]), requires_grad=True)

# Opérations qui construisent le graphe de calcul
z = x.matmul(y)
loss = z.sum()

# Rétropropagation
loss.backward()

# Accéder aux gradients
x_grad = x.grad
print(f"Gradient de x: {x_grad}")
```

### Fonctions de Différenciation

```python
# Fonction personnalisée avec gradient
def custom_function(input_var):
    # Passe avant
    output = input_var.pow(2.0)
    
    # Le gradient sera calculé automatiquement
    return output

# Contexte sans calcul de gradients
with torch.no_grad():
    result = model.forward(input_data)
```

## Réseaux de Neurones

### Couches de Base

#### `Linear`

Transformation linéaire (couche entièrement connectée).

```python
import rustorch_py.nn as nn

linear = nn.Linear(784, 256)  # entrée: 784, sortie: 256
input_tensor = torch.randn([32, 784])
output = linear.forward(input_tensor)
```

#### Fonctions d'Activation

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

### Couches Convolutionnelles

```python
# Convolution 2D
conv2d = nn.Conv2d(
    in_channels=3,     # canaux d'entrée
    out_channels=64,   # canaux de sortie
    kernel_size=3,     # taille du noyau
    stride=1,          # pas
    padding=1          # rembourrage
)

input_tensor = torch.randn([1, 3, 224, 224])
output = conv2d.forward(input_tensor)
```

### Modèles Séquentiels

```python
model = nn.Sequential([
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
])

# Passe avant
input_data = torch.randn([32, 784])
output = model.forward(input_data)
```

## Optimisation

### Optimiseurs

#### `Adam`

```python
import rustorch_py.optim as optim

optimizer = optim.Adam(
    params=model.parameters(),  # paramètres du modèle
    lr=0.001,                   # taux d'apprentissage
    betas=(0.9, 0.999),        # coefficients beta
    eps=1e-8                   # epsilon
)

# Boucle d'entraînement
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
    lr=0.01,                   # taux d'apprentissage
    momentum=0.9               # momentum
)
```

### Fonctions de Perte

```python
import rustorch_py.nn.functional as F

# Erreur quadratique moyenne
mse_loss = F.mse_loss(prediction, target)

# Entropie croisée
ce_loss = F.cross_entropy(prediction, target)

# Entropie croisée binaire
bce_loss = F.binary_cross_entropy(prediction, target)
```

## Vision par Ordinateur

### Transformations d'Image

```python
import rustorch_py.vision.transforms as transforms

# Redimensionner l'image
resize = transforms.Resize((224, 224))
resized = resize.forward(image)

# Normalisation
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # moyenne
    std=[0.229, 0.224, 0.225]    # écart-type
)
normalized = normalize.forward(image)

# Transformations aléatoires
random_crop = transforms.RandomCrop(32, padding=4)
cropped = random_crop.forward(image)
```

### Modèles Pré-entraînés

```python
import rustorch_py.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
output = resnet18.forward(input_tensor)

# VGG
vgg16 = models.vgg16(pretrained=True)
features = vgg16.features(input_tensor)
```

## GPU et Dispositifs

### Gestion des Dispositifs

```python
import rustorch_py as torch

# CPU
cpu = torch.device('cpu')

# CUDA
cuda = torch.device('cuda:0')  # GPU 0
cuda_available = torch.cuda.is_available()

# Metal (macOS)
metal = torch.device('metal:0')

# Déplacer le tenseur vers le dispositif
tensor_gpu = tensor.to(cuda)
```

### Opérations Multi-GPU

```python
import rustorch_py.distributed as dist

# Initialisation du traitement distribué
dist.init_process_group("nccl", rank=0, world_size=2)

# AllReduce pour la synchronisation des gradients
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## Utilitaires

### Sérialisation

```python
import rustorch_py.serialize as serialize

# Sauvegarder le modèle
serialize.save(model, "model.pth")

# Charger le modèle
loaded_model = serialize.load("model.pth")
```

### Métriques

```python
import rustorch_py.metrics as metrics

# Précision
accuracy = metrics.accuracy(predictions, targets)

# Score F1
f1 = metrics.f1_score(predictions, targets, average="macro")

# Matrice de confusion
confusion_matrix = metrics.confusion_matrix(predictions, targets)
```

### Utilitaires de Données

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

## Exemples Complets

### Classification avec CNN

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

# Créer le modèle
model = CNN()

# Optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fonction de perte
criterion = nn.CrossEntropyLoss()

# Boucle d'entraînement
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Passe avant
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Passe arrière
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Époque {epoch}: Perte = {loss.item():.4f}")
```

Cette référence couvre les fonctionnalités principales de l'API Python de RusTorch. Pour des exemples plus détaillés et des cas d'usage avancés, consultez le [Guide Complet des Liaisons Python](python_bindings_overview.md) et le [Guide Jupyter](jupyter-guide.md).