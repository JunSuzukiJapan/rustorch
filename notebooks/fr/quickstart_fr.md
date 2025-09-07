# Guide de Démarrage Rapide RusTorch

## Installation

### 1. Prérequis
```bash
# Rust 1.70 ou supérieur
rustc --version

# Python 3.8 ou supérieur
python --version

# Installer les dépendances requises
pip install maturin numpy matplotlib
```

### 2. Compiler et Installer RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Créer un environnement virtuel Python (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Compiler et installer en mode développement
maturin develop --release
```

## Exemples d'Usage de Base

### 1. Création de Tenseurs et Opérations de Base

```python
import rustorch
import numpy as np

# Création de tenseurs
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tenseur x:\n{x}")
print(f"Forme: {x.shape()}")  # [2, 2]

# Matrices de zéros et d'identité
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Zéros:\n{zeros}")
print(f"Uns:\n{ones}")
print(f"Identité:\n{identity}")

# Tenseurs aléatoires
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Aléatoire normal:\n{random_normal}")
print(f"Aléatoire uniforme:\n{random_uniform}")

# Intégration NumPy
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"Depuis NumPy:\n{tensor_from_numpy}")

# Convertir vers NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"Retour vers NumPy:\n{back_to_numpy}")
```

### 2. Opérations Arithmétiques

```python
# Opérations arithmétiques de base
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Opérations élément par élément
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (élément par élément)
div_result = a.div(b)  # a / b (élément par élément)

print(f"Addition:\n{add_result}")
print(f"Soustraction:\n{sub_result}")
print(f"Multiplication:\n{mul_result}")
print(f"Division:\n{div_result}")

# Opérations scalaires
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Addition scalaire (+2):\n{scalar_add}")
print(f"Multiplication scalaire (*3):\n{scalar_mul}")

# Multiplication matricielle
matmul_result = a.matmul(b)
print(f"Multiplication matricielle:\n{matmul_result}")

# Fonctions mathématiques
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Racine carrée:\n{sqrt_result}")
print(f"Exponentielle:\n{exp_result}")
print(f"Logarithme naturel:\n{log_result}")
```

### 3. Manipulation de Forme de Tenseurs

```python
# Exemples de manipulation de forme
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Forme originale: {original.shape()}")  # [2, 4]

# Redimensionner
reshaped = original.reshape([4, 2])
print(f"Redimensionné [4, 2]:\n{reshaped}")

# Transposer
transposed = original.transpose(0, 1)
print(f"Transposé:\n{transposed}")

# Ajout/suppression de dimensions
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Avant compression: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"Après compression: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"Après expansion: {expanded.shape()}")  # [1, 3]
```

### 4. Opérations Statistiques

```python
# Fonctions statistiques
data = rustorch.randn([3, 4])
print(f"Données:\n{data}")

# Statistiques de base
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Moyenne: {mean_val.item():.4f}")
print(f"Somme: {sum_val.item():.4f}")
print(f"Écart-type: {std_val.item():.4f}")
print(f"Variance: {var_val.item():.4f}")
print(f"Maximum: {max_val.item():.4f}")
print(f"Minimum: {min_val.item():.4f}")

# Statistiques par dimension spécifique
row_mean = data.mean(dim=1)  # Moyenne de chaque ligne
col_sum = data.sum(dim=0)    # Somme de chaque colonne

print(f"Moyennes des lignes: {row_mean}")
print(f"Sommes des colonnes: {col_sum}")
```

## Bases de la Différentiation Automatique

### 1. Calcul de Gradient

```python
# Exemple de différentiation automatique
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Tenseur d'entrée: {x}")

# Créer Variable
var_x = rustorch.autograd.Variable(x)

# Construire le graphe de calcul
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Sortie: {y.data().item()}")

# Rétropropagation
y.backward()

# Obtenir le gradient
grad = var_x.grad()
print(f"Gradient: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Graphes de Calcul Complexes

```python
# Exemple plus complexe
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Fonction complexe: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), puis somme

print(f"Résultat: {z.data().item():.4f}")

# Rétropropagation
z.backward()
grad = var_x.grad()
print(f"Gradient: {grad}")
```

## Bases des Réseaux de Neurones

### 1. Couche Linéaire Simple

```python
# Créer une couche linéaire
linear_layer = rustorch.nn.Linear(3, 1)  # 3 entrées -> 1 sortie

# Entrée aléatoire
input_data = rustorch.randn([2, 3])  # Taille de batch 2, 3 caractéristiques
print(f"Entrée: {input_data}")

# Passage avant
output = linear_layer.forward(input_data)
print(f"Sortie: {output}")

# Vérifier les paramètres
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Forme du poids: {weight.shape()}")
print(f"Poids: {weight}")
if bias is not None:
    print(f"Biais: {bias}")
```

### 2. Fonctions d'Activation

```python
# Diverses fonctions d'activation
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

### 3. Fonctions de Perte

```python
# Exemples d'usage des fonctions de perte
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Erreur quadratique moyenne
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"Perte MSE: {loss_value.item():.6f}")

# Entropie croisée (pour classification)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Indices de classe

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Perte d'Entropie Croisée: {ce_loss_value.item():.6f}")
```

## Traitement des Données

### 1. Jeux de Données et Chargeurs de Données

```python
# Créer un jeu de données
import numpy as np

# Générer des données d'exemple
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 échantillons, 4 caractéristiques
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # Classification à 3 classes

# Convertir en tenseurs
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Créer le jeu de données
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Taille du jeu de données: {len(dataset)}")

# Créer le chargeur de données
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Obtenir des lots du chargeur de données
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Montrer seulement les 3 premiers lots
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Lot {batch_idx}: Forme d'entrée {inputs.shape()}, Forme de cible {targets.shape()}")
```

### 2. Transformations de Données

```python
# Exemples de transformation de données
data = rustorch.randn([10, 10])
print(f"Moyenne des données originales: {data.mean().item():.4f}")
print(f"Écart-type des données originales: {data.std().item():.4f}")

# Transformation de normalisation
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Moyenne des données normalisées: {normalized_data.mean().item():.4f}")
print(f"Écart-type des données normalisées: {normalized_data.std().item():.4f}")
```

## Exemple d'Entraînement Complet

### Régression Linéaire

```python
# Exemple complet de régression linéaire
import numpy as np

# Générer des données
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Convertir en tenseurs
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Créer le jeu de données et le chargeur
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Définir le modèle
model = rustorch.nn.Linear(1, 1)  # 1 entrée -> 1 sortie

# Fonction de perte et optimiseur
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Boucle d'entraînement
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
            
            # Remettre à zéro les gradients
            optimizer.zero_grad()
            
            # Passage avant
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Rétropropagation (simplifiée)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Époque {epoch}: Perte = {avg_loss:.6f}")

print("Entraînement terminé!")

# Paramètres finaux
final_weight = model.weight()
final_bias = model.bias()
print(f"Poids appris: {final_weight.item():.4f} (vrai: 2.0)")
if final_bias is not None:
    print(f"Biais appris: {final_bias.item():.4f} (vrai: 1.0)")
```

## Dépannage

### Problèmes Courants et Solutions

1. **Problèmes d'Installation**
```bash
# Si maturin n'est pas trouvé
pip install --upgrade maturin

# Si Rust est obsolète
rustup update

# Problèmes d'environnement Python
python -m pip install --upgrade pip
```

2. **Erreurs d'Exécution**
```python
# Vérifier les formes de tenseur
print(f"Forme du tenseur: {tensor.shape()}")
print(f"Type de données du tenseur: {tensor.dtype()}")

# Attention aux types de données dans la conversion NumPy
np_array = np.array(data, dtype=np.float32)  # float32 explicite
```

3. **Optimisation des Performances**
```python
# Compiler en mode release
# maturin develop --release

# Ajuster la taille du lot
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Lot plus grand
```

## Prochaines Étapes

1. **Essayer des Exemples Avancés**: Consultez les exemples dans `docs/examples/neural_networks/`
2. **Utiliser l'API style Keras**: `rustorch.training.Model` pour une construction de modèle plus facile
3. **Fonctionnalités de Visualisation**: `rustorch.visualization` pour visualiser le progrès d'entraînement
4. **Entraînement Distribué**: `rustorch.distributed` pour le traitement parallèle

Documentation Détaillée:
- [Référence API Python](../fr/python_api_reference.md)
- [Documentation Vue d'Ensemble](../fr/python_bindings_overview.md)
- [Collection d'Exemples](../examples/)