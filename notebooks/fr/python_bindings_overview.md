# Aperçu des Bindings Python RusTorch

## Aperçu

RusTorch est un framework d'apprentissage profond haute performance implémenté en Rust, fournissant des APIs similaires à PyTorch tout en exploitant les avantages de sécurité et de performance de Rust. Grâce aux bindings Python, vous pouvez accéder aux fonctionnalités RusTorch directement depuis Python.

## Caractéristiques Principales

### 🚀 **Haute Performance**
- **Cœur Rust** : Atteint des performances de niveau C++ tout en garantissant la sécurité mémoire
- **Support SIMD** : Vectorisation automatique pour des calculs numériques optimisés
- **Traitement Parallèle** : Calcul parallèle efficace utilisant rayon
- **Copie Zéro** : Copie minimale de données entre NumPy et RusTorch

### 🛡️ **Sécurité**
- **Sécurité Mémoire** : Évite les fuites mémoire et les conditions de course grâce au système de propriété de Rust
- **Sécurité de Type** : Vérification de type au moment de la compilation réduit les erreurs d'exécution
- **Gestion d'Erreur** : Gestion complète d'erreurs avec conversion automatique vers les exceptions Python

### 🎯 **Facilité d'Utilisation**
- **API Compatible PyTorch** : Migration facile depuis le code PyTorch existant
- **API Haut Niveau style Keras** : Interfaces intuitives comme model.fit()
- **Intégration NumPy** : Conversion bidirectionnelle avec les tableaux NumPy

## Architecture

Les bindings Python de RusTorch consistent en 10 modules :

### 1. **tensor** - Opérations de Tenseur
```python
import rustorch

# Création de tenseurs
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# Intégration NumPy
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - Différenciation Automatique
```python
# Calcul de gradients
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # Obtenir les gradients
```

### 3. **nn** - Réseaux de Neurones
```python
# Création de couches
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# Fonctions de perte
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - Optimiseurs
```python
# Optimiseurs
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Planificateurs de taux d'apprentissage
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - Chargement de Données
```python
# Création de jeu de données
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Transformations de données
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - API d'Entraînement Haut Niveau
```python
# API style Keras
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Exécution d'entraînement
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - Entraînement Distribué
```python
# Configuration d'entraînement distribué
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# Parallélisme de données
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - Visualisation
```python
# Tracer l'historique d'entraînement
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# Visualisation de tenseur
plotter.plot_tensor_as_image(tensor, title="Carte de Caractéristiques")
```

### 9. **utils** - Utilitaires
```python
# Sauvegarder/charger modèle
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# Profilage
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## Installation

### Prérequis
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (pour usage GPU)

### Construire et Installer
```bash
# Cloner le dépôt
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Créer un environnement virtuel Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Installer les dépendances
pip install maturin numpy

# Construire et installer
maturin develop --release

# Ou installer depuis PyPI (prévu pour l'avenir)
# pip install rustorch
```

## Démarrage Rapide

### 1. Opérations de Tenseur de Base
```python
import rustorch
import numpy as np

# Création de tenseur
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Forme: {x.shape()}")  # Forme: [2, 2]

# Opérations mathématiques
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Résultat: {z.to_numpy()}")
```

### 2. Exemple de Régression Linéaire
```python
import rustorch
import numpy as np

# Générer des données
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convertir en tenseurs
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Définir le modèle
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# Créer le jeu de données
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Entraîner
history = model.fit(dataloader, epochs=100, verbose=True)

# Afficher les résultats
print(f"Perte finale: {history.train_loss()[-1]:.4f}")
```

### 3. Classification avec Réseau de Neurones
```python
import rustorch

# Préparer les données
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Construire le modèle
model = rustorch.Model("RéseauClassification")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# Compiler le modèle
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Configuration d'entraînement
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# Entraîner
history = trainer.train(model, train_loader, val_loader)

# Évaluer
metrics = model.evaluate(test_loader)
print(f"Précision de test: {metrics['accuracy']:.4f}")
```

## Optimisation de Performance

### Utilisation SIMD
```python
# Activer l'optimisation SIMD durant la construction
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # Calcul optimisé SIMD
```

### Usage GPU
```python
# Usage CUDA (prévu pour l'avenir)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # Calcul GPU
```

### Chargement de Données Parallèle
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # Nombre de workers parallèles
)
```

## Meilleures Pratiques

### 1. Efficacité Mémoire
```python
# Utiliser conversion copie zéro
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # Pas de copie

# Utiliser opérations en place
tensor.add_(1.0)  # Efficace en mémoire
```

### 2. Gestion d'Erreurs
```python
try:
    result = model(entrée_invalide)
except rustorch.RusTorchError as e:
    print(f"Erreur RusTorch: {e}")
except Exception as e:
    print(f"Erreur inattendue: {e}")
```

### 3. Débogage et Profilage
```python
# Utiliser le profileur
profiler = rustorch.utils.Profiler()
profiler.start()

# Exécuter le calcul
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## Limitations

### Limitations Actuelles
- **Support GPU** : Support CUDA/ROCm en développement
- **Graphes Dynamiques** : Supporte actuellement seulement les graphes statiques
- **Entraînement Distribué** : Seule la fonctionnalité de base est implémentée

### Extensions Futures
- Accélération GPU (CUDA, Metal, ROCm)
- Support pour graphes de calcul dynamiques
- Plus de couches de réseaux de neurones
- Quantisation et élagage de modèles
- Fonctionnalité d'export ONNX

## Contribution

### Participation au Développement
```bash
# Configurer l'environnement de développement
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# Exécuter les tests
cargo test
python -m pytest tests/

# Vérifications de qualité de code
cargo clippy
cargo fmt
```

### Communauté
- GitHub Issues : Rapports de bugs et demandes de fonctionnalités
- Discussions : Questions et discussions
- Discord : Support en temps réel

## Licence

RusTorch est publié sous la Licence MIT. Libre d'utilisation pour des fins commerciales et non-commerciales.

## Liens Connexes

- [Dépôt GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Documentation API](./python_api_reference.md)
- [Exemples et Tutoriels](../examples/)
- [Benchmarks de Performance](./benchmarks.md)