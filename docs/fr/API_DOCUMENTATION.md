# Documentation API RusTorch

## 📚 Référence API Complète

Ce document fournit une documentation API complète pour RusTorch v0.5.15, organisée par module et fonctionnalité. Il comprend une gestion d'erreurs unifiée avec `RusTorchError` et `RusTorchResult<T>` pour une gestion cohérente des erreurs sur plus de 1060 tests. **Phase 8 TERMINÉE** ajoute des utilitaires de tenseurs avancés incluant des opérations conditionnelles, l'indexation et les fonctions statistiques. **Phase 9 TERMINÉE** introduit un système de sérialisation complet avec sauvegarde/chargement de modèles, compilation JIT et support de formats multiples incluant la compatibilité PyTorch.

## 🏗️ Architecture de Base

### Structure des Modules

```
rustorch/
├── tensor/              # Opérations de tenseurs de base et structures de données
├── nn/                  # Couches et fonctions de réseaux de neurones
├── autograd/            # Moteur de différentiation automatique
├── optim/               # Optimiseurs et planificateurs de taux d'apprentissage
├── special/             # Fonctions mathématiques spéciales
├── distributions/       # Distributions statistiques
├── vision/              # Transformations de vision par ordinateur
├── linalg/              # Opérations d'algèbre linéaire (BLAS/LAPACK)
├── gpu/                 # Accélération GPU (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # Opérations de tenseurs creux et élagage (Phase 12)
├── serialization/       # Sérialisation de modèles et compilation JIT (Phase 9)
└── wasm/                # Liaisons WebAssembly (voir [Documentation API WASM](WASM_API_DOCUMENTATION.md))
```

## 📊 Module Tensor

### Création de Tenseurs de Base

```rust
use rustorch::tensor::Tensor;

// Création de base
let tensor = Tensor::new(vec![2, 3]);               // Création basée sur la forme
let tensor = Tensor::from_vec(data, vec![2, 3]);    // À partir d'un vecteur de données
let tensor = Tensor::zeros(vec![10, 10]);           // Tenseur rempli de zéros
let tensor = Tensor::ones(vec![5, 5]);              // Tenseur rempli de uns
let tensor = Tensor::randn(vec![3, 3]);             // Distribution normale aléatoire
let tensor = Tensor::rand(vec![3, 3]);              // Distribution uniforme aléatoire [0,1)
let tensor = Tensor::eye(5);                        // Matrice identité
let tensor = Tensor::full(vec![2, 2], 3.14);       // Remplit avec une valeur spécifique
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // Tenseur de plage
let tensor = Tensor::linspace(0.0, 1.0, 100);      // Espacement linéaire
```

### Opérations de Tenseurs

```rust
// Opérations arithmétiques
let result = a.add(&b);                             // Addition élément par élément
let result = a.sub(&b);                             // Soustraction élément par élément
let result = a.mul(&b);                             // Multiplication élément par élément
let result = a.div(&b);                             // Division élément par élément
let result = a.pow(&b);                             // Puissance élément par élément
let result = a.rem(&b);                             // Reste élément par élément

// Opérations matricielles
let result = a.matmul(&b);                          // Multiplication matricielle
let result = a.transpose();                         // Transposition matricielle
let result = a.dot(&b);                             // Produit scalaire

// Fonctions mathématiques
let result = tensor.exp();                          // Exponentielle
let result = tensor.ln();                           // Logarithme naturel
let result = tensor.log10();                        // Logarithme base 10
let result = tensor.sqrt();                         // Racine carrée
let result = tensor.abs();                          // Valeur absolue
let result = tensor.sin();                          // Fonction sinus
let result = tensor.cos();                          // Fonction cosinus
let result = tensor.tan();                          // Fonction tangente
let result = tensor.asin();                         // Arcsinus
let result = tensor.acos();                         // Arccosinus
let result = tensor.atan();                         // Arctangente
let result = tensor.sinh();                         // Sinus hyperbolique
let result = tensor.cosh();                         // Cosinus hyperbolique
let result = tensor.tanh();                         // Tangente hyperbolique
let result = tensor.floor();                        // Fonction plancher
let result = tensor.ceil();                         // Fonction plafond
let result = tensor.round();                        // Fonction arrondir
let result = tensor.sign();                         // Fonction signe
let result = tensor.max();                          // Valeur maximale
let result = tensor.min();                          // Valeur minimale
let result = tensor.sum();                          // Somme de tous les éléments
let result = tensor.mean();                         // Valeur moyenne
let result = tensor.std();                          // Écart-type
let result = tensor.var();                          // Variance

// Manipulation de forme
let result = tensor.reshape(vec![6, 4]);            // Redimensionner le tenseur
let result = tensor.squeeze();                      // Supprimer les dimensions de taille 1
let result = tensor.unsqueeze(1);                   // Ajouter une dimension à l'index
let result = tensor.permute(vec![1, 0, 2]);         // Permuter les dimensions
let result = tensor.expand(vec![10, 10, 5]);        // Étendre les dimensions du tenseur
```

## 🧠 Module Neural Network (nn)

### Couches de Base

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// Couche linéaire
let linear = Linear::new(784, 256)?;                // entrée 784, sortie 256
let output = linear.forward(&input)?;

// Couche de convolution
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// Normalisation par lots
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### Fonctions d'Activation

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// Fonctions d'activation de base
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// Fonctions d'activation paramétrées
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// Exemple d'utilisation
let activated = relu.forward(&input)?;
```

## 🚀 Module d'Accélération GPU

### Gestion des Dispositifs

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// Vérifier les dispositifs disponibles
let device_count = get_device_count()?;
let device = Device::best_available()?;            // Sélection du meilleur dispositif

// Configuration du dispositif
set_device(&device)?;

// Déplacer le tenseur vers GPU
let gpu_tensor = tensor.to_device(&device)?;
```

### Opérations CUDA

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// Opérations de dispositif CUDA
let cuda_device = CudaDevice::new(0)?;              // Utiliser GPU 0
let stats = memory_stats(0)?;                      // Statistiques mémoire
println!("Mémoire utilisée: {} MB", stats.used_memory / (1024 * 1024));
```

### Opérations Metal (macOS)

```rust
#[cfg(feature = "metal")]
use rustorch::gpu::metal::MetalDevice;

// Opérations de dispositif Metal
let metal_device = MetalDevice::new()?;
let gpu_tensor = tensor.to_metal(&metal_device)?;
```

## 🎯 Module Optimiseur (Optim)

### Optimiseurs de Base

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Optimiseur Adam
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// Optimiseur SGD
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// Étape d'optimisation
optimizer.zero_grad()?;
// ... calcul en avant et rétropropagation ...
optimizer.step()?;
```

### Planificateurs de Taux d'Apprentissage

```rust
use rustorch::optim::scheduler::{StepLR, CosineAnnealingLR, ReduceLROnPlateau};

// Planificateur à étapes
let step_scheduler = StepLR::new(&mut optimizer, 10, 0.1)?;

// Recuit cosinus
let cosine_scheduler = CosineAnnealingLR::new(&mut optimizer, 100)?;

// Réduction sur plateau
let plateau_scheduler = ReduceLROnPlateau::new(&mut optimizer, "min", 0.1, 10)?;

// Utilisation du planificateur
step_scheduler.step()?;
plateau_scheduler.step(validation_loss)?;
```

## 💾 Module de Sérialisation (Phase 9)

### Sauvegarde et Chargement de Modèles

```rust
use rustorch::serialization::{save_model, load_model, ModelFormat};

// Sauvegarde de modèle
save_model(&model, "model.pt", ModelFormat::PyTorch)?;
save_model(&model, "model.rustorch", ModelFormat::Native)?;

// Chargement de modèle
let loaded_model = load_model("model.pt", ModelFormat::PyTorch)?;
let native_model = load_model("model.rustorch", ModelFormat::Native)?;
```

### Compilation JIT

```rust
use rustorch::serialization::jit::{trace, script, JitModule};

// JIT basé sur traçage
let traced_module = trace(&model, &example_input)?;
let output = traced_module.forward(&input)?;

// JIT basé sur script
let scripted = script(&model)?;
let optimized_output = scripted.forward(&input)?;
```

## 🔢 Module d'Algèbre Linéaire (Linalg)

### Opérations de Décomposition

```rust
use rustorch::linalg::{svd, qr, eig, cholesky, lu};

// Décomposition en valeurs singulières
let (u, s, vt) = svd(&tensor, true)?;              // full_matrices=true

// Décomposition QR
let (q, r) = qr(&tensor, "reduced")?;

// Décomposition en valeurs propres
let (eigenvalues, eigenvectors) = eig(&tensor)?;

// Décomposition de Cholesky
let l = cholesky(&tensor)?;

// Décomposition LU
let (p, l, u) = lu(&tensor)?;
```

## 📖 Exemple d'Utilisation

### Régression Linéaire

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// Préparation des données
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// Définition du modèle
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// Boucle d'entraînement
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("Époque {}: Perte = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ Limitations Connues

1. **Limitation mémoire GPU**: La gestion explicite de la mémoire est requise pour les gros tenseurs (>8GB)
2. **Limitation WebAssembly**: Certaines opérations BLAS ne sont pas disponibles dans l'environnement WASM
3. **Apprentissage distribué**: Le backend NCCL n'est supporté que sur Linux
4. **Limitation Metal**: Certaines opérations avancées ne sont disponibles qu'avec le backend CUDA

## 🔗 Liens Connexes

- [README Principal](../README.md)
- [Documentation API WASM](WASM_API_DOCUMENTATION.md)
- [Guide Jupyter](jupyter-guide.md)
- [Dépôt GitHub](https://github.com/JunSuzukiJapan/RusTorch)
- [Package Crates.io](https://crates.io/crates/rustorch)

---

**Dernière Mise à Jour**: v0.5.15 | **Licence**: MIT | **Auteur**: Jun Suzuki