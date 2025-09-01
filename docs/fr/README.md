# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**Une bibliothèque d'apprentissage profond prête pour la production en Rust avec une API similaire à PyTorch, l'accélération GPU et des performances de niveau entreprise**

RusTorch est une bibliothèque d'apprentissage profond entièrement fonctionnelle qui exploite la sécurité et les performances de Rust, fournissant des opérations tensorielles complètes, la différenciation automatique, des couches de réseau de neurones, des architectures de transformateur, l'accélération GPU multi-backend (CUDA/Metal/OpenCL), des optimisations SIMD avancées, la gestion de mémoire de niveau entreprise, la validation de données et l'assurance qualité, ainsi que des systèmes de débogage et de journalisation complets.

## ✨ Fonctionnalités

- 🔥 **Opérations Tensorielles Complètes** : Opérations mathématiques, diffusion, indexation et statistiques
- 🤖 **Architecture Transformer** : Implémentation complète de transformer avec attention multi-têtes
- 🧮 **Décomposition Matricielle** : SVD, QR, décomposition en valeurs propres avec compatibilité PyTorch
- 🧠 **Différenciation Automatique** : Graphe computationnel basé sur bande pour le calcul de gradient
- 🚀 **Moteur d'Exécution Dynamique** : Compilation JIT et optimisation à l'exécution
- 🏗️ **Couches de Réseaux de Neurones** : Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, et plus
- ⚡ **Optimisations Multi-Plateformes** : SIMD (AVX2/SSE/NEON), spécifiques à la plateforme et optimisations conscientes du matériel
- 🎮 **Intégration GPU** : Support CUDA/Metal/OpenCL avec sélection automatique de périphérique
- 🌐 **Support WebAssembly** : ML complet dans le navigateur avec couches de réseau de neurones, vision par ordinateur et inférence en temps réel
- 🎮 **Intégration WebGPU** : Accélération GPU optimisée Chrome avec repli CPU pour la compatibilité multi-navigateurs
- 📁 **Support de Formats de Modèles** : Safetensors, inférence ONNX, compatibilité state dict PyTorch
- ✅ **Prêt pour la Production** : 968 tests réussis, système de gestion d'erreurs unifié
- 📐 **Fonctions Mathématiques Améliorées** : Ensemble complet de fonctions mathématiques (exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **Surcharges d'Opérateurs Avancées** : Support complet d'opérateurs pour tenseurs avec opérations scalaires et affectations en place
- 📈 **Optimiseurs Avancés** : SGD, Adam, AdamW, RMSprop, AdaGrad avec planificateurs de taux d'apprentissage
- 🔍 **Validation de Données et Assurance Qualité** : Analyse statistique, détection d'anomalies, vérification de cohérence, surveillance en temps réel
- 🐛 **Débogage et Journalisation Complets** : Journalisation structurée, profilage des performances, suivi de mémoire, alertes automatisées

## 🚀 Démarrage Rapide

**📓 Pour le guide complet de configuration Jupyter, voir [README_JUPYTER.md](../../README_JUPYTER.md)**

### Démo Python Jupyter Lab

📓 **[Guide Complet Jupyter](../../README_JUPYTER.md)** | **[Guide Jupyter](jupyter-guide.md)**

#### Démo CPU Standard
Lancez RusTorch avec Jupyter Lab en une commande :

```bash
./start_jupyter.sh
```

#### Démo Accélérée WebGPU
Lancez RusTorch avec support WebGPU pour l'accélération GPU basée sur navigateur :

```bash
./start_jupyter_webgpu.sh
```

Les deux scripts vont :
- 📦 Créer l'environnement virtuel automatiquement
- 🔧 Construire les bindings Python RusTorch
- 🚀 Lancer Jupyter Lab avec le notebook de démonstration
- 📍 Ouvrir le notebook de démonstration prêt à exécuter

**Fonctionnalités WebGPU :**
- 🌐 Accélération GPU basée sur navigateur
- ⚡ Opérations matricielles haute performance dans le navigateur
- 🔄 Repli automatique vers CPU quand GPU indisponible
- 🎯 Optimisé Chrome/Edge (navigateurs recommandés)

#### Noyau Rust pour Jupyter
Lancez le noyau Rust natif dans Jupyter (evcxr_jupyter) :

```bash
./quick_start_rust_kernel.sh
```

Cela va :
- 🦀 Installer le noyau Rust evcxr_jupyter
- 📓 Créer un notebook de démo Rust kernel
- 🚀 Lancer Jupyter avec support Rust natif
- 📍 Opérations tensorielles directes en Rust

### Installation

Ajoutez ceci à votre `Cargo.toml` :

```toml
[dependencies]
rustorch = "0.5.10"

# Fonctionnalités optionnelles
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Opérations d'algèbre linéaire (SVD, QR, valeurs propres)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # Support WebAssembly pour ML navigateur
webgpu = ["rustorch/webgpu"]            # Accélération WebGPU optimisée Chrome

# Pour désactiver les fonctionnalités linalg (éviter les dépendances OpenBLAS/LAPACK) :
rustorch = { version = "0.5.10", default-features = false }
```

### Utilisation de Base

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // Créer des tenseurs
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Opérations de base avec surcharges d'opérateurs
    let c = &a + &b;  // Addition élément par élément
    let d = &a - &b;  // Soustraction élément par élément
    let e = &a * &b;  // Multiplication élément par élément
    let f = &a / &b;  // Division élément par élément
    
    // Opérations scalaires
    let g = &a + 10.0;  // Ajouter scalaire à tous les éléments
    let h = &a * 2.0;   // Multiplier par scalaire
    
    // Fonctions mathématiques
    let exp_result = a.exp();   // Fonction exponentielle
    let ln_result = a.ln();     // Logarithme naturel
    let sin_result = a.sin();   // Fonction sinus
    let sqrt_result = a.sqrt(); // Racine carrée
    
    // Opérations matricielles
    let matmul_result = a.matmul(&b);  // Multiplication matricielle
    
    // Opérations d'algèbre linéaire (nécessite la fonctionnalité linalg)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // Décomposition SVD
        let qr_result = a.qr();         // Décomposition QR
        let eig_result = a.eigh();      // Décomposition en valeurs propres
    }
    
    // Optimiseurs avancés avec planification de taux d'apprentissage
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Échauffement à 0.1 sur 5 époques
    
    println!("Forme : {:?}", c.shape());
    println!("Résultat : {:?}", c.as_slice());
}
```

### Utilisation WebAssembly

Pour les applications ML basées sur navigateur :

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // Couches de réseau de neurones
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // Fonctions mathématiques améliorées
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // Distributions statistiques
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // Optimiseurs pour l'entraînement
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // taux_apprentissage, momentum
    
    // Traitement d'image
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // Passage avant
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('Prédictions ML navigateur :', predictions);
}
```

## 📚 Documentation

- **[Guide de Démarrage](../getting-started.md)** - Utilisation de base et exemples
- **[Fonctionnalités](../features.md)** - Liste complète des fonctionnalités et spécifications
- **[Performance](../performance.md)** - Benchmarks et détails d'optimisation
- **[Guide Jupyter WASM](jupyter-guide.md)** - Configuration pas à pas de Jupyter Notebook

### WebAssembly et ML Navigateur
- **[Guide WebAssembly](../WASM_GUIDE.md)** - Intégration WASM complète et référence API
- **[Intégration WebGPU](../WEBGPU_INTEGRATION.md)** - Accélération GPU optimisée Chrome

### Production et Opérations
- **[Guide d'Accélération GPU](../GPU_ACCELERATION_GUIDE.md)** - Configuration et utilisation GPU
- **[Guide de Production](../PRODUCTION_GUIDE.md)** - Déploiement et mise à l'échelle

## 📊 Performance

**Résultats de benchmarks récents :**

| Opération | Performance | Détails |
|-----------|-------------|---------|
| **Décomposition SVD** | ~1ms (matrice 8x8) | ✅ Basé sur LAPACK |
| **Décomposition QR** | ~24μs (matrice 8x8) | ✅ Décomposition rapide |
| **Valeurs Propres** | ~165μs (matrice 8x8) | ✅ Matrices symétriques |
| **FFT Complexe** | 10-312μs (8-64 échantillons) | ✅ Optimisé Cooley-Tukey |
| **Réseau de Neurones** | 1-7s entraînement | ✅ Démo Boston housing |
| **Fonctions d'Activation** | <1μs | ✅ ReLU, Sigmoid, Tanh |

## 🧪 Tests

**968 tests réussis** - Assurance qualité prête pour la production avec système de gestion d'erreurs unifié.

```bash
# Exécuter tous les tests
cargo test --no-default-features

# Exécuter tests avec fonctionnalités d'algèbre linéaire
cargo test --features linalg
```

## 🤝 Contribution

Nous accueillons les contributions ! Voyez les domaines où l'aide est particulièrement nécessaire :

- **🎯 Précision des Fonctions Spéciales** : Améliorer la précision numérique
- **⚡ Optimisation des Performances** : Améliorations SIMD, optimisation GPU
- **🧪 Tests** : Cas de test plus complets
- **📚 Documentation** : Exemples, tutoriels, améliorations
- **🌐 Support de Plateformes** : WebAssembly, plateformes mobiles

## Licence

Sous licence soit :

 * Licence Apache, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) ou http://www.apache.org/licenses/LICENSE-2.0)
 * Licence MIT ([LICENSE-MIT](../../LICENSE-MIT) ou http://opensource.org/licenses/MIT)

à votre choix.