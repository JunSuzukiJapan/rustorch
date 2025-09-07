# Guide RusTorch WASM Jupyter Notebook

Un guide étape par étape pour utiliser facilement RusTorch WASM dans Jupyter Notebook, conçu pour les débutants.

## 📚 Table des Matières

1. [Exigences](#exigences)
2. [Instructions d'Installation](#instructions-dinstallation)
3. [Utilisation de Base](#utilisation-de-base)
4. [Exemples Pratiques](#exemples-pratiques)
5. [Résolution de Problèmes](#résolution-de-problèmes)
6. [FAQ](#faq)

## Exigences

### Exigences Minimales
- **Python 3.8+**
- **Jupyter Notebook** ou **Jupyter Lab**
- **Node.js 16+** (pour les builds WASM)
- **Rust** (dernière version stable)
- **wasm-pack** (pour convertir le code Rust en WASM)

### Environnement Recommandé
- Mémoire : 8Go ou plus
- Navigateur : Dernières versions de Chrome, Firefox, Safari
- OS : Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Instructions d'Installation

### 🚀 Démarrage Rapide (Recommandé)

#### Installateur Universel (Nouveau)
**La méthode la plus simple** : Un installateur qui détecte automatiquement votre environnement
```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/install_jupyter.sh | bash
```

**Ce qu'il fait :**
- 🔍 **Détection Automatique** : Détecte automatiquement votre environnement (OS, CPU, GPU)
- 🦀🐍 **Environnement Hybride** : Installe un environnement dual Python+Rust par défaut
- 📦 **Commande Globale** : Crée une commande `rustorch-jupyter` qui fonctionne partout
- ⚡ **Optimisation** : S'adapte à votre matériel (CUDA, Metal, WebGPU, CPU)

#### Méthode Classique
**Méthode traditionnelle** : Lance seulement Python avec RusTorch
```bash
./start_jupyter.sh
```

Ce script fait automatiquement :
- Crée et active l'environnement virtuel
- Installe les dépendances (numpy, jupyter, matplotlib)
- Construit les bindings Python RusTorch
- Lance Jupyter Lab avec le notebook de démonstration ouvert

#### Prochains Lancements
```bash
rustorch-jupyter          # Commande globale (après installateur)
# OU
./start_jupyter_quick.sh  # Menu interactif
```

### Installation Manuelle

#### Étape 1 : Installer les Outils de Base

```bash
# Vérifier la version Python
python --version

# Installer Jupyter Lab
pip install jupyterlab

# Installer Node.js (macOS avec Homebrew)
brew install node

# Installer Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Installer wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### Étape 2 : Construire RusTorch WASM

```bash
# Cloner le projet
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Ajouter la cible WASM
rustup target add wasm32-unknown-unknown

# Construire avec wasm-pack
wasm-pack build --target web --out-dir pkg
```

#### Étape 3 : Démarrer Jupyter

```bash
# Démarrer Jupyter Lab
jupyter lab
```

## Types d'Environnement

### 🦀🐍 Environnement Hybride (Par Défaut)
- **Idéal pour** : Développement ML complet
- **Fonctionnalités** : Kernels Python + Rust, bridge RusTorch, notebooks d'exemple
- **Matériel** : S'adapte au GPU disponible (CUDA/Metal/CPU)

### 🐍 Environnement Python
- **Idéal pour** : Développeurs Python qui veulent les fonctionnalités RusTorch
- **Fonctionnalités** : Kernel Python avec bindings Python RusTorch
- **Matériel** : Optimisé pour CPU/GPU

### ⚡ Environnement WebGPU
- **Idéal pour** : Accélération GPU basée sur navigateur
- **Fonctionnalités** : WebAssembly + WebGPU, optimisé pour Chrome
- **Matériel** : Navigateurs modernes avec support WebGPU

### 🦀 Environnement Kernel Rust
- **Idéal pour** : Développement natif Rust
- **Fonctionnalités** : Kernel evcxr, accès direct à la bibliothèque RusTorch
- **Matériel** : Performance native, toutes les fonctionnalités disponibles

## Utilisation de Base

### Créer des Tenseurs

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Tenseur 1D
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('Tenseur 1D :', vec.to_array());
    
    // Tenseur 2D (matrice)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // forme : 2 lignes, 3 colonnes
    );
    console.log('Forme tenseur 2D :', matrix.shape());
});
```

### Opérations de Base

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // Addition
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // Multiplication matricielle
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### Différenciation Automatique

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Créer tenseur avec suivi de gradient
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // Calcul : y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // Rétropropagation
    y.backward();
    
    // Obtenir gradient (dy/dx = 2x + 3 = 7 quand x=2)
    console.log('Gradient :', x.grad().to_array());
});
```

## Exemples Pratiques

### Régression Linéaire

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Préparer les données
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // Initialiser les paramètres
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // Boucle d'entraînement
    for (let epoch = 0; epoch < 100; epoch++) {
        // Prédiction : y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // Perte : MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // Calculer gradients
        loss.backward();
        
        // Mettre à jour paramètres
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // Réinitialiser gradients
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`Époque ${epoch} : Perte = ${loss.item()}`);
        }
    }
    
    console.log(`w final : ${w.item()}, b final : ${b.item()}`);
});
```

## Résolution de Problèmes

### 🚀 Accélérer le Noyau Rust (Recommandé)
Si l'exécution initiale est lente, activez le cache pour une amélioration significative des performances :

```bash
# Créer répertoire cache
mkdir -p ~/.config/evcxr

# Activer cache 500MB
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**Effets :**
- Première fois : Temps de compilation normal
- Exécutions suivantes : Pas de recompilation des dépendances (plusieurs fois plus rapide)
- La bibliothèque `rustorch` est aussi mise en cache après la première utilisation

**Note :** Après les mises à jour de bibliothèque, exécutez `:clear_cache` pour actualiser le cache

### Erreurs Communes

#### Erreur "RusTorch is not defined"
**Solution** : Toujours attendre RusTorchReady
```javascript
window.RusTorchReady.then((rt) => {
    // Utiliser RusTorch ici
});
```

#### Erreur "Failed to load WASM module"
**Solutions** :
1. Vérifier que le répertoire `pkg` a été généré correctement
2. Vérifier la console du navigateur pour les messages d'erreur
3. S'assurer que les chemins des fichiers WASM sont corrects

#### Erreur de Manque de Mémoire
**Solutions** :
```javascript
// Libérer mémoire explicitement
tensor.free();

// Utiliser des tailles de batch plus petites
const batchSize = 32;  // Utiliser 32 au lieu de 1000
```

### Conseils de Performance

1. **Utiliser le Traitement par Lots** : Traiter les données par lots plutôt qu'en boucles
2. **Gestion de Mémoire** : Libérer explicitement les gros tenseurs
3. **Types de Données Appropriés** : Utiliser f32 quand une haute précision n'est pas nécessaire

## FAQ

### Q : Puis-je utiliser ceci dans Google Colab ?
**R** : Oui, téléchargez les fichiers WASM et utilisez des chargeurs JavaScript personnalisés.

### Q : Puis-je mélanger code Python et WASM ?
**R** : Oui, utilisez IPython.display.Javascript pour passer des données entre Python et JavaScript.

### Q : Comment déboguer ?
**R** : Utilisez les outils développeur du navigateur (F12) et vérifiez l'onglet Console pour les erreurs.

### Q : Quelles fonctionnalités avancées sont disponibles ?
**R** : Actuellement supporte les opérations tensorielles de base, différenciation automatique et réseaux de neurones simples. Les couches CNN et RNN sont prévues.

## Prochaines Étapes

1. 📖 [API RusTorch WASM Détaillée](../wasm.md)
2. 🔬 [Exemples Avancés](../examples/)
3. 🚀 [Guide d'Optimisation Performance](../wasm-memory-optimization.md)

## Communauté et Support

- GitHub : [Dépôt RusTorch](https://github.com/JunSuzukiJapan/rustorch)
- Issues : Signaler bugs et demander fonctionnalités sur GitHub

---

Bon Apprentissage avec RusTorch WASM ! 🦀🔥📓