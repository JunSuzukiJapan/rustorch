# Guida RusTorch WASM Jupyter Notebook

Una guida passo-passo per usare facilmente RusTorch WASM in Jupyter Notebook, progettata per principianti.

## 📚 Indice

1. [Requisiti](#requisiti)
2. [Istruzioni di Installazione](#istruzioni-di-installazione)
3. [Uso Base](#uso-base)
4. [Esempi Pratici](#esempi-pratici)
5. [Risoluzione Problemi](#risoluzione-problemi)
6. [FAQ](#faq)

## Requisiti

### Requisiti Minimi
- **Python 3.8+**
- **Jupyter Notebook** o **Jupyter Lab**
- **Node.js 16+** (per build WASM)
- **Rust** (ultima versione stabile)
- **wasm-pack** (per convertire codice Rust in WASM)

### Ambiente Raccomandato
- Memoria: 8GB o più
- Browser: Ultime versioni di Chrome, Firefox, Safari
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Istruzioni di Installazione

### 🚀 Avvio Rapido (Raccomandato)

**Metodo più semplice**: Lancia Jupyter Lab con un comando
```bash
./start_jupyter.sh
```

Questo script automaticamente:
- Crea e attiva ambiente virtuale
- Installa dipendenze (numpy, jupyter, matplotlib)
- Costruisce i binding Python RusTorch
- Lancia Jupyter Lab con notebook demo aperto

### Installazione Manuale

#### Passo 1: Installare Strumenti Base

```bash
# Verificare versione Python
python --version

# Installare Jupyter Lab
pip install jupyterlab

# Installare Node.js (macOS con Homebrew)
brew install node

# Installare Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Installare wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### Passo 2: Costruire RusTorch WASM

```bash
# Clonare progetto
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Aggiungere target WASM
rustup target add wasm32-unknown-unknown

# Costruire con wasm-pack
wasm-pack build --target web --out-dir pkg
```

#### Passo 3: Avviare Jupyter

```bash
# Avviare Jupyter Lab
jupyter lab
```

## Uso Base

### Creare Tensori

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Tensore 1D
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('Tensore 1D:', vec.to_array());
    
    // Tensore 2D (matrice)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // forma: 2 righe, 3 colonne
    );
    console.log('Forma tensore 2D:', matrix.shape());
});
```

### Operazioni Base

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // Addizione
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // Moltiplicazione matriciale
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### Differenziazione Automatica

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Creare tensore con tracking gradiente
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // Calcolo: y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // Backpropagation
    y.backward();
    
    // Ottenere gradiente (dy/dx = 2x + 3 = 7 quando x=2)
    console.log('Gradiente:', x.grad().to_array());
});
```

## Esempi Pratici

### Regressione Lineare

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Preparare dati
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // Inizializzare parametri
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // Loop di training
    for (let epoch = 0; epoch < 100; epoch++) {
        // Predizione: y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // Loss: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // Calcolare gradienti
        loss.backward();
        
        // Aggiornare parametri
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // Reset gradienti
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`Epoca ${epoch}: Loss = ${loss.item()}`);
        }
    }
    
    console.log(`w finale: ${w.item()}, b finale: ${b.item()}`);
});
```

## Risoluzione Problemi

### 🚀 Accelerare Kernel Rust (Raccomandato)
Se l'esecuzione iniziale è lenta, abilita la cache per miglioramento significativo delle prestazioni:

```bash
# Creare directory cache
mkdir -p ~/.config/evcxr

# Abilitare cache 500MB
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**Effetti:**
- Prima volta: Tempo di compilazione normale
- Esecuzioni successive: Nessuna ricompilazione delle dipendenze (diverse volte più veloce)
- La libreria `rustorch` viene anche memorizzata nella cache dopo il primo uso

### Errori Comuni

#### Errore "RusTorch is not defined"
**Soluzione**: Aspettare sempre RusTorchReady
```javascript
window.RusTorchReady.then((rt) => {
    // Usare RusTorch qui
});
```

#### Errore "Failed to load WASM module"
**Soluzioni**:
1. Verificare che directory `pkg` sia stata generata correttamente
2. Controllare console browser per messaggi errore
3. Assicurarsi che percorsi file WASM siano corretti

#### Errore Memoria Insufficiente
**Soluzioni**:
```javascript
// Liberare memoria esplicitamente
tensor.free();

// Usare dimensioni batch più piccole
const batchSize = 32;  // Usare 32 invece di 1000
```

### Suggerimenti Prestazioni

1. **Usare Elaborazione Batch**: Elaborare dati in batch invece di loop
2. **Gestione Memoria**: Liberare esplicitamente tensori grandi
3. **Tipi Dati Appropriati**: Usare f32 quando alta precisione non necessaria

## FAQ

### D: Posso usare questo in Google Colab?
**R**: Sì, carica file WASM e usa loader JavaScript personalizzati.

### D: Posso mescolare codice Python e WASM?
**R**: Sì, usa IPython.display.Javascript per passare dati tra Python e JavaScript.

### D: Come faccio debug?
**R**: Usa strumenti sviluppatore browser (F12) e controlla tab Console per errori.

### D: Quali funzionalità avanzate sono disponibili?
**R**: Attualmente supporta operazioni tensoriali base, differenziazione automatica e reti neurali semplici. Layer CNN e RNN sono pianificati.

## Prossimi Passi

1. 📖 [API RusTorch WASM Dettagliata](../wasm.md)
2. 🔬 [Esempi Avanzati](../examples/)
3. 🚀 [Guida Ottimizzazione Prestazioni](../wasm-memory-optimization.md)

## Comunità e Supporto

- GitHub: [Repository RusTorch](https://github.com/JunSuzukiJapan/rustorch)
- Issues: Segnala bug e richiedi funzionalità su GitHub

---

Buon Apprendimento con RusTorch WASM! 🦀🔥📓