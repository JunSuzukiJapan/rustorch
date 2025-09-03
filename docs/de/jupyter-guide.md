# RusTorch WASM Jupyter Notebook Anleitung

Eine schrittweise Anleitung zur einfachen Verwendung von RusTorch WASM in Jupyter Notebook, entwickelt für Anfänger.

## 📚 Inhaltsverzeichnis

1. [Anforderungen](#anforderungen)
2. [Setup-Anweisungen](#setup-anweisungen)
3. [Grundlegende Verwendung](#grundlegende-verwendung)
4. [Praktische Beispiele](#praktische-beispiele)
5. [Fehlerbehebung](#fehlerbehebung)
6. [FAQ](#faq)

## Anforderungen

### Mindestanforderungen
- **Python 3.8+**
- **Jupyter Notebook** oder **Jupyter Lab**
- **Node.js 16+** (für WASM-Builds)
- **Rust** (neueste stabile Version)
- **wasm-pack** (um Rust-Code zu WASM zu konvertieren)

### Empfohlene Umgebung
- Speicher: 8GB oder mehr
- Browser: Neueste Versionen von Chrome, Firefox, Safari
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Setup-Anweisungen

### 🚀 Schnellstart (Empfohlen)

**Einfachste Methode**: Jupyter Lab mit einem Befehl starten
```bash
./start_jupyter.sh
```

Dieses Skript führt automatisch aus:
- Erstellt und aktiviert virtuelle Umgebung
- Installiert Abhängigkeiten (numpy, jupyter, matplotlib)
- Erstellt RusTorch Python-Bindings
- Startet Jupyter Lab mit geöffnetem Demo-Notebook

### Manuelle Einrichtung

#### Schritt 1: Grundlegende Tools installieren

```bash
# Python-Version prüfen
python --version

# Rust installieren (falls nicht vorhanden)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# wasm-pack installieren
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Node.js prüfen
node --version
npm --version
```

#### Schritt 2: Python-Umgebung einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv rustorch_env

# Aktivieren (Linux/macOS)
source rustorch_env/bin/activate

# Aktivieren (Windows)
rustorch_env\\Scripts\\activate

# Abhängigkeiten installieren
pip install jupyter numpy matplotlib seaborn pandas
```

#### Schritt 3: RusTorch WASM erstellen

```bash
# WASM-Package erstellen
wasm-pack build --target web --features wasm

# Jupyter Lab starten
jupyter lab
```

## Grundlegende Verwendung

### WASM-Module in Jupyter laden

```javascript
%%javascript
// RusTorch WASM laden
import init, { WasmTensor, WasmAdvancedMath } from './pkg/rustorch.js';

async function setupRusTorch() {
    await init();
    
    // Grundlegendes Tensor-Beispiel
    const data = [1.0, 2.0, 3.0, 4.0];
    const shape = [2, 2];
    const tensor = new WasmTensor(data, shape);
    
    console.log('Tensor erstellt:', tensor.data());
    console.log('Tensor-Form:', tensor.shape());
    
    // Mathematische Operationen
    const math = new WasmAdvancedMath();
    const result = math.exp(tensor);
    console.log('Exponential-Ergebnis:', result.data());
}

setupRusTorch();
```

### Python-Integration

```python
# Python-Zelle
import numpy as np
import matplotlib.pyplot as plt

# Daten für WASM vorbereiten
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(f"Eingabedaten: {data}")

# Ergebnisse visualisieren
plt.figure(figsize=(10, 6))
plt.plot(data, 'o-', label='Original')
plt.legend()
plt.title('RusTorch WASM Tensor-Operationen')
plt.show()
```

## Praktische Beispiele

### 1. Tensor-Arithmetik

```javascript
%%javascript
// Tensor-Operationen Demo
const a = new WasmTensor([1, 2, 3, 4], [2, 2]);
const b = new WasmTensor([2, 3, 4, 5], [2, 2]);

// Addition
const sum = a.add(b);
console.log('Addition:', sum.data());

// Element-weise Multiplikation
const product = a.multiply(b);
console.log('Multiplikation:', product.data());
```

### 2. Erweiterte Mathematik

```javascript
%%javascript
const math = new WasmAdvancedMath();
const tensor = new WasmTensor([0.5, 1.0, 1.5, 2.0], [4]);

// Trigonometrische Funktionen
const sin_result = math.sin(tensor);
const cos_result = math.cos(tensor);
const tan_result = math.tan(tensor);

console.log('Sin:', sin_result.data());
console.log('Cos:', cos_result.data());
console.log('Tan:', tan_result.data());
```

### 3. Qualitätsmetriken

```javascript
%%javascript
const quality = new WasmQualityMetrics(0.8);
const data_tensor = new WasmTensor([...Array(100)].map(() => Math.random()), [100]);

// Datenqualität bewerten
const quality_score = quality.overall_quality(data_tensor);
console.log('Qualitätsscore:', quality_score);

// Detaillierter Bericht
const report = quality.quality_report(data_tensor);
console.log('Qualitätsbericht:', report);
```

### 4. Anomalieerkennung

```javascript
%%javascript
const detector = new WasmAnomalyDetector(2.0, 50);
const time_series = new WasmTimeSeriesDetector(30, 12);

// Statistische Anomalien erkennen
const anomalies = detector.detect_statistical(data_tensor);
console.log('Gefundene Anomalien:', anomalies.length());

// Echtzeit-Erkennung
for (let i = 0; i < data_tensor.data().length; i++) {
    const value = data_tensor.data()[i];
    const anomaly = detector.detect_realtime(value);
    if (anomaly) {
        console.log(`Anomalie bei Index ${i}: ${value}`);
    }
}
```

## Fehlerbehebung

### Häufige Probleme

#### Problem: "Module nicht gefunden"
```bash
# Lösung: WASM erneut erstellen
wasm-pack build --target web --features wasm
```

#### Problem: "Speicherfehler"
```javascript
// Lösung: Memory Manager initialisieren
import { MemoryManager } from './pkg/rustorch.js';
MemoryManager.init_pool(200);  // Pool-Größe erhöhen
```

#### Problem: "Langsame Performance"
```javascript
// Lösung: Cache aktivieren und Garbage Collection verwenden
const pipeline = new WasmTransformPipeline(true);  // Cache aktiviert
MemoryManager.gc();  // Speicher freigeben
```

### Debug-Tipps

1. **Browser-Konsole prüfen**: Öffnen Sie die Entwicklertools (F12)
2. **Speicherverbrauch überwachen**: 
   ```javascript
   console.log('Speicherstatistiken:', MemoryManager.get_stats());
   ```
3. **Fehler-Callbacks verwenden**:
   ```javascript
   try {
       const result = tensor.operation();
   } catch (error) {
       console.error('Fehler:', error.message);
   }
   ```

## FAQ

### F: Welche Browser werden unterstützt?
A: Chrome 90+, Firefox 89+, Safari 14+, Edge 90+

### F: Kann ich RusTorch WASM in Node.js verwenden?
A: Ja, aber diese Anleitung konzentriert sich auf Browser-Verwendung

### F: Wie groß ist das WASM-Bundle?
A: ~2-5MB komprimiert, abhängig von aktivierten Features

### F: Kann ich GPU-Beschleunigung verwenden?
A: Ja, mit WebGPU in unterstützten Browsern (hauptsächlich Chrome)

### F: Ist es produktionsbereit?
A: Ja, RusTorch ist vollständig getestet und optimiert für Produktionsumgebungen

---

**🎯 Für erweiterte Beispiele siehe [examples/](../../examples/) Verzeichnis**