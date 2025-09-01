#!/bin/bash

# RusTorch Quick Start Script - Multilingual Support
# Auto-detects system language for international users
# Supports: English, Japanese, Spanish, French, German, Chinese, Korean
# 
# Usage: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start.sh | bash
# 使用法: curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/quick_start.sh | bash

set -e

# Language Detection and Message System
detect_language() {
    local lang_code
    
    # Try multiple methods to detect language
    if [[ -n "${LC_ALL:-}" ]]; then
        lang_code="${LC_ALL%.*}"
    elif [[ -n "${LC_MESSAGES:-}" ]]; then
        lang_code="${LC_MESSAGES%.*}"
    elif [[ -n "${LANG:-}" ]]; then
        lang_code="${LANG%.*}"
    else
        lang_code="en_US"
    fi
    
    # Extract language prefix
    lang_code="${lang_code%_*}"
    
    case "$lang_code" in
        ja) echo "ja" ;;
        es) echo "es" ;;
        fr) echo "fr" ;;
        de) echo "de" ;;
        zh|zh_CN|zh_TW) echo "zh" ;;
        ko) echo "ko" ;;
        *) echo "en" ;;
    esac
}

# Multilingual message function
msg() {
    local key="$1"
    local lang="${DETECTED_LANG:-en}"
    
    case "$key" in
        "welcome_title")
            case "$lang" in
                en) echo "🚀 RusTorch Quick Start" ;;
                ja) echo "🚀 RusTorch クイックスタート" ;;
                es) echo "🚀 Inicio Rápido de RusTorch" ;;
                fr) echo "🚀 Démarrage Rapide RusTorch" ;;
                de) echo "🚀 RusTorch Schnellstart" ;;
                zh) echo "🚀 RusTorch 快速开始" ;;
                ko) echo "🚀 RusTorch 빠른 시작" ;;
            esac ;;
        "creating_workspace")
            case "$lang" in
                en) echo "📁 Creating RusTorch workspace" ;;
                ja) echo "📁 RusTorchワークスペースを作成" ;;
                es) echo "📁 Creando espacio de trabajo RusTorch" ;;
                fr) echo "📁 Création de l'espace de travail RusTorch" ;;
                de) echo "📁 RusTorch-Arbeitsbereich erstellen" ;;
                zh) echo "📁 创建 RusTorch 工作空间" ;;
                ko) echo "📁 RusTorch 작업공간 생성 중" ;;
            esac ;;
        "dir_exists")
            case "$lang" in
                en) echo "⚠️  Directory exists. Updating..." ;;
                ja) echo "⚠️  ディレクトリが存在します。更新中..." ;;
                es) echo "⚠️  El directorio existe. Actualizando..." ;;
                fr) echo "⚠️  Le répertoire existe. Mise à jour..." ;;
                de) echo "⚠️  Verzeichnis existiert. Aktualisierung..." ;;
                zh) echo "⚠️  目录已存在。正在更新..." ;;
                ko) echo "⚠️  디렉터리가 존재합니다. 업데이트 중..." ;;
            esac ;;
        "downloading")
            case "$lang" in
                en) echo "📥 Downloading RusTorch..." ;;
                ja) echo "📥 RusTorchをダウンロード中..." ;;
                es) echo "📥 Descargando RusTorch..." ;;
                fr) echo "📥 Téléchargement de RusTorch..." ;;
                de) echo "📥 RusTorch wird heruntergeladen..." ;;
                zh) echo "📥 正在下载 RusTorch..." ;;
                ko) echo "📥 RusTorch 다운로드 중..." ;;
            esac ;;
        "checking_requirements")
            case "$lang" in
                en) echo "🔍 Checking system requirements..." ;;
                ja) echo "🔍 システム要件を確認中..." ;;
                es) echo "🔍 Verificando requisitos del sistema..." ;;
                fr) echo "🔍 Vérification des prérequis..." ;;
                de) echo "🔍 Systemanforderungen prüfen..." ;;
                zh) echo "🔍 检查系统要求..." ;;
                ko) echo "🔍 시스템 요구사항 확인 중..." ;;
            esac ;;
        "python_required")
            case "$lang" in
                en) echo "❌ Python 3 is required. Please install Python 3.8+" ;;
                ja) echo "❌ Python 3が必要です。Python 3.8+をインストールしてください" ;;
                es) echo "❌ Se requiere Python 3. Por favor instale Python 3.8+" ;;
                fr) echo "❌ Python 3 est requis. Veuillez installer Python 3.8+" ;;
                de) echo "❌ Python 3 ist erforderlich. Bitte installieren Sie Python 3.8+" ;;
                zh) echo "❌ 需要 Python 3。请安装 Python 3.8+" ;;
                ko) echo "❌ Python 3가 필요합니다. Python 3.8+를 설치해주세요" ;;
            esac ;;
        "rust_not_found")
            case "$lang" in
                en) echo "⚠️  Rust not found - will use pre-built binaries" ;;
                ja) echo "⚠️  Rustが見つかりません - ビルド済みバイナリを使用します" ;;
                es) echo "⚠️  Rust no encontrado - se usarán binarios precompilados" ;;
                fr) echo "⚠️  Rust non trouvé - utilisation des binaires pré-compilés" ;;
                de) echo "⚠️  Rust nicht gefunden - verwende vorkompilierte Binärdateien" ;;
                zh) echo "⚠️  未找到 Rust - 将使用预编译二进制文件" ;;
                ko) echo "⚠️  Rust를 찾을 수 없습니다 - 미리 빌드된 바이너리를 사용합니다" ;;
            esac ;;
        "setup_complete")
            case "$lang" in
                en) echo "🎉 Setup Complete!" ;;
                ja) echo "🎉 セットアップ完了！" ;;
                es) echo "🎉 ¡Configuración Completa!" ;;
                fr) echo "🎉 Configuration Terminée!" ;;
                de) echo "🎉 Setup Abgeschlossen!" ;;
                zh) echo "🎉 设置完成！" ;;
                ko) echo "🎉 설정 완료！" ;;
            esac ;;
        "available_notebooks")
            case "$lang" in
                en) echo "📋 Available notebooks:" ;;
                ja) echo "📋 利用可能なノートブック:" ;;
                es) echo "📋 Notebooks disponibles:" ;;
                fr) echo "📋 Notebooks disponibles:" ;;
                de) echo "📋 Verfügbare Notebooks:" ;;
                zh) echo "📋 可用笔记本：" ;;
                ko) echo "📋 사용 가능한 노트북:" ;;
            esac ;;
        "jupyter_info")
            case "$lang" in
                en) echo "🌐 Jupyter Lab will open at: http://localhost:8888" ;;
                ja) echo "🌐 Jupyter Labは次のURLで開きます: http://localhost:8888" ;;
                es) echo "🌐 Jupyter Lab se abrirá en: http://localhost:8888" ;;
                fr) echo "🌐 Jupyter Lab s'ouvrira à: http://localhost:8888" ;;
                de) echo "🌐 Jupyter Lab öffnet sich unter: http://localhost:8888" ;;
                zh) echo "🌐 Jupyter Lab 将在此处打开: http://localhost:8888" ;;
                ko) echo "🌐 Jupyter Lab이 다음에서 열립니다: http://localhost:8888" ;;
            esac ;;
        "stop_info")
            case "$lang" in
                en) echo "🛑 Press Ctrl+C to stop" ;;
                ja) echo "🛑 停止するにはCtrl+Cを押してください" ;;
                es) echo "🛑 Presiona Ctrl+C para detener" ;;
                fr) echo "🛑 Appuyez sur Ctrl+C pour arrêter" ;;
                de) echo "🛑 Drücken Sie Ctrl+C zum Stoppen" ;;
                zh) echo "🛑 按 Ctrl+C 停止" ;;
                ko) echo "🛑 중지하려면 Ctrl+C를 누르세요" ;;
            esac ;;
        "documentation_info")
            case "$lang" in
                en) echo "📖 Documentation: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
                ja) echo "📖 ドキュメント: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide.md" ;;
                es) echo "📖 Documentación: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
                fr) echo "📖 Documentation: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
                de) echo "📖 Dokumentation: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
                zh) echo "📖 文档: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
                ko) echo "📖 문서: https://github.com/JunSuzukiJapan/rustorch/blob/main/docs/jupyter-wasm-guide-en.md" ;;
            esac ;;
    esac
}

# Detect system language
DETECTED_LANG=$(detect_language)

# Display welcome message in user's language
echo "$(msg "welcome_title")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌍 Language detected: $DETECTED_LANG | $(msg "documentation_info")"
echo ""

# Create temporary directory for RusTorch
RUSTORCH_DIR="$HOME/rustorch-jupyter"
echo "$(msg "creating_workspace"): $RUSTORCH_DIR"

if [ -d "$RUSTORCH_DIR" ]; then
    echo "$(msg "dir_exists")"
    cd "$RUSTORCH_DIR"
    git pull origin main || echo "Git pull failed, continuing..."
else
    echo "$(msg "downloading")"
    git clone --depth 1 https://github.com/JunSuzukiJapan/rustorch.git "$RUSTORCH_DIR"
    cd "$RUSTORCH_DIR"
fi

# Check system requirements
echo ""
echo "$(msg "checking_requirements")"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python: $PYTHON_VERSION"
else
    echo "$(msg "python_required")"
    exit 1
fi

# Check Rust (optional for quick start)
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    echo "✅ Rust: $RUST_VERSION"
    HAS_RUST=true
else
    echo "$(msg "rust_not_found")"
    HAS_RUST=false
fi

# Check for pip
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 available"
elif command -v pip &> /dev/null; then
    echo "✅ pip available"
    alias pip3=pip
else
    echo "❌ pip is required. Please install pip"
    echo "❌ pipが必要です。pipをインストールしてください"
    exit 1
fi

echo ""
echo "🛠️  Setting up Python environment..."
echo "🛠️  Python環境をセットアップ中..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    echo "📦 仮想環境を作成中..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
echo "🔌 仮想環境をアクティベート中..."
source .venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
echo "📚 Python依存関係をインストール中..."
pip install --upgrade pip
pip install jupyter jupyterlab matplotlib pandas numpy maturin

# Try to install pre-built wheel from PyPI if available
echo ""
echo "🎯 Installing RusTorch..."
echo "🎯 RusTorchをインストール中..."

if pip install rustorch 2>/dev/null; then
    echo "✅ Installed RusTorch from PyPI"
    echo "✅ PyPIからRusTorchをインストールしました"
    RUSTORCH_INSTALLED=true
else
    echo "⚠️  PyPI package not available, building from source..."
    echo "⚠️  PyPIパッケージが利用できません、ソースからビルド中..."
    RUSTORCH_INSTALLED=false
    
    if [ "$HAS_RUST" = true ]; then
        echo "🔧 Building RusTorch Python bindings..."
        echo "🔧 RusTorch Pythonバインディングをビルド中..."
        maturin develop --features python --release
        RUSTORCH_INSTALLED=true
    else
        echo "❌ Cannot build from source without Rust"
        echo "❌ Rustなしではソースからビルドできません"
        echo ""
        echo "📋 Please install Rust first:"
        echo "📋 まずRustをインストールしてください:"
        echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo ""
        echo "Then run this script again."
        echo "その後、このスクリプトを再度実行してください。"
        exit 1
    fi
fi

if [ "$RUSTORCH_INSTALLED" = true ]; then
    echo ""
    echo "🧪 Testing RusTorch installation..."
    echo "🧪 RusTorchインストールをテスト中..."
    
    python3 -c "
import rustorch
print('✅ RusTorch imported successfully!')
print('✅ RusTorchのインポートに成功しました！')
print(f'📍 RusTorch version: {rustorch.__version__ if hasattr(rustorch, \"__version__\") else \"unknown\"}')
" || echo "⚠️  Import test failed, but installation may still work"

    echo ""
    echo "$(msg "setup_complete")"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "$(msg "available_notebooks")"
    echo "   • rustorch_demo.ipynb - Basic tensor operations"
    echo "   • webgpu_ml_demo.ipynb - WebGPU acceleration demo"  
    echo "   • webgpu_performance_demo.ipynb - Performance benchmarks"
    echo ""
    echo "$(msg "jupyter_info")"
    echo "$(msg "stop_info")"
    echo ""
    
    # Launch Jupyter Lab
    exec jupyter lab --port=8888 --no-browser
fi