#!/bin/bash

# RusTorch Sample Notebooks Downloader
# サンプルノートブックダウンローダー

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect system language for default selection
detect_language() {
    local lang_code="en"  # Default to English
    
    # Check system locale
    if [[ -n "$LANG" ]]; then
        case "$LANG" in
            ja*) lang_code="ja" ;;
            zh*) lang_code="zh" ;;
            ko*) lang_code="ko" ;;
            es*) lang_code="es" ;;
            fr*) lang_code="fr" ;;
            de*) lang_code="de" ;;
            it*) lang_code="it" ;;
            pt*) lang_code="pt" ;;
            ru*) lang_code="ru" ;;
            *) lang_code="en" ;;
        esac
    fi
    
    echo "$lang_code"
}

# Download notebooks function
download_notebooks() {
    local target_dir="$1"
    local language="$2"
    local repo_url="https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main"
    
    echo -e "${GREEN}📥 Downloading sample notebooks...${NC}"
    echo -e "${GREEN}📥 サンプルノートブックをダウンロード中...${NC}"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Core notebooks (always download)
    echo "📝 Downloading core notebooks..."
    curl -sSL "$repo_url/notebooks/rustorch_demo.ipynb" -o "$target_dir/rustorch_demo.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/rustorch_rust_kernel_demo.ipynb" -o "$target_dir/rustorch_rust_kernel_demo.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/rustorch_rust_kernel_demo_ja.ipynb" -o "$target_dir/rustorch_rust_kernel_demo_ja.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/webgpu_ml_demo.ipynb" -o "$target_dir/webgpu_ml_demo.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/webgpu_performance_demo.ipynb" -o "$target_dir/webgpu_performance_demo.ipynb" 2>/dev/null || true
    
    # Download notebooks for specific language
    if [[ "$language" != "en" && "$language" != "ja" ]]; then
        echo "📝 Downloading $language language notebooks..."
        mkdir -p "$target_dir/$language"
        curl -sSL "$repo_url/notebooks/$language/rustorch_demo_${language}.ipynb" -o "$target_dir/$language/rustorch_demo_${language}.ipynb" 2>/dev/null || true
        curl -sSL "$repo_url/notebooks/$language/rustorch_rust_kernel_demo_${language}.ipynb" -o "$target_dir/$language/rustorch_rust_kernel_demo_${language}.ipynb" 2>/dev/null || true
    fi
    
    # Download English notebooks
    echo "📝 Downloading English notebooks..."
    mkdir -p "$target_dir/en"
    curl -sSL "$repo_url/notebooks/en/rustorch_demo_en.ipynb" -o "$target_dir/en/rustorch_demo_en.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/en/rustorch_rust_kernel_demo_en.ipynb" -o "$target_dir/en/rustorch_rust_kernel_demo_en.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/en/quickstart_en.md" -o "$target_dir/en/quickstart_en.md" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/en/python_api_reference.md" -o "$target_dir/en/python_api_reference.md" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/en/python_bindings_overview.md" -o "$target_dir/en/python_bindings_overview.md" 2>/dev/null || true
    
    # Download Japanese notebooks
    echo "📝 Downloading Japanese notebooks..."
    mkdir -p "$target_dir/ja"
    curl -sSL "$repo_url/notebooks/ja/quickstart_ja.md" -o "$target_dir/ja/quickstart_ja.md" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/ja/python_api_reference.md" -o "$target_dir/ja/python_api_reference.md" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/ja/python_bindings_overview.md" -o "$target_dir/ja/python_bindings_overview.md" 2>/dev/null || true
    
    # Download hybrid notebooks
    echo "📝 Downloading hybrid Python+Rust notebooks..."
    mkdir -p "$target_dir/hybrid"
    curl -sSL "$repo_url/notebooks/hybrid/python_rust_demo.ipynb" -o "$target_dir/hybrid/python_rust_demo.ipynb" 2>/dev/null || true
    curl -sSL "$repo_url/notebooks/hybrid/pure_rust_demo.ipynb" -o "$target_dir/hybrid/pure_rust_demo.ipynb" 2>/dev/null || true
    
    # Download README
    curl -sSL "$repo_url/notebooks/README.md" -o "$target_dir/README.md" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Sample notebooks downloaded successfully!${NC}"
    echo -e "${GREEN}✅ サンプルノートブックのダウンロードが完了しました！${NC}"
}

# Download all language notebooks
download_all_notebooks() {
    local target_dir="$1"
    local repo_url="https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main"
    
    echo -e "${GREEN}📥 Downloading all language notebooks...${NC}"
    echo -e "${GREEN}📥 全言語のノートブックをダウンロード中...${NC}"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # List of supported languages
    local languages=("en" "ja" "zh" "ko" "es" "fr" "de" "it" "pt" "ru")
    
    # Download core notebooks first
    download_notebooks "$target_dir" "en"
    
    # Download for each language
    for lang in "${languages[@]}"; do
        if [[ "$lang" != "en" ]]; then
            echo "📝 Downloading $lang language notebooks..."
            mkdir -p "$target_dir/$lang"
            
            # Try to download language-specific notebooks
            curl -sSL "$repo_url/notebooks/$lang/rustorch_demo_${lang}.ipynb" -o "$target_dir/$lang/rustorch_demo_${lang}.ipynb" 2>/dev/null || true
            curl -sSL "$repo_url/notebooks/$lang/rustorch_rust_kernel_demo_${lang}.ipynb" -o "$target_dir/$lang/rustorch_rust_kernel_demo_${lang}.ipynb" 2>/dev/null || true
            
            # Remove empty directories
            if [[ -d "$target_dir/$lang" ]] && [[ -z "$(ls -A "$target_dir/$lang")" ]]; then
                rmdir "$target_dir/$lang"
            fi
        fi
    done
    
    echo -e "${GREEN}✅ All language notebooks downloaded successfully!${NC}"
    echo -e "${GREEN}✅ 全言語ノートブックのダウンロードが完了しました！${NC}"
}

# Main function
main() {
    local target_dir="${1:-notebooks}"
    local language="${2:-$(detect_language)}"
    local download_all="${3:-false}"
    
    echo -e "${BLUE}🚀 RusTorch Sample Notebooks Downloader${NC}"
    echo -e "${BLUE}🚀 RusTorch サンプルノートブックダウンローダー${NC}"
    echo ""
    
    if [[ "$download_all" == "true" ]]; then
        download_all_notebooks "$target_dir"
    else
        download_notebooks "$target_dir" "$language"
    fi
    
    # List downloaded notebooks
    echo ""
    echo -e "${YELLOW}📋 Downloaded notebooks:${NC}"
    echo -e "${YELLOW}📋 ダウンロード済みノートブック:${NC}"
    find "$target_dir" -name "*.ipynb" -o -name "*.md" | sort | head -20
    
    local total_count=$(find "$target_dir" -name "*.ipynb" -o -name "*.md" | wc -l)
    if [[ $total_count -gt 20 ]]; then
        echo "... and $((total_count - 20)) more files"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Ready to explore RusTorch!${NC}"
    echo -e "${GREEN}🎉 RusTorchを探索する準備ができました！${NC}"
}

# Command line interface
case "${1:-}" in
    "all")
        main "${2:-notebooks}" "en" "true"
        ;;
    "help"|"--help"|"-h")
        echo "RusTorch Sample Notebooks Downloader"
        echo ""
        echo "Usage:"
        echo "  $0                    # Download notebooks for system language"
        echo "  $0 all                # Download all language notebooks"
        echo "  $0 [directory] [lang] # Download to specific directory with language"
        echo ""
        echo "Languages: en, ja, zh, ko, es, fr, de, it, pt, ru"
        echo ""
        ;;
    *)
        main "$@"
        ;;
esac