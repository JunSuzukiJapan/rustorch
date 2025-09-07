# RusTorch Jupyter Notebooks

このディレクトリには、RusTorchをJupyterで学習・使用するためのノートブックが含まれています。

## 📁 Directory Structure

```
notebooks/
├── ja/                    # 日本語ノートブック
│   └── quickstart_ja.md
├── en/                    # English notebooks  
│   └── quickstart_en.md
├── zh/                    # 中文笔记本
│   └── quickstart_zh.md
├── es/                    # Español notebooks
│   └── quickstart_es.md
├── fr/                    # Français notebooks
│   └── quickstart_fr.md
├── de/                    # Deutsch notebooks
│   └── quickstart_de.md
├── ko/                    # 한국어 notebooks
│   └── quickstart_ko.md
├── ru/                    # Русский notebooks
│   └── quickstart_ru.md
└── pt/                    # Português notebooks
    └── quickstart_pt.md
```

## 🚀 Getting Started

### 1. Prerequisites

```bash
# Install Jupyter and Jupytext
pip install jupyter jupyterlab jupytext

# Install RusTorch
pip install maturin
# Then build RusTorch (see quickstart guides)
```

### 2. Open in JupyterLab

```bash
# Navigate to notebooks directory
cd notebooks

# Launch JupyterLab
jupyter lab

# Or open specific language
jupyter lab ja/quickstart_ja.md    # Japanese
jupyter lab en/quickstart_en.md    # English  
jupyter lab zh/quickstart_zh.md    # Chinese
jupyter lab es/quickstart_es.md    # Spanish
jupyter lab fr/quickstart_fr.md    # French
jupyter lab de/quickstart_de.md    # German
jupyter lab ko/quickstart_ko.md    # Korean
jupyter lab ru/quickstart_ru.md    # Russian
jupyter lab pt/quickstart_pt.md    # Portuguese
```

### 3. Enable Executable Notebooks

```bash
# Convert .md to executable notebook format (examples)
jupytext --set-formats ipynb,md:myst ja/quickstart_ja.md
jupytext --set-formats ipynb,md:myst en/quickstart_en.md
jupytext --set-formats ipynb,md:myst zh/quickstart_zh.md
jupytext --set-formats ipynb,md:myst es/quickstart_es.md
jupytext --set-formats ipynb,md:myst fr/quickstart_fr.md
jupytext --set-formats ipynb,md:myst de/quickstart_de.md
jupytext --set-formats ipynb,md:myst ko/quickstart_ko.md
jupytext --set-formats ipynb,md:myst ru/quickstart_ru.md
jupytext --set-formats ipynb,md:myst pt/quickstart_pt.md
```

## 🌍 Language Selection

Choose your preferred language:

- **🇯🇵 日本語**: [`ja/quickstart_ja.md`](ja/quickstart_ja.md) - 日本語でRusTorchを学習
- **🇺🇸 English**: [`en/quickstart_en.md`](en/quickstart_en.md) - Learn RusTorch in English
- **🇨🇳 中文**: [`zh/quickstart_zh.md`](zh/quickstart_zh.md) - 用中文学习RusTorch
- **🇪🇸 Español**: [`es/quickstart_es.md`](es/quickstart_es.md) - Aprende RusTorch en Español
- **🇫🇷 Français**: [`fr/quickstart_fr.md`](fr/quickstart_fr.md) - Apprenez RusTorch en Français
- **🇩🇪 Deutsch**: [`de/quickstart_de.md`](de/quickstart_de.md) - Lernen Sie RusTorch auf Deutsch
- **🇰🇷 한국어**: [`ko/quickstart_ko.md`](ko/quickstart_ko.md) - 한국어로 RusTorch 배우기
- **🇷🇺 Русский**: [`ru/quickstart_ru.md`](ru/quickstart_ru.md) - Изучайте RusTorch на русском языке
- **🇧🇷 Português**: [`pt/quickstart_pt.md`](pt/quickstart_pt.md) - Aprenda RusTorch em Português

## 💡 Usage Tips

### JupyterLab Features
- **Markdown Preview**: Real-time preview of documentation
- **Code Execution**: Run Python code blocks interactively  
- **Split View**: View docs and code side-by-side
- **Export Options**: Save as .ipynb, .pdf, or .html

### Jupytext Integration
- **Bi-directional Sync**: Changes sync between .md and .ipynb
- **Version Control**: Track changes in readable .md format
- **Collaboration**: Share notebooks as markdown files

### Best Practices
1. **Start Fresh**: Always start from the quickstart guide
2. **Sequential Execution**: Run code blocks in order
3. **Environment Setup**: Ensure Python virtual environment is activated
4. **Dependencies**: Install all required packages before starting

## 🔧 Troubleshooting

### Common Issues

1. **RusTorch Import Error**
   ```bash
   # Ensure RusTorch is built and installed
   cd .. && maturin develop --release
   ```

2. **Jupyter Not Found**
   ```bash
   # Install Jupyter in your environment
   pip install jupyter jupyterlab
   ```

3. **Kernel Issues**
   ```bash
   # Reset Jupyter kernel
   jupyter kernelspec list
   jupyter kernelspec remove python3
   python -m ipykernel install --user
   ```

## 📚 Additional Resources

- [RusTorch Documentation](../docs/)
- [Python API Reference](../docs/en/python_api_reference.md)
- [GitHub Repository](https://github.com/JunSuzukiJapan/rustorch)
- [crates.io Package](https://crates.io/crates/rustorch)

## 🤝 Contributing

Contributions to improve these notebooks are welcome! Please:

1. Follow existing format and structure
2. Maintain multilingual consistency
3. Test all code examples before submitting
4. Update documentation accordingly

---

Happy learning with RusTorch! 🚀