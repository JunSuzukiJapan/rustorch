#!/bin/bash

# RusTorch WebAssembly Build Script
# This script builds the WebAssembly version of RusTorch for web and Node.js targets

set -e  # Exit on any error

echo "🦀 Building RusTorch for WebAssembly..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed. Please install it first:"
    echo "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/
rm -rf pkg-node/
rm -rf examples/pkg/
rm -rf examples/pkg-node/

# Build for web (ES6 modules)
echo "🌐 Building for web browsers..."
wasm-pack build --target web --out-dir pkg --release

if [ $? -eq 0 ]; then
    echo "✅ Web build completed successfully"
else
    echo "❌ Web build failed"
    exit 1
fi

# Build for Node.js
echo "📦 Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg-node --release

if [ $? -eq 0 ]; then
    echo "✅ Node.js build completed successfully"
else
    echo "❌ Node.js build failed"
    exit 1
fi

# Copy builds to examples directory
echo "📁 Copying builds to examples directory..."
cp -r pkg/ examples/
cp -r pkg-node/ examples/

# Verify builds
echo "🔍 Verifying builds..."

# Check web build
if [ -f "pkg/rustorch.js" ] && [ -f "pkg/rustorch_bg.wasm" ]; then
    echo "✅ Web build files verified"
else
    echo "❌ Web build files missing"
    exit 1
fi

# Check Node.js build
if [ -f "pkg-node/rustorch.js" ] && [ -f "pkg-node/rustorch_bg.wasm" ]; then
    echo "✅ Node.js build files verified"
else
    echo "❌ Node.js build files missing"
    exit 1
fi

# Display build information
echo ""
echo "📊 Build Information:"
echo "====================="

# Web build size
WEB_WASM_SIZE=$(ls -la pkg/rustorch_bg.wasm | awk '{print $5}')
WEB_JS_SIZE=$(ls -la pkg/rustorch.js | awk '{print $5}')

echo "Web build:"
echo "  - WASM file: $(echo $WEB_WASM_SIZE | numfmt --to=iec-i --suffix=B)"
echo "  - JS file:   $(echo $WEB_JS_SIZE | numfmt --to=iec-i --suffix=B)"

# Node.js build size
NODE_WASM_SIZE=$(ls -la pkg-node/rustorch_bg.wasm | awk '{print $5}')
NODE_JS_SIZE=$(ls -la pkg-node/rustorch.js | awk '{print $5}')

echo "Node.js build:"
echo "  - WASM file: $(echo $NODE_WASM_SIZE | numfmt --to=iec-i --suffix=B)"
echo "  - JS file:   $(echo $NODE_JS_SIZE | numfmt --to=iec-i --suffix=B)"

echo ""
echo "🎉 WebAssembly build completed successfully!"
echo ""
echo "Next steps:"
echo "1. For web browsers:"
echo "   cd examples && python -m http.server 8000"
echo "   Open http://localhost:8000/wasm_basic.html"
echo ""
echo "2. For Node.js:"
echo "   cd examples && npm install && npm run demo"
echo ""
echo "3. Run performance tests:"
echo "   cd examples && node wasm_performance_test.js"
echo ""
echo "📚 See examples/README.md for more information"