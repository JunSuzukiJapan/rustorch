#!/bin/bash
# Quick script to run rustorch-cli during development

set -e

# Default to no features for faster builds
FEATURES=""
BUILD_MODE="--release"
BACKEND_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_MODE=""
            shift
            ;;
        --metal)
            FEATURES="metal"
            BACKEND_ARG="--backend metal"
            shift
            ;;
        --coreml)
            FEATURES="coreml"
            BACKEND_ARG="--backend hybrid"
            shift
            ;;
        --mac-hybrid)
            FEATURES="mac-hybrid"
            BACKEND_ARG="--backend hybrid"
            shift
            ;;
        --hybrid-f32)
            FEATURES="hybrid-f32"
            BACKEND_ARG="--backend hybrid-f32"
            shift
            ;;
        --cuda)
            FEATURES="cuda"
            BACKEND_ARG="--backend cuda"
            shift
            ;;
        --help)
            echo "Usage: ./run-cli.sh [OPTIONS] [CLI_ARGS...]"
            echo ""
            echo "Options:"
            echo "  --debug        Build in debug mode (faster compilation)"
            echo "  --metal        Enable Metal GPU backend"
            echo "  --coreml       Enable CoreML backend"
            echo "  --mac-hybrid   Enable Mac hybrid backend (Metal + CoreML)"
            echo "  --hybrid-f32   Enable f32 hybrid mode"
            echo "  --cuda         Enable CUDA backend"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run-cli.sh                           # CPU backend, release mode"
            echo "  ./run-cli.sh --debug                   # CPU backend, debug mode (fast build)"
            echo "  ./run-cli.sh --metal                   # Metal backend"
            echo "  ./run-cli.sh --metal -- chat           # Metal backend, run chat command"
            echo "  ./run-cli.sh --mac-hybrid -- download  # Mac hybrid, run download command"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# Change to project root if running from example-cli directory
if [ -f "Cargo.toml" ] && grep -q "rustorch-cli" "Cargo.toml"; then
    cd ..
fi

# Build command
CMD="cargo run -p rustorch-cli"

if [ -n "$FEATURES" ]; then
    CMD="$CMD --features $FEATURES"
fi

if [ -n "$BUILD_MODE" ]; then
    CMD="$CMD $BUILD_MODE"
fi

# Add backend argument if set
if [ -n "$BACKEND_ARG" ]; then
    if [ $# -gt 0 ]; then
        CMD="$CMD -- $BACKEND_ARG $@"
    else
        CMD="$CMD -- $BACKEND_ARG"
    fi
elif [ $# -gt 0 ]; then
    CMD="$CMD -- $@"
fi

echo "Running: $CMD"
echo ""

eval $CMD
