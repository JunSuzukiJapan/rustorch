#!/bin/bash

# RusTorch Benchmark Runner
# Usage: ./run_benchmarks.sh [all|cpu|metal|coreml] [matrix|convolution|transformer|all]

set -e

BACKEND=${1:-all}
BENCHMARK=${2:-all}
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="performance_records"

echo "🚀 RusTorch Benchmark Runner"
echo "============================="
echo "Backend: $BACKEND"
echo "Benchmark: $BENCHMARK"
echo "Timestamp: $TIMESTAMP"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Function to run a single benchmark
run_benchmark() {
    local backend=$1
    local benchmark_type=$2
    local features=$3

    echo "🔧 Running $backend backend with $benchmark_type benchmark..."

    if [ -n "$features" ]; then
        cargo run --example simple_performance_demo --features "$features" --release -- --backend "$backend" --benchmark "$benchmark_type"
    else
        cargo run --example simple_performance_demo --release -- --backend "$backend" --benchmark "$benchmark_type"
    fi

    echo ""
}

# Run benchmarks based on input
case $BACKEND in
    "cpu")
        run_benchmark "cpu" "$BENCHMARK" ""
        ;;
    "metal")
        run_benchmark "metal" "$BENCHMARK" "metal"
        ;;
    "coreml")
        run_benchmark "coreml" "$BENCHMARK" "coreml"
        ;;
    "all")
        echo "🔄 Running all backends..."
        echo ""

        echo "1/3: CPU Backend"
        run_benchmark "cpu" "$BENCHMARK" ""

        echo "2/3: Metal GPU Backend"
        run_benchmark "metal" "$BENCHMARK" "metal"

        echo "3/3: CoreML Neural Engine Backend"
        run_benchmark "coreml" "$BENCHMARK" "coreml"
        ;;
    *)
        echo "❌ Unknown backend: $BACKEND"
        echo "Available backends: cpu, metal, coreml, all"
        exit 1
        ;;
esac

echo "✅ Benchmark completed!"
echo "📊 Results saved to: $RESULTS_DIR/"
echo ""
echo "💡 To record these results permanently:"
echo "   Create a new file: $RESULTS_DIR/${TIMESTAMP}_${BACKEND}_${BENCHMARK}_results.md"