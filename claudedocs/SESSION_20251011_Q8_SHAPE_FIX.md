# Q8_0 Shape Reversal Fix Session - 2025-10-11

## 🎯 Root Cause Identified

**Problem**: Q8_0 (and all quantized formats) produce gibberish output because **tensor shapes are not reversed** when loading from GGUF files.

### Discovery Process

1. **Layer-by-layer analysis** revealed exponential value growth:
   - Embedding: RMS = 0.008
   - Layer 0: RMS = 0.014 (1.7x)
   - Layer 21: RMS = 1.062 (127x!)

2. **Weight shape inspection** showed the issue:
   ```
   GGUF original_dims: [2048, 32000]
   Final shape: [2048, 32000]  ← Should be [32000, 2048]!
   ```

3. **Code inspection** found TWO loading paths:
   - `from_gguf_with_device()`: Lines 177-252 - Only reverses F32/F16
   - `from_gguf_with_config()`: Lines 1363-1443 - Conditional reversal (square matrices + embeddings)

### Why This Breaks Everything

**GGUF Layout**: `[inner_dim, outer_dim]` - C-order (row-major)
**PyTorch/RusTorch**: `[outer_dim, inner_dim]` - Fortran-order (column-major)

**Example: token_embd.weight**
- GGUF stores: `[2048, 32000]` (2048 features × 32000 tokens)
- Should be: `[32000, 2048]` (32000 tokens × 2048 features)
- Matmul: `x[tokens, 32000] @ weight[32000, 2048] = result[tokens, 2048]` ✓
- But with wrong shape: `x[tokens, 32000] @ weight[2048, 32000]` → Dimension mismatch or wrong result!

When shapes don't match perfectly, the matmul still executes but **operates on wrong dimensions**, producing garbage output.

## 🔧 Fixes Applied

### Fix 1: `from_gguf_with_device()` (Lines 220-225)

**Before**:
```rust
let shape: Vec<usize> = match ggml_type {
    Some(GGMLType::F32) | Some(GGMLType::F16) => {
        let mut s = original_dims.clone();
        s.reverse();
        s
    }
    _ => original_dims.clone(),  // Q8_0, Q4_K not reversed!
};
```

**After**:
```rust
// IMPORTANT: ALL GGUF tensors need shape reversal
let mut shape = original_dims.clone();
shape.reverse();
```

### Fix 2: `from_gguf_with_config()` (Lines 1395-1421) - **STILL NEEDS FIX**

**Current (INCOMPLETE)**:
```rust
let is_square = original_dims.len() == 2 && original_dims[0] == original_dims[1];
let is_embedding_or_output = name.contains("output.weight") || name.contains("token_embd.weight");
let needs_reversal = is_square || is_embedding_or_output;
```

**Issue**: Only reverses:
- Square matrices (2048×2048)
- Embeddings and output weights
- **MISSING**: All other quantized weights!

**Required Fix**:
```rust
// ALL GGUF tensors need shape reversal
let mut shape = original_dims.clone();
shape.reverse();
```

## ⚠️ Current Status

- ✅ `from_gguf_with_device()` fixed (Lines 220-225)
- ❌ `from_gguf_with_config()` still incomplete (Lines 1395-1421)
- ❌ example-cli uses `from_gguf_with_config()`, so fix not active yet
- ❌ Q8_0 still produces gibberish output

## 📋 Next Steps

1. **Fix `from_gguf_with_config()`**: Apply same shape reversal logic
2. **Remove conditional logic**: All tensors need reversal, not just square/embedding
3. **Test Q8_0**: Verify output matches llama.cpp
4. **Test other formats**: Q6_K, Q5_K_M, Q4_K_M (should already work)
5. **Clean up debug output**: Remove temporary debug print statements

## 📁 Files Modified

- `src/hybrid_f32/models/llama.rs`:
  - Lines 220-231: Fixed `from_gguf_with_device()` ✅
  - Lines 1395-1421: Need to fix `from_gguf_with_config()` ❌

## 🧪 Testing Commands

```bash
# Build
cargo build --release --features hybrid-f32,metal

# Test Q8_0
printf "1\n" | ./target/release/rustorch-cli \
  --model ~/.rustorch/models/.../tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --backend hybrid-f32 --max-tokens 10

# Expected output: "Hello\nGreetings! It was pleasure chat"
# Current output: "anthanthertanthert..." ❌
```

## 📊 Evidence

**Before Fix**:
```
Layer 21: RMS=1.062, Max=4.26
Output: "anthanthertanthert"
```

**After Fix 1 (incomplete)**:
```
Layer 21: RMS=1.058, Max=4.08
Output: "anthanthertanthert" (still wrong)
```

**Why still wrong**: example-cli uses `from_gguf_with_config()` which still has incomplete shape reversal logic.

## 💡 Key Insight

**ALL GGUF tensor dimensions must be reversed**, regardless of:
- Quantization type (F32, F16, Q8_0, Q4_K, etc.)
- Shape (square, non-square)
- Weight type (embedding, attention, FFN, etc.)

This is because GGUF uses C-order (row-major) while PyTorch/RusTorch uses Fortran-order (column-major).

The previous conditional logic was a workaround that happened to work for some cases but breaks for quantized formats.
