# 🎯 BREAKTHROUGH: Q4_K Quantization Bug Identified

**Date**: 2025-10-09 Evening
**Status**: ROOT CAUSE ISOLATED
**Severity**: CRITICAL
**Impact**: Q4_K_M models produce gibberish, all other quantizations work correctly

## Discovery

### Cross-Quantization Testing

Tested Metal backend with 4 different quantization levels using input "1":

| Quantization | Top Token | Logit | Status |
|-------------|-----------|-------|--------|
| Q8_0        | 24155     | 8.10  | ✅ Correct |
| Q6_K        | 24155     | 7.97  | ✅ Correct |
| Q5_K_M      | 24155     | 8.26  | ✅ Correct |
| **Q4_K_M**  | **3499**  | **8.47** | ❌ **Gibberish** |

**llama.cpp reference**: All quantizations produce correct output (" 10")

### Key Evidence

1. **Q8_0, Q6_K, Q5_K_M work perfectly** with Metal backend
2. **Only Q4_K_M fails** - produces completely different token
3. Same hardware, same code path, same Metal kernels
4. llama.cpp handles Q4_K_M correctly

### Root Cause

**Bug is in Q4_K dequantization implementation**, NOT in:
- ❌ Metal kernels
- ❌ RMS Norm
- ❌ Attention mechanism
- ❌ RoPE
- ❌ Embeddings

**Location**: `src/formats/gguf.rs` - Q4_K dequantization function

## Technical Details

### Q4_K Format Structure

From `src/formats/gguf.rs`:
```rust
// Q4_K: Super-blocks of 256 elements (8 blocks of 32 elements each)
// Each super-block contains:
// - scales: f16[8] - scale factors for each block
// - mins: u8[8] - min values for each block
// - quants: u8[128] - 4-bit quantized values (2 values per byte)
```

### Dequantization Process

1. Read super-block header (scales + mins)
2. For each of 8 blocks (32 elements each):
   - Extract 4-bit quantized values
   - Apply scale: `value = scale * quant + min`
   - Convert to f32

### Comparison with Working Quantizations

**Q8_0** (works):
- Simple structure: scale + 8-bit values
- No super-blocks
- Straightforward dequantization

**Q6_K** (works):
- Similar super-block structure to Q4_K
- 6-bit quantization
- More precision than Q4_K

**Q4_K_M** (broken):
- Complex super-block structure
- 4-bit quantization (lowest precision)
- Most susceptible to dequantization errors

## Hypothesis

**Most Likely Bugs**:

1. **Incorrect bit extraction** from 4-bit packed format
   - Each byte contains 2x 4-bit values
   - Wrong nibble extraction (high/low bits swapped?)
   - Off-by-one in indexing

2. **Scale/min application error**
   - Wrong scale selected for block
   - Incorrect min value application
   - f16 → f32 conversion issue

3. **Block boundary error**
   - Super-block of 256 not properly divided into 8 blocks of 32
   - Index calculation error

## Next Steps (URGENT)

### 1. Compare with llama.cpp Implementation

**File to review**: `Temp/llama.cpp/ggml/src/ggml-quants.c`

Look for `dequantize_row_q4_K`:
```c
void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int k) {
    // Reference implementation
}
```

### 2. Add Debug Logging to Q4_K Dequantization

Instrument `src/formats/gguf.rs` to log:
- Super-block header values (scales, mins)
- First 10 dequantized values
- Compare with Q8_0 for same weight tensor

### 3. Create Unit Test

```rust
// Test Q4_K dequantization with known values
// Compare output with llama.cpp reference
```

### 4. Verify Bit Manipulation

```rust
// Current implementation (suspected bug area):
for i in 0..32 {
    let quant_byte = quants[i / 2];
    let quant = if i % 2 == 0 {
        quant_byte & 0x0F  // Low nibble
    } else {
        (quant_byte >> 4) & 0x0F  // High nibble
    };
    // ... apply scale and min
}
```

Check:
- Is nibble order correct?
- Should it be `(i % 2 == 0)` or `(i % 2 == 1)`?
- Is index `i / 2` correct?

## Impact Assessment

**Affected**:
- All Q4_K_M models (most common quantization for efficiency)
- Metal GPU backend only (CPU path may differ)

**Not Affected**:
- Q8_0, Q6_K, Q5_K_M (confirmed working)
- llama.cpp (reference implementation correct)

**User Impact**:
- Users with Q4_K_M models get gibberish output
- Workaround: Use Q5_K_M or Q6_K models

## Estimated Fix Time

- **Investigation**: 1-2 hours (compare with llama.cpp, add logging)
- **Fix**: 30 minutes - 2 hours (depending on bug complexity)
- **Testing**: 30 minutes (verify all quantizations still work)
- **Total**: 2-5 hours

## Files to Modify

1. `src/formats/gguf.rs` - Fix Q4_K dequantization
2. `tests/` - Add Q4_K unit tests
3. `METAL_GPU_DEBUGGING_STATUS.md` - Update with resolution

## Success Criteria

1. Q4_K_M model produces Token 24155 (like Q8_0/Q6_K/Q5_K_M)
2. Output matches llama.cpp for same input
3. All existing quantizations continue to work
4. Unit test passes for Q4_K dequantization
