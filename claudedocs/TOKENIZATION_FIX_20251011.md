# Tokenization Fix - October 11, 2025

## Problem

RusTorch was producing gibberish output for ALL quantization formats (Q4_K, Q5_K, Q6_K, Q8_0).

## Root Cause Discovered

**SentencePiece (SPM) tokenizer was missing the automatic space prefix** that llama.cpp adds when:
1. Model has `add_space_prefix=true` (TinyLlama does)
2. Previous token is a special token (BOS)

### Tokenization Comparison

**Before Fix:**
```
Input: "<|user|>\n1<|assistant|>"
RusTorch: [1, 29966, 29989, 1792, 29989, 29958, 13, 29896, 29966, 29989, 465, 22137, 29989, 29958]
llama.cpp: [1, 529, 29989, 1792, 29989, 29958, 13, 29896, 29966, 29989, 465, 22137, 29989, 29958]
                ^^ WRONG!
Token 1 mismatch:
- RusTorch: 29966 = '<' (no space)
- llama.cpp: 529 = ' <' (with space)
```

**After Fix:**
```
Both produce: [1, 529, 29989, 1792, 29989, 29958, 13, 29896, 29966, 29989, 465, 22137, 29989, 29958]
✅ Tokenization now matches exactly!
```

## Fix Applied

**File**: `/Users/junsuzuki/Program/Rust/RusTorch/rustorch/example-cli/src/tokenizer/llama_spm.rs`
**Location**: Lines 290-299

```rust
// CRITICAL FIX: llama.cpp SPM tokenizer adds a space prefix when:
// 1. Model has add_space_prefix=true (TinyLlama does)
// 2. Previous token is a special token (BOS)
// This matches llama.cpp's tokenizer behavior for SentencePiece models
let text_to_encode = if add_special_tokens && !text.is_empty() {
    eprintln!("🔍 [LLAMA_SPM] Adding space prefix (SPM behavior after BOS)");
    format!(" {}", text)
} else {
    text.to_string()
};

let text_tokens = self.tokenize(&text_to_encode);
```

## Verification

**Token Decoding:**
```
Token 529  = ' <'  (2 bytes: 0x20 0x3c - space + less-than)
Token 29966 = '<'   (1 byte: 0x3c - just less-than)
```

**Test Results:**
```bash
# Before: Wrong tokenization
RusTorch tokens: [1, 29966, ...]  # Missing space prefix

# After: Correct tokenization
RusTorch tokens: [1, 529, ...]    # Space prefix added ✓
```

## Impact

This fix ensures RusTorch's tokenizer produces **identical token sequences** to llama.cpp, which is critical for:
1. **Model compatibility**: Models trained with space-prefix behavior
2. **Output correctness**: Wrong tokens → wrong embeddings → wrong everything
3. **Cross-implementation consistency**: RusTorch outputs match llama.cpp

## Remaining Issues

**⚠️ OUTPUT STILL GIBBERISH!**

Even with correct tokenization, the model still produces wrong output:
```
Input: "Hello"
Expected: Coherent response
Actual: "anthanthertanthertrun ChallengeniASEörtrinder"
```

**Logit Comparison (with correct tokenization):**
```
Token     RusTorch    llama.cpp    Diff
------    ---------   ----------   ------
0         -3.037      -7.701       4.665
13         3.540      19.808      16.268  ← HUGE DIFF!
```

Top token: RusTorch=9716, llama.cpp=13 (newline) ❌

### Conclusion

✅ **Tokenization is now correct**
❌ **Transformer layers still produce wrong outputs**

The problem is deeper than tokenization - it's in the transformer implementation itself (attention, FFN, or how they process the quantized weights).

## Next Steps

1. ✅ Tokenization fix complete and verified
2. ❌ Need to investigate transformer layer implementation
3. Focus areas:
   - Attention mechanism (Q, K, V projections)
   - Feed-forward network (gate, up, down projections)
   - RMSNorm application
   - Hidden state computation
   - Weight dequantization (already verified as correct)

## References

- **llama.cpp behavior**: SPM tokenizer automatically adds space prefix after BOS
- **File modified**: `example-cli/src/tokenizer/llama_spm.rs:290-299`
- **Commit**: Tokenizer space prefix fix
