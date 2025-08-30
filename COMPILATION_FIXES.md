# RusTorch Compilation Fixes Plan

## Issues Identified:

1. **DeviceType Pattern Matching Issues**: 
   - `DeviceType::OpenCL(_)` treated as tuple variant instead of unit variant
   - Same issues with `DeviceType::Cuda(_)` and `DeviceType::Metal(_)`
   - Root cause: Two conflicting DeviceType enums

2. **Missing Trait Methods**:
   - `execute` method missing from `UnifiedKernelExecutor` trait
   - Only has `execute_f32`, `execute_f64` but implementation uses generic `execute`

3. **DeviceManager Constructor Issues**:
   - Some calls use `DeviceManager::new()` without arguments
   - Others use `DeviceManager::new(SelectionStrategy)`
   - Need to align signatures

4. **Missing Error Variants**:
   - `validation` and `profiling` error variants exist in enum but may be missing constructors

5. **Move/Borrow Issues**:
   - Need proper cloning in validation and profiler modules

## Fix Strategy:
1. Fix DeviceType enum conflicts in backends vs GPU modules
2. Add missing trait methods to UnifiedKernelExecutor
3. Standardize DeviceManager constructor calls
4. Add missing error variant constructors
5. Fix move/borrow issues with proper cloning

## Files to Fix:
- src/backends/mod.rs (DeviceType enum)
- src/backends/compute_backend.rs (DeviceType enum)
- src/gpu/unified_kernel.rs (trait method)
- All files with DeviceType pattern matching
- DeviceManager::new() calls
- Validation and profiler modules