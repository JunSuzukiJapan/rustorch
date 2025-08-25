#!/usr/bin/env node
/**
 * Basic test for RusTorch WASM examples
 */

import fs from 'fs';
import path from 'path';

console.log('🧪 Running RusTorch WASM tests...');

// Test 1: Check if WASM files exist
const pkgDir = './pkg';
const wasmFile = path.join(pkgDir, 'rustorch_bg.wasm');
const jsFile = path.join(pkgDir, 'rustorch.js');

if (!fs.existsSync(wasmFile)) {
    console.error('❌ WASM file not found:', wasmFile);
    console.log('💡 Run "npm run build-wasm" first');
    process.exit(1);
}

if (!fs.existsSync(jsFile)) {
    console.error('❌ JS binding file not found:', jsFile);
    console.log('💡 Run "npm run build-wasm" first');
    process.exit(1);
}

console.log('✅ WASM files found');
console.log('✅ JS binding files found');

// Test 2: Check file sizes
const wasmStats = fs.statSync(wasmFile);
const jsStats = fs.statSync(jsFile);

console.log(`📊 WASM file size: ${(wasmStats.size / 1024).toFixed(2)} KB`);
console.log(`📊 JS file size: ${(jsStats.size / 1024).toFixed(2)} KB`);

if (wasmStats.size < 1000) {
    console.warn('⚠️  WASM file seems too small, build might be incomplete');
    process.exit(1);
}

console.log('✅ All tests passed!');
console.log('🎉 RusTorch WASM examples are ready to run');