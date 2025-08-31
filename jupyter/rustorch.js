let wasm;

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_export_2.set(idx, obj);
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_export_2.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}
/**
 * Initialize WASM module
 */
export function init_wasm() {
    wasm.init_wasm();
}

/**
 * Create tensor from Float32Array
 * @param {Float32Array} data
 * @param {Array<any>} shape
 * @returns {WasmTensor}
 */
export function tensor_from_float32_array(data, shape) {
    const ret = wasm.tensor_from_float32_array(data, shape);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmTensor.__wrap(ret[0]);
}

/**
 * Convert tensor to Float32Array
 * @param {WasmTensor} tensor
 * @returns {Float32Array}
 */
export function tensor_to_float32_array(tensor) {
    _assertClass(tensor, WasmTensor);
    const ret = wasm.tensor_to_float32_array(tensor.__wbg_ptr);
    return ret;
}

/**
 * Create tensor from nested JavaScript array
 * @param {any} array
 * @returns {WasmTensor}
 */
export function tensor_from_nested_array(array) {
    const ret = wasm.tensor_from_nested_array(array);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmTensor.__wrap(ret[0]);
}

/**
 * Convert tensor to nested JavaScript array
 * @param {WasmTensor} tensor
 * @returns {Array<any>}
 */
export function tensor_to_nested_array(tensor) {
    _assertClass(tensor, WasmTensor);
    const ret = wasm.tensor_to_nested_array(tensor.__wbg_ptr);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Memory-efficient tensor slicing
 * @param {WasmTensor} tensor
 * @param {number} start
 * @param {number} end
 * @returns {WasmTensor}
 */
export function tensor_slice(tensor, start, end) {
    _assertClass(tensor, WasmTensor);
    const ret = wasm.tensor_slice(tensor.__wbg_ptr, start, end);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmTensor.__wrap(ret[0]);
}

/**
 * Simple benchmark for tensor operations
 * @param {number} size
 * @param {number} iterations
 * @returns {BenchmarkResult}
 */
export function benchmark_matmul(size, iterations) {
    const ret = wasm.benchmark_matmul(size, iterations);
    return BenchmarkResult.__wrap(ret);
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
 * Initialize WASM runtime
 */
export function initialize_wasm_runtime() {
    wasm.initialize_wasm_runtime();
}

/**
 * Detect WASM runtime features
 * @returns {object}
 */
export function detect_wasm_features() {
    const ret = wasm.detect_wasm_features();
    return ret;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
 * Version information
 * @returns {string}
 */
export function wasm_advanced_math_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.wasm_advanced_math_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Version information
 * @returns {string}
 */
export function wasm_anomaly_detection_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.wasm_anomaly_detection_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Create a simple anomaly detector for web applications
 * @param {number} threshold
 * @returns {WasmAnomalyDetector}
 */
export function create_simple_detector(threshold) {
    const ret = wasm.create_simple_detector(threshold);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmAnomalyDetector.__wrap(ret[0]);
}

/**
 * Create a time series detector for streaming data
 * @param {number} window_size
 * @returns {WasmTimeSeriesDetector}
 */
export function create_streaming_detector(window_size) {
    const ret = wasm.create_streaming_detector(window_size);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmTimeSeriesDetector.__wrap(ret[0]);
}

/**
 * Batch anomaly detection for arrays
 * @param {Float32Array} data
 * @param {number} threshold
 * @returns {Array<any>}
 */
export function detect_anomalies_batch(data, threshold) {
    const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_anomalies_batch(ptr0, len0, threshold);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * Version information
 * @returns {string}
 */
export function wasm_transforms_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.wasm_transforms_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Create ImageNet preprocessing pipeline
 * @returns {WasmNormalize}
 */
export function create_imagenet_preprocessing() {
    const ret = wasm.create_imagenet_preprocessing();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmNormalize.__wrap(ret[0]);
}

/**
 * Create CIFAR preprocessing pipeline
 * @returns {WasmNormalize}
 */
export function create_cifar_preprocessing() {
    const ret = wasm.create_cifar_preprocessing();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmNormalize.__wrap(ret[0]);
}

/**
 * Version information
 * @returns {string}
 */
export function wasm_quality_metrics_version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.wasm_quality_metrics_version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Create quality analyzer with default threshold
 * @returns {WasmQualityMetrics}
 */
export function create_quality_analyzer() {
    const ret = wasm.create_quality_analyzer();
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return WasmQualityMetrics.__wrap(ret[0]);
}

/**
 * Quick quality assessment for web applications
 * @param {WasmTensor} tensor
 * @returns {string}
 */
export function quick_quality_assessment(tensor) {
    let deferred2_0;
    let deferred2_1;
    try {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.quick_quality_assessment(tensor.__wbg_ptr);
        var ptr1 = ret[0];
        var len1 = ret[1];
        if (ret[3]) {
            ptr1 = 0; len1 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred2_0 = ptr1;
        deferred2_1 = len1;
        return getStringFromWasm0(ptr1, len1);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * @param {number} x
 * @returns {number}
 */
export function gamma_wasm(x) {
    const ret = wasm.gamma_wasm(x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function lgamma_wasm(x) {
    const ret = wasm.lgamma_wasm(x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function digamma_wasm(x) {
    const ret = wasm.digamma_wasm(x);
    return ret;
}

/**
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function beta_wasm(a, b) {
    const ret = wasm.beta_wasm(a, b);
    return ret;
}

/**
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
export function lbeta_wasm(a, b) {
    const ret = wasm.lbeta_wasm(a, b);
    return ret;
}

/**
 * @param {number} n
 * @param {number} x
 * @returns {number}
 */
export function bessel_j_wasm(n, x) {
    const ret = wasm.bessel_j_wasm(n, x);
    return ret;
}

/**
 * @param {number} n
 * @param {number} x
 * @returns {number}
 */
export function bessel_y_wasm(n, x) {
    const ret = wasm.bessel_y_wasm(n, x);
    return ret;
}

/**
 * @param {number} n
 * @param {number} x
 * @returns {number}
 */
export function bessel_i_wasm(n, x) {
    const ret = wasm.bessel_i_wasm(n, x);
    return ret;
}

/**
 * @param {number} n
 * @param {number} x
 * @returns {number}
 */
export function bessel_k_wasm(n, x) {
    const ret = wasm.bessel_k_wasm(n, x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function erf_wasm(x) {
    const ret = wasm.erf_wasm(x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function erfc_wasm(x) {
    const ret = wasm.erfc_wasm(x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function erfinv_wasm(x) {
    const ret = wasm.erfinv_wasm(x);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function erfcinv_wasm(x) {
    const ret = wasm.erfcinv_wasm(x);
    return ret;
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}
/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function gamma_array_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.gamma_array_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {number} n
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function bessel_j_array_wasm(n, values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bessel_j_array_wasm(n, ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function erf_array_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.erf_array_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {number} n
 * @returns {number}
 */
export function factorial_wasm(n) {
    const ret = wasm.factorial_wasm(n);
    return ret;
}

/**
 * @param {number} n
 * @returns {number}
 */
export function log_factorial_wasm(n) {
    const ret = wasm.log_factorial_wasm(n);
    return ret;
}

/**
 * @returns {number}
 */
export function euler_gamma() {
    const ret = wasm.euler_gamma();
    return ret;
}

/**
 * @returns {number}
 */
export function sqrt_2pi() {
    const ret = wasm.sqrt_2pi();
    return ret;
}

/**
 * @returns {number}
 */
export function log_sqrt_2pi() {
    const ret = wasm.log_sqrt_2pi();
    return ret;
}

/**
 * @param {number} x
 * @param {number} mean
 * @param {number} std
 * @returns {number}
 */
export function normal_cdf_wasm(x, mean, std) {
    const ret = wasm.normal_cdf_wasm(x, mean, std);
    return ret;
}

/**
 * @param {number} p
 * @param {number} mean
 * @param {number} std
 * @returns {number}
 */
export function normal_quantile_wasm(p, mean, std) {
    const ret = wasm.normal_quantile_wasm(p, mean, std);
    return ret;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function quick_stats_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.quick_stats_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {number} iterations
 * @returns {Float64Array}
 */
export function benchmark_special_functions_wasm(iterations) {
    const ret = wasm.benchmark_special_functions_wasm(iterations);
    var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v1;
}

/**
 * @param {number} initial_lr
 * @param {bigint} step
 * @param {number} decay_rate
 * @param {bigint} decay_steps
 * @returns {number}
 */
export function learning_rate_schedule_wasm(initial_lr, step, decay_rate, decay_steps) {
    const ret = wasm.learning_rate_schedule_wasm(initial_lr, step, decay_rate, decay_steps);
    return ret;
}

/**
 * @param {number} initial_lr
 * @param {bigint} current_step
 * @param {bigint} total_steps
 * @returns {number}
 */
export function cosine_annealing_wasm(initial_lr, current_step, total_steps) {
    const ret = wasm.cosine_annealing_wasm(initial_lr, current_step, total_steps);
    return ret;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function relu_wasm(x) {
    const ret = wasm.relu_wasm(x);
    return ret;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function relu_array_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.relu_array_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function sigmoid_wasm(x) {
    const ret = wasm.sigmoid_wasm(x);
    return ret;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function sigmoid_array_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.sigmoid_array_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {number} x
 * @returns {number}
 */
export function tanh_wasm(x) {
    const ret = wasm.tanh_wasm(x);
    return ret;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function tanh_array_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.tanh_array_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * @param {Float64Array} values
 * @returns {Float64Array}
 */
export function softmax_wasm(values) {
    const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.softmax_wasm(ptr0, len0);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

const AdaGradWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_adagradwasm_free(ptr >>> 0, 1));

export class AdaGradWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AdaGradWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_adagradwasm_free(ptr, 0);
    }
    /**
     * @param {number} learning_rate
     * @param {number} epsilon
     * @param {number} weight_decay
     */
    constructor(learning_rate, epsilon, weight_decay) {
        const ret = wasm.adagradwasm_new(learning_rate, epsilon, weight_decay);
        this.__wbg_ptr = ret >>> 0;
        AdaGradWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} param_name
     * @param {Float64Array} params
     * @param {Float64Array} gradients
     */
    step(param_name, params, gradients) {
        const ptr0 = passStringToWasm0(param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = passArrayF64ToWasm0(params, wasm.__wbindgen_malloc);
        var len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.adagradwasm_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, params, ptr2, len2);
    }
    /**
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.adagradwasm_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.adagradwasm_set_learning_rate(this.__wbg_ptr, lr);
    }
    reset_state() {
        wasm.adagradwasm_reset_state(this.__wbg_ptr);
    }
}

const AdamWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_adamwasm_free(ptr >>> 0, 1));

export class AdamWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AdamWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_adamwasm_free(ptr, 0);
    }
    /**
     * @param {number} learning_rate
     * @param {number} beta1
     * @param {number} beta2
     * @param {number} epsilon
     * @param {number} weight_decay
     */
    constructor(learning_rate, beta1, beta2, epsilon, weight_decay) {
        const ret = wasm.adamwasm_new(learning_rate, beta1, beta2, epsilon, weight_decay);
        this.__wbg_ptr = ret >>> 0;
        AdamWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} param_name
     * @param {Float64Array} params
     * @param {Float64Array} gradients
     */
    step(param_name, params, gradients) {
        const ptr0 = passStringToWasm0(param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = passArrayF64ToWasm0(params, wasm.__wbindgen_malloc);
        var len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.adamwasm_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, params, ptr2, len2);
    }
    /**
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.adamwasm_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.adamwasm_set_learning_rate(this.__wbg_ptr, lr);
    }
    /**
     * @returns {bigint}
     */
    get_step_count() {
        const ret = wasm.adamwasm_get_step_count(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    reset_state() {
        wasm.adamwasm_reset_state(this.__wbg_ptr);
    }
}

const BenchmarkResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_benchmarkresult_free(ptr >>> 0, 1));
/**
 * Performance benchmarking utility
 */
export class BenchmarkResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(BenchmarkResult.prototype);
        obj.__wbg_ptr = ptr;
        BenchmarkResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BenchmarkResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_benchmarkresult_free(ptr, 0);
    }
    /**
     * Get operation name
     * @returns {string}
     */
    get operation() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.benchmarkresult_operation(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get duration in milliseconds
     * @returns {number}
     */
    get duration_ms() {
        const ret = wasm.benchmarkresult_duration_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get throughput (operations per second)
     * @returns {number}
     */
    get throughput() {
        const ret = wasm.benchmarkresult_throughput(this.__wbg_ptr);
        return ret;
    }
}

const BernoulliDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bernoullidistributionwasm_free(ptr >>> 0, 1));

export class BernoulliDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BernoulliDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bernoullidistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} p
     */
    constructor(p) {
        const ret = wasm.exponentialdistributionwasm_new(p);
        this.__wbg_ptr = ret >>> 0;
        BernoulliDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {boolean}
     */
    sample() {
        const ret = wasm.bernoullidistributionwasm_sample(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {number} n
     * @returns {Uint8Array}
     */
    sample_array(n) {
        const ret = wasm.bernoullidistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {number}
     */
    sample_f64() {
        const ret = wasm.bernoullidistributionwasm_sample_f64(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_f64_array(n) {
        const ret = wasm.bernoullidistributionwasm_sample_f64_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {boolean} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.bernoullidistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.benchmarkresult_duration_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.bernoullidistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
}

const BetaDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_betadistributionwasm_free(ptr >>> 0, 1));

export class BetaDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BetaDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_betadistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} alpha
     * @param {number} beta
     */
    constructor(alpha, beta) {
        const ret = wasm.normaldistributionwasm_new(alpha, beta);
        this.__wbg_ptr = ret >>> 0;
        BetaDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    sample() {
        const ret = wasm.betadistributionwasm_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_array(n) {
        const ret = wasm.betadistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.betadistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.betadistributionwasm_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.betadistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
}

const BrowserStorageFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_browserstorage_free(ptr >>> 0, 1));
/**
 * Browser storage utilities
 */
export class BrowserStorage {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BrowserStorageFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_browserstorage_free(ptr, 0);
    }
    /**
     * Create new browser storage utility
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        BrowserStorageFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Save tensor to localStorage
     * @param {string} key
     * @param {WasmTensor} tensor
     */
    save_tensor(key, tensor) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(tensor, WasmTensor);
        const ret = wasm.browserstorage_save_tensor(this.__wbg_ptr, ptr0, len0, tensor.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Load tensor from localStorage
     * @param {string} key
     * @returns {WasmTensor}
     */
    load_tensor(key) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.browserstorage_load_tensor(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * List all saved tensor keys
     * @returns {Array<any>}
     */
    list_tensor_keys() {
        const ret = wasm.browserstorage_list_tensor_keys(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Clear all saved tensors
     */
    clear_tensors() {
        const ret = wasm.browserstorage_clear_tensors(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}

const CanvasRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_canvasrenderer_free(ptr >>> 0, 1));
/**
 * Canvas utilities for tensor visualization
 */
export class CanvasRenderer {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CanvasRendererFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_canvasrenderer_free(ptr, 0);
    }
    /**
     * Create new canvas renderer for the specified canvas element
     * @param {string} canvas_id
     */
    constructor(canvas_id) {
        const ptr0 = passStringToWasm0(canvas_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.canvasrenderer_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        CanvasRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Render 2D tensor as heatmap
     * @param {WasmTensor} tensor
     */
    render_heatmap(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.canvasrenderer_render_heatmap(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Clear canvas
     */
    clear() {
        wasm.canvasrenderer_clear(this.__wbg_ptr);
    }
}

const ComputationGraphWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_computationgraphwasm_free(ptr >>> 0, 1));

export class ComputationGraphWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ComputationGraphWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_computationgraphwasm_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.computationgraphwasm_new();
        this.__wbg_ptr = ret >>> 0;
        ComputationGraphWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float64Array} data
     * @param {Uint32Array} shape
     * @param {boolean} requires_grad
     * @returns {string}
     */
    create_variable(data, shape, requires_grad) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
            const len1 = WASM_VECTOR_LEN;
            const ret = wasm.computationgraphwasm_create_variable(this.__wbg_ptr, ptr0, len0, ptr1, len1, requires_grad);
            deferred3_0 = ret[0];
            deferred3_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * @param {string} id
     * @returns {Float64Array | undefined}
     */
    get_variable_data(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.computationgraphwasm_get_variable_data(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        }
        return v2;
    }
    /**
     * @param {string} id
     * @returns {Float64Array | undefined}
     */
    get_variable_grad(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.computationgraphwasm_get_variable_grad(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        }
        return v2;
    }
    /**
     * @param {string} id1
     * @param {string} id2
     * @returns {string | undefined}
     */
    add_variables(id1, id2) {
        const ptr0 = passStringToWasm0(id1, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(id2, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.computationgraphwasm_add_variables(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        let v3;
        if (ret[0] !== 0) {
            v3 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v3;
    }
    /**
     * @param {string} id1
     * @param {string} id2
     * @returns {string | undefined}
     */
    mul_variables(id1, id2) {
        const ptr0 = passStringToWasm0(id1, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(id2, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.computationgraphwasm_mul_variables(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        let v3;
        if (ret[0] !== 0) {
            v3 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v3;
    }
    /**
     * @param {string} id
     */
    backward(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.computationgraphwasm_backward(this.__wbg_ptr, ptr0, len0);
    }
    zero_grad_all() {
        wasm.computationgraphwasm_zero_grad_all(this.__wbg_ptr);
    }
    clear_graph() {
        wasm.computationgraphwasm_clear_graph(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    variable_count() {
        const ret = wasm.computationgraphwasm_variable_count(this.__wbg_ptr);
        return ret >>> 0;
    }
}

const ExponentialDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_exponentialdistributionwasm_free(ptr >>> 0, 1));

export class ExponentialDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ExponentialDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_exponentialdistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} rate
     */
    constructor(rate) {
        const ret = wasm.exponentialdistributionwasm_new(rate);
        this.__wbg_ptr = ret >>> 0;
        ExponentialDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    sample() {
        const ret = wasm.exponentialdistributionwasm_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_array(n) {
        const ret = wasm.exponentialdistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.exponentialdistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.exponentialdistributionwasm_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.exponentialdistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
}

const FileLoaderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fileloader_free(ptr >>> 0, 1));
/**
 * File API utilities for loading tensors from files
 */
export class FileLoader {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FileLoaderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fileloader_free(ptr, 0);
    }
    /**
     * Create new file loader utility
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        FileLoaderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create file input element for tensor loading
     * @returns {HTMLInputElement}
     */
    create_file_input() {
        const ret = wasm.fileloader_create_file_input(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const GammaDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_gammadistributionwasm_free(ptr >>> 0, 1));

export class GammaDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GammaDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_gammadistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} shape
     * @param {number} scale
     */
    constructor(shape, scale) {
        const ret = wasm.normaldistributionwasm_new(shape, scale);
        this.__wbg_ptr = ret >>> 0;
        GammaDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    sample() {
        const ret = wasm.gammadistributionwasm_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_array(n) {
        const ret = wasm.gammadistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.gammadistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.gammadistributionwasm_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.gammadistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
}

const JsInteropFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsinterop_free(ptr >>> 0, 1));
/**
 * JavaScript interop utilities
 */
export class JsInterop {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsInteropFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsinterop_free(ptr, 0);
    }
    /**
     * Create new JavaScript interop utility
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        JsInteropFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create tensor filled with ones
     * @param {Array<any>} shape
     * @returns {WasmTensor}
     */
    ones(shape) {
        const ret = wasm.jsinterop_ones(this.__wbg_ptr, shape);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Create tensor filled with zeros
     * @param {Array<any>} shape
     * @returns {WasmTensor}
     */
    zeros(shape) {
        const ret = wasm.jsinterop_zeros(this.__wbg_ptr, shape);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Create random tensor
     * @param {Array<any>} shape
     * @param {number} min
     * @param {number} max
     * @returns {WasmTensor}
     */
    random_tensor(shape, min, max) {
        const ret = wasm.jsinterop_random_tensor(this.__wbg_ptr, shape, min, max);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Log tensor information to console
     * @param {WasmTensor} tensor
     * @param {string} name
     */
    log_tensor(tensor, name) {
        _assertClass(tensor, WasmTensor);
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.jsinterop_log_tensor(this.__wbg_ptr, tensor.__wbg_ptr, ptr0, len0);
    }
}

const LinearLayerWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_linearlayerwasm_free(ptr >>> 0, 1));

export class LinearLayerWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LinearLayerWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_linearlayerwasm_free(ptr, 0);
    }
    /**
     * @param {number} input_size
     * @param {number} output_size
     */
    constructor(input_size, output_size) {
        const ret = wasm.linearlayerwasm_new(input_size, output_size);
        this.__wbg_ptr = ret >>> 0;
        LinearLayerWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float64Array} input
     * @returns {Float64Array | undefined}
     */
    forward(input) {
        const ptr0 = passArrayF64ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.linearlayerwasm_forward(this.__wbg_ptr, ptr0, len0);
        let v2;
        if (ret[0] !== 0) {
            v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        }
        return v2;
    }
    /**
     * @returns {Float64Array}
     */
    get_weights() {
        const ret = wasm.linearlayerwasm_get_weights(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    get_bias() {
        const ret = wasm.linearlayerwasm_get_bias(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {Float64Array} new_weights
     * @returns {boolean}
     */
    update_weights(new_weights) {
        const ptr0 = passArrayF64ToWasm0(new_weights, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.linearlayerwasm_update_weights(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * @param {Float64Array} new_bias
     * @returns {boolean}
     */
    update_bias(new_bias) {
        const ptr0 = passArrayF64ToWasm0(new_bias, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.linearlayerwasm_update_bias(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
}

const NormalDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_normaldistributionwasm_free(ptr >>> 0, 1));

export class NormalDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NormalDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_normaldistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} mean
     * @param {number} std
     */
    constructor(mean, std) {
        const ret = wasm.normaldistributionwasm_new(mean, std);
        this.__wbg_ptr = ret >>> 0;
        NormalDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    sample() {
        const ret = wasm.normaldistributionwasm_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_array(n) {
        const ret = wasm.normaldistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.normaldistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @param {Float64Array} values
     * @returns {Float64Array}
     */
    log_prob_array(values) {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.normaldistributionwasm_log_prob_array(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.benchmarkresult_duration_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.normaldistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    std_dev() {
        const ret = wasm.benchmarkresult_throughput(this.__wbg_ptr);
        return ret;
    }
}

const OptimizedOpsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_optimizedops_free(ptr >>> 0, 1));
/**
 * Optimized tensor operations for WASM
 */
export class OptimizedOps {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        OptimizedOpsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_optimizedops_free(ptr, 0);
    }
    /**
     * Create new optimized operations utility
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        OptimizedOpsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Fast matrix multiplication using blocking for cache efficiency
     * @param {WasmTensor} a
     * @param {WasmTensor} b
     * @returns {WasmTensor}
     */
    fast_matmul(a, b) {
        _assertClass(a, WasmTensor);
        _assertClass(b, WasmTensor);
        const ret = wasm.optimizedops_fast_matmul(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Vectorized element-wise operations
     * @param {WasmTensor} a
     * @param {WasmTensor} b
     * @returns {WasmTensor}
     */
    vectorized_add(a, b) {
        _assertClass(a, WasmTensor);
        _assertClass(b, WasmTensor);
        const ret = wasm.optimizedops_vectorized_add(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Fast ReLU with fused operations
     * @param {WasmTensor} input
     * @param {WasmTensor} bias
     * @returns {WasmTensor}
     */
    fused_relu_add(input, bias) {
        _assertClass(input, WasmTensor);
        _assertClass(bias, WasmTensor);
        const ret = wasm.optimizedops_fused_relu_add(this.__wbg_ptr, input.__wbg_ptr, bias.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Memory-efficient convolution-like operation (simplified 1D)
     * @param {WasmTensor} input
     * @param {WasmTensor} kernel
     * @param {number} stride
     * @returns {WasmTensor}
     */
    conv1d(input, kernel, stride) {
        _assertClass(input, WasmTensor);
        _assertClass(kernel, WasmTensor);
        const ret = wasm.optimizedops_conv1d(this.__wbg_ptr, input.__wbg_ptr, kernel.__wbg_ptr, stride);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Batch normalization-like operation
     * @param {WasmTensor} input
     * @param {number} epsilon
     * @returns {WasmTensor}
     */
    batch_normalize(input, epsilon) {
        _assertClass(input, WasmTensor);
        const ret = wasm.optimizedops_batch_normalize(this.__wbg_ptr, input.__wbg_ptr, epsilon);
        return WasmTensor.__wrap(ret);
    }
}

const ParallelOpsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_parallelops_free(ptr >>> 0, 1));
/**
 * Parallel execution utilities (simulated for WASM single-threaded environment)
 */
export class ParallelOps {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ParallelOpsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_parallelops_free(ptr, 0);
    }
    /**
     * Parallel-style reduction (sequential in WASM)
     * @param {Float32Array} data
     * @returns {number}
     */
    static parallel_sum(data) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.parallelops_parallel_sum(ptr0, len0);
        return ret;
    }
    /**
     * Parallel-style element-wise operation
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
    static parallel_map_add(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.parallelops_parallel_map_add(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
}

const PerformanceMonitorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_performancemonitor_free(ptr >>> 0, 1));
/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PerformanceMonitorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_performancemonitor_free(ptr, 0);
    }
    /**
     * Get memory usage information
     * @returns {object}
     */
    static get_memory_info() {
        const ret = wasm.performancemonitor_get_memory_info();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Measure function execution time
     * @param {string} name
     */
    static time_function(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.performancemonitor_time_function(ptr0, len0);
    }
    /**
     * End timing measurement
     * @param {string} name
     */
    static time_end(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.performancemonitor_time_end(ptr0, len0);
    }
}

const RMSpropWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_rmspropwasm_free(ptr >>> 0, 1));

export class RMSpropWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RMSpropWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_rmspropwasm_free(ptr, 0);
    }
    /**
     * @param {number} learning_rate
     * @param {number} alpha
     * @param {number} epsilon
     * @param {number} weight_decay
     * @param {number} momentum
     */
    constructor(learning_rate, alpha, epsilon, weight_decay, momentum) {
        const ret = wasm.rmspropwasm_new(learning_rate, alpha, epsilon, weight_decay, momentum);
        this.__wbg_ptr = ret >>> 0;
        RMSpropWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} param_name
     * @param {Float64Array} params
     * @param {Float64Array} gradients
     */
    step(param_name, params, gradients) {
        const ptr0 = passStringToWasm0(param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = passArrayF64ToWasm0(params, wasm.__wbindgen_malloc);
        var len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.rmspropwasm_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, params, ptr2, len2);
    }
    /**
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.adamwasm_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.adamwasm_set_learning_rate(this.__wbg_ptr, lr);
    }
    reset_state() {
        wasm.rmspropwasm_reset_state(this.__wbg_ptr);
    }
}

const SGDWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sgdwasm_free(ptr >>> 0, 1));

export class SGDWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SGDWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sgdwasm_free(ptr, 0);
    }
    /**
     * @param {number} learning_rate
     * @param {number} momentum
     * @param {number} dampening
     * @param {number} weight_decay
     * @param {boolean} nesterov
     */
    constructor(learning_rate, momentum, dampening, weight_decay, nesterov) {
        const ret = wasm.sgdwasm_new(learning_rate, momentum, dampening, weight_decay, nesterov);
        this.__wbg_ptr = ret >>> 0;
        SGDWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {string} param_name
     * @param {Float64Array} params
     * @param {Float64Array} gradients
     */
    step(param_name, params, gradients) {
        const ptr0 = passStringToWasm0(param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = passArrayF64ToWasm0(params, wasm.__wbindgen_malloc);
        var len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.sgdwasm_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, params, ptr2, len2);
    }
    /**
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.adagradwasm_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.adagradwasm_set_learning_rate(this.__wbg_ptr, lr);
    }
    reset_state() {
        wasm.sgdwasm_reset_state(this.__wbg_ptr);
    }
}

const SpecialFunctionsBatchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_specialfunctionsbatch_free(ptr >>> 0, 1));

export class SpecialFunctionsBatch {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SpecialFunctionsBatchFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_specialfunctionsbatch_free(ptr, 0);
    }
    /**
     * @param {number} cache_size
     */
    constructor(cache_size) {
        const ret = wasm.wasmrng_new(cache_size);
        this.__wbg_ptr = ret >>> 0;
        SpecialFunctionsBatchFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float64Array} values
     * @returns {Float64Array}
     */
    gamma_batch(values) {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.specialfunctionsbatch_gamma_batch(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * @param {Float64Array} values
     * @returns {Float64Array}
     */
    bessel_j0_batch(values) {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.specialfunctionsbatch_bessel_j0_batch(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
    /**
     * @param {Float64Array} values
     * @returns {Float64Array}
     */
    erf_batch(values) {
        const ptr0 = passArrayF64ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.specialfunctionsbatch_erf_batch(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v2;
    }
}

const UniformDistributionWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_uniformdistributionwasm_free(ptr >>> 0, 1));

export class UniformDistributionWasm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UniformDistributionWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_uniformdistributionwasm_free(ptr, 0);
    }
    /**
     * @param {number} low
     * @param {number} high
     */
    constructor(low, high) {
        const ret = wasm.normaldistributionwasm_new(low, high);
        this.__wbg_ptr = ret >>> 0;
        UniformDistributionWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    sample() {
        const ret = wasm.uniformdistributionwasm_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     * @returns {Float64Array}
     */
    sample_array(n) {
        const ret = wasm.uniformdistributionwasm_sample_array(this.__wbg_ptr, n);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} x
     * @returns {number}
     */
    log_prob(x) {
        const ret = wasm.uniformdistributionwasm_log_prob(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * @returns {number}
     */
    mean() {
        const ret = wasm.uniformdistributionwasm_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    variance() {
        const ret = wasm.uniformdistributionwasm_variance(this.__wbg_ptr);
        return ret;
    }
}

const VariableWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_variablewasm_free(ptr >>> 0, 1));

export class VariableWasm {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(VariableWasm.prototype);
        obj.__wbg_ptr = ptr;
        VariableWasmFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VariableWasmFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_variablewasm_free(ptr, 0);
    }
    /**
     * @param {Float64Array} data
     * @param {Uint32Array} shape
     * @param {boolean} requires_grad
     */
    constructor(data, shape, requires_grad) {
        const ptr0 = passArrayF64ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.variablewasm_new(ptr0, len0, ptr1, len1, requires_grad);
        this.__wbg_ptr = ret >>> 0;
        VariableWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Float64Array}
     */
    data() {
        const ret = wasm.variablewasm_data(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Uint32Array}
     */
    shape() {
        const ret = wasm.variablewasm_shape(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float64Array | undefined}
     */
    grad() {
        const ret = wasm.variablewasm_grad(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        }
        return v1;
    }
    /**
     * @returns {boolean}
     */
    requires_grad() {
        const ret = wasm.variablewasm_requires_grad(this.__wbg_ptr);
        return ret !== 0;
    }
    zero_grad() {
        wasm.variablewasm_zero_grad(this.__wbg_ptr);
    }
    backward() {
        wasm.variablewasm_backward(this.__wbg_ptr);
    }
    /**
     * @returns {VariableWasm}
     */
    sum() {
        const ret = wasm.variablewasm_sum(this.__wbg_ptr);
        return VariableWasm.__wrap(ret);
    }
    /**
     * @returns {VariableWasm}
     */
    mean() {
        const ret = wasm.variablewasm_mean(this.__wbg_ptr);
        return VariableWasm.__wrap(ret);
    }
    /**
     * @param {number} exponent
     * @returns {VariableWasm}
     */
    pow(exponent) {
        const ret = wasm.variablewasm_pow(this.__wbg_ptr, exponent);
        return VariableWasm.__wrap(ret);
    }
}

const WasmActivationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmactivation_free(ptr >>> 0, 1));
/**
 * WASM-compatible activation functions
 * WASM互換活性化関数
 */
export class WasmActivation {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmActivationFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmactivation_free(ptr, 0);
    }
    /**
     * ReLU (Rectified Linear Unit) activation function
     * ReLU(x) = max(0, x)
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static relu(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_relu(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * ReLU derivative for backward pass
     * ReLUの微分（逆伝播用）
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static relu_derivative(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_relu_derivative(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Leaky ReLU activation function
     * Leaky ReLU(x) = max(alpha * x, x)
     * @param {Float32Array} input
     * @param {number} alpha
     * @returns {Float32Array}
     */
    static leaky_relu(input, alpha) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_leaky_relu(ptr0, len0, alpha);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Leaky ReLU derivative
     * @param {Float32Array} input
     * @param {number} alpha
     * @returns {Float32Array}
     */
    static leaky_relu_derivative(input, alpha) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_leaky_relu_derivative(ptr0, len0, alpha);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Sigmoid activation function
     * Sigmoid(x) = 1 / (1 + exp(-x))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static sigmoid(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_sigmoid(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Sigmoid derivative
     * σ'(x) = σ(x) * (1 - σ(x))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static sigmoid_derivative(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_sigmoid_derivative(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Tanh (Hyperbolic Tangent) activation function
     * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static tanh(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_tanh(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Tanh derivative
     * tanh'(x) = 1 - tanh²(x)
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static tanh_derivative(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_tanh_derivative(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Softmax activation function
     * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static softmax(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_softmax(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Log Softmax activation function (numerically stable)
     * LogSoftmax(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static log_softmax(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_log_softmax(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * GELU (Gaussian Error Linear Unit) activation function
     * GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
     * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static gelu(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_gelu(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * GELU derivative (approximate)
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static gelu_derivative(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_gelu_derivative(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Swish/SiLU activation function
     * Swish(x) = x * sigmoid(x)
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static swish(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_swish(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Mish activation function
     * Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static mish(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_mish(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * ELU (Exponential Linear Unit) activation function
     * ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
     * @param {Float32Array} input
     * @param {number} alpha
     * @returns {Float32Array}
     */
    static elu(input, alpha) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_elu(ptr0, len0, alpha);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * ELU derivative
     * @param {Float32Array} input
     * @param {number} alpha
     * @returns {Float32Array}
     */
    static elu_derivative(input, alpha) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_elu_derivative(ptr0, len0, alpha);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Softplus activation function
     * Softplus(x) = ln(1 + exp(x))
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static softplus(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_softplus(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Softsign activation function
     * Softsign(x) = x / (1 + |x|)
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    static softsign(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_softsign(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply activation function to 2D data (batch processing)
     * 2Dデータに活性化関数を適用（バッチ処理）
     * @param {Float32Array} input
     * @param {number} rows
     * @param {number} cols
     * @returns {Float32Array}
     */
    static relu_2d(input, rows, cols) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_relu_2d(ptr0, len0, rows, cols);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply softmax along specified axis for 2D data
     * 2Dデータの指定軸に沿ってソフトマックスを適用
     * @param {Float32Array} input
     * @param {number} rows
     * @param {number} cols
     * @param {number} axis
     * @returns {Float32Array}
     */
    static softmax_2d(input, rows, cols, axis) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_softmax_2d(ptr0, len0, rows, cols, axis);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Combined activation function selector
     * 活性化関数セレクター
     * @param {Float32Array} input
     * @param {string} activation_type
     * @returns {Float32Array}
     */
    static apply_activation(input, activation_type) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(activation_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmactivation_apply_activation(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
}

const WasmAdaGradFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmadagrad_free(ptr >>> 0, 1));
/**
 * AdaGrad optimizer for WASM (simpler than Adam)
 * WASM用AdaGradオプティマイザ（Adamより簡単）
 */
export class WasmAdaGrad {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmAdaGrad.prototype);
        obj.__wbg_ptr = ptr;
        WasmAdaGradFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAdaGradFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmadagrad_free(ptr, 0);
    }
    /**
     * Create new AdaGrad optimizer
     * @param {number} learning_rate
     * @param {number} epsilon
     */
    constructor(learning_rate, epsilon) {
        const ret = wasm.wasmadagrad_new(learning_rate, epsilon);
        this.__wbg_ptr = ret >>> 0;
        WasmAdaGradFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Update parameters with AdaGrad algorithm
     * @param {string} param_id
     * @param {Float32Array} parameters
     * @param {Float32Array} gradients
     * @returns {Float32Array}
     */
    step(param_id, parameters, gradients) {
        const ptr0 = passStringToWasm0(param_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(parameters, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmadagrad_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
}

const WasmAdamFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmadam_free(ptr >>> 0, 1));
/**
 * Adam optimizer for WASM
 * WASM用Adamオプティマイザ
 */
export class WasmAdam {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmAdam.prototype);
        obj.__wbg_ptr = ptr;
        WasmAdamFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAdamFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmadam_free(ptr, 0);
    }
    /**
     * Create new Adam optimizer
     * @param {number} learning_rate
     */
    constructor(learning_rate) {
        const ret = wasm.wasmadam_new(learning_rate);
        this.__wbg_ptr = ret >>> 0;
        WasmAdamFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create Adam with custom parameters
     * @param {number} learning_rate
     * @param {number} beta1
     * @param {number} beta2
     * @param {number} epsilon
     * @param {number} weight_decay
     * @returns {WasmAdam}
     */
    static with_params(learning_rate, beta1, beta2, epsilon, weight_decay) {
        const ret = wasm.wasmadam_with_params(learning_rate, beta1, beta2, epsilon, weight_decay);
        return WasmAdam.__wrap(ret);
    }
    /**
     * Update parameters with gradients using Adam algorithm
     * @param {string} param_id
     * @param {Float32Array} parameters
     * @param {Float32Array} gradients
     * @returns {Float32Array}
     */
    step(param_id, parameters, gradients) {
        const ptr0 = passStringToWasm0(param_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(parameters, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmadam_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * Get learning rate
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.wasmadam_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set learning rate
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.wasmadam_set_learning_rate(this.__wbg_ptr, lr);
    }
    /**
     * Get step count
     * @returns {number}
     */
    get_step_count() {
        const ret = wasm.wasmadam_get_step_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reset optimizer state
     */
    reset() {
        wasm.wasmadam_reset(this.__wbg_ptr);
    }
}

const WasmAdvancedMathFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmadvancedmath_free(ptr >>> 0, 1));
/**
 * WASM wrapper for advanced mathematical operations
 */
export class WasmAdvancedMath {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAdvancedMathFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmadvancedmath_free(ptr, 0);
    }
    /**
     * Create new advanced math instance
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        WasmAdvancedMathFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Hyperbolic sine
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    sinh(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_sinh(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Hyperbolic cosine
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    cosh(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_cosh(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Hyperbolic tangent
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    tanh(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_tanh(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Inverse sine (arcsine)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    asin(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_asin(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Inverse cosine (arccosine)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    acos(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_acos(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Inverse tangent (arctangent)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    atan(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_atan(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Two-argument arctangent
     * @param {WasmTensor} y
     * @param {WasmTensor} x
     * @returns {WasmTensor}
     */
    atan2(y, x) {
        _assertClass(y, WasmTensor);
        _assertClass(x, WasmTensor);
        const ret = wasm.wasmadvancedmath_atan2(this.__wbg_ptr, y.__wbg_ptr, x.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Error function (approximate)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    erf(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_erf(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Complementary error function
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    erfc(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_erfc(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Gamma function (approximate)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    gamma(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_gamma(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Log gamma function
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    lgamma(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_lgamma(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Clamp values between min and max
     * @param {WasmTensor} tensor
     * @param {number} min_val
     * @param {number} max_val
     * @returns {WasmTensor}
     */
    clamp(tensor, min_val, max_val) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_clamp(this.__wbg_ptr, tensor.__wbg_ptr, min_val, max_val);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Sign function
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    sign(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_sign(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Linear interpolation between two tensors
     * @param {WasmTensor} start
     * @param {WasmTensor} end
     * @param {number} weight
     * @returns {WasmTensor}
     */
    lerp(start, end, weight) {
        _assertClass(start, WasmTensor);
        _assertClass(end, WasmTensor);
        const ret = wasm.wasmadvancedmath_lerp(this.__wbg_ptr, start.__wbg_ptr, end.__wbg_ptr, weight);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Power function with scalar exponent
     * @param {WasmTensor} base
     * @param {number} exponent
     * @returns {WasmTensor}
     */
    pow(base, exponent) {
        _assertClass(base, WasmTensor);
        const ret = wasm.wasmadvancedmath_pow(this.__wbg_ptr, base.__wbg_ptr, exponent);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Element-wise power
     * @param {WasmTensor} base
     * @param {WasmTensor} exponent
     * @returns {WasmTensor}
     */
    pow_tensor(base, exponent) {
        _assertClass(base, WasmTensor);
        _assertClass(exponent, WasmTensor);
        const ret = wasm.wasmadvancedmath_pow_tensor(this.__wbg_ptr, base.__wbg_ptr, exponent.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Round to nearest integer
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    round(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_round(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Floor function
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    floor(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_floor(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Ceiling function
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    ceil(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_ceil(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Truncate to integer
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    trunc(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_trunc(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Check if values are finite
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    is_finite(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_is_finite(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Check if values are infinite
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    is_infinite(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_is_infinite(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Check if values are NaN
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    is_nan(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmadvancedmath_is_nan(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
}

const WasmAnomalyDetectorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmanomalydetector_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Anomaly Detector
 */
export class WasmAnomalyDetector {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmAnomalyDetector.prototype);
        obj.__wbg_ptr = ptr;
        WasmAnomalyDetectorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAnomalyDetectorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmanomalydetector_free(ptr, 0);
    }
    /**
     * Create new anomaly detector
     * @param {number} threshold
     * @param {number} window_size
     */
    constructor(threshold, window_size) {
        const ret = wasm.wasmanomalydetector_new(threshold, window_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmAnomalyDetectorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Detect anomalies using statistical method
     * @param {WasmTensor} data
     * @returns {Array<any>}
     */
    detect_statistical(data) {
        _assertClass(data, WasmTensor);
        const ret = wasm.wasmanomalydetector_detect_statistical(this.__wbg_ptr, data.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Detect anomalies using isolation forest method (simplified)
     * @param {WasmTensor} data
     * @param {number} _n_trees
     * @returns {Array<any>}
     */
    detect_isolation_forest(data, _n_trees) {
        _assertClass(data, WasmTensor);
        const ret = wasm.wasmanomalydetector_detect_isolation_forest(this.__wbg_ptr, data.__wbg_ptr, _n_trees);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Real-time anomaly detection for streaming data
     * @param {number} value
     * @returns {any}
     */
    detect_realtime(value) {
        const ret = wasm.wasmanomalydetector_detect_realtime(this.__wbg_ptr, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get detector statistics
     * @returns {string}
     */
    get_statistics() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmanomalydetector_get_statistics(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Reset detector state
     */
    reset() {
        wasm.wasmanomalydetector_reset(this.__wbg_ptr);
    }
    /**
     * Update threshold
     * @param {number} threshold
     */
    set_threshold(threshold) {
        const ret = wasm.wasmanomalydetector_set_threshold(this.__wbg_ptr, threshold);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get current threshold
     * @returns {number}
     */
    get_threshold() {
        const ret = wasm.wasmanomalydetector_get_threshold(this.__wbg_ptr);
        return ret;
    }
}

const WasmBatchNormFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmbatchnorm_free(ptr >>> 0, 1));
/**
 * Batch Normalization layer for WASM
 * WASM用のバッチ正規化レイヤー
 */
export class WasmBatchNorm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmBatchNormFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmbatchnorm_free(ptr, 0);
    }
    /**
     * Create a new Batch Normalization layer
     * 新しいバッチ正規化レイヤーを作成
     * @param {number} num_features
     * @param {number} momentum
     * @param {number} epsilon
     */
    constructor(num_features, momentum, epsilon) {
        const ret = wasm.wasmbatchnorm_new(num_features, momentum, epsilon);
        this.__wbg_ptr = ret >>> 0;
        WasmBatchNormFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set training mode
     * 訓練モードを設定
     * @param {boolean} training
     */
    set_training(training) {
        wasm.wasmbatchnorm_set_training(this.__wbg_ptr, training);
    }
    /**
     * Set scale (gamma) parameters
     * スケール（ガンマ）パラメータを設定
     * @param {Float32Array} gamma
     */
    set_gamma(gamma) {
        const ptr0 = passArrayF32ToWasm0(gamma, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmbatchnorm_set_gamma(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set shift (beta) parameters
     * シフト（ベータ）パラメータを設定
     * @param {Float32Array} beta
     */
    set_beta(beta) {
        const ptr0 = passArrayF32ToWasm0(beta, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmbatchnorm_set_beta(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Forward pass through batch normalization
     * バッチ正規化の順伝播
     * @param {Float32Array} input
     * @param {number} batch_size
     * @returns {Float32Array}
     */
    forward(input, batch_size) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmbatchnorm_forward(this.__wbg_ptr, ptr0, len0, batch_size);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get running mean for inspection
     * 実行中の平均値を取得（検査用）
     * @returns {Float32Array}
     */
    get_running_mean() {
        const ret = wasm.wasmbatchnorm_get_running_mean(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get running variance for inspection
     * 実行中の分散値を取得（検査用）
     * @returns {Float32Array}
     */
    get_running_var() {
        const ret = wasm.wasmbatchnorm_get_running_var(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}

const WasmBernoulliFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmbernoulli_free(ptr >>> 0, 1));
/**
 * Bernoulli distribution for WASM
 * WASM用ベルヌーイ分布
 */
export class WasmBernoulli {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmBernoulliFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmbernoulli_free(ptr, 0);
    }
    /**
     * Create new Bernoulli distribution
     * @param {number} p
     * @param {number} seed
     */
    constructor(p, seed) {
        const ret = wasm.wasmbernoulli_new(p, seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmBernoulliFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Sample single value (0 or 1)
     * @returns {number}
     */
    sample() {
        const ret = wasm.wasmbernoulli_sample(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Sample multiple values
     * @param {number} n
     * @returns {Uint32Array}
     */
    sample_n(n) {
        const ret = wasm.wasmbernoulli_sample_n(this.__wbg_ptr, n);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Probability mass function
     * @param {number} x
     * @returns {number}
     */
    pmf(x) {
        const ret = wasm.wasmbernoulli_pmf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Log probability mass function
     * @param {number} x
     * @returns {number}
     */
    log_pmf(x) {
        const ret = wasm.wasmbernoulli_log_pmf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Get mean
     * @returns {number}
     */
    mean() {
        const ret = wasm.wasmbernoulli_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get variance
     * @returns {number}
     */
    variance() {
        const ret = wasm.wasmbernoulli_variance(this.__wbg_ptr);
        return ret;
    }
}

const WasmCenterCropFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcentercrop_free(ptr >>> 0, 1));
/**
 * WASM wrapper for CenterCrop transformation
 */
export class WasmCenterCrop {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCenterCropFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcentercrop_free(ptr, 0);
    }
    /**
     * Create new center crop transform
     * @param {number} height
     * @param {number} width
     */
    constructor(height, width) {
        const ret = wasm.wasmcentercrop_new(height, width);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmCenterCropFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply center crop to tensor
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmcentercrop_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmcentercrop_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmColorJitterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcolorjitter_free(ptr >>> 0, 1));
/**
 * WASM wrapper for ColorJitter transformation
 */
export class WasmColorJitter {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmColorJitterFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcolorjitter_free(ptr, 0);
    }
    /**
     * Create new color jitter transform
     * @param {number} brightness
     * @param {number} contrast
     * @param {number} saturation
     * @param {number} hue
     */
    constructor(brightness, contrast, saturation, hue) {
        const ret = wasm.wasmcolorjitter_new(brightness, contrast, saturation, hue);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmColorJitterFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply color jitter to tensor
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmcolorjitter_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmcolorjitter_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmConv2dFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmconv2d_free(ptr >>> 0, 1));
/**
 * 2D Convolutional layer for WASM
 * WASM用2次元畳み込みレイヤー
 */
export class WasmConv2d {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmConv2dFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmconv2d_free(ptr, 0);
    }
    /**
     * Create new 2D convolutional layer
     * 新しい2次元畳み込みレイヤーを作成
     * @param {number} in_channels
     * @param {number} out_channels
     * @param {number} kernel_size
     * @param {number} stride
     * @param {number} padding
     * @param {boolean} bias
     */
    constructor(in_channels, out_channels, kernel_size, stride, padding, bias) {
        const ret = wasm.wasmconv2d_new(in_channels, out_channels, kernel_size, stride, padding, bias);
        this.__wbg_ptr = ret >>> 0;
        WasmConv2dFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Forward pass through convolution layer
     * 畳み込みレイヤーの順伝播
     * @param {Float32Array} input
     * @param {number} batch_size
     * @param {number} input_height
     * @param {number} input_width
     * @returns {Float32Array}
     */
    forward(input, batch_size, input_height, input_width) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmconv2d_forward(this.__wbg_ptr, ptr0, len0, batch_size, input_height, input_width);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Calculate output dimensions for given input
     * 入力に対する出力次元を計算
     * @param {number} input_height
     * @param {number} input_width
     * @returns {Uint32Array}
     */
    output_shape(input_height, input_width) {
        const ret = wasm.wasmconv2d_output_shape(this.__wbg_ptr, input_height, input_width);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get layer weights
     * @returns {Float32Array}
     */
    get_weights() {
        const ret = wasm.wasmconv2d_get_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get layer bias
     * @returns {Float32Array | undefined}
     */
    get_bias() {
        const ret = wasm.wasmconv2d_get_bias(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * Update weights
     * @param {Float32Array} new_weights
     */
    update_weights(new_weights) {
        const ptr0 = passArrayF32ToWasm0(new_weights, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmconv2d_update_weights(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get layer configuration
     * @returns {object}
     */
    get_config() {
        const ret = wasm.wasmconv2d_get_config(this.__wbg_ptr);
        return ret;
    }
}

const WasmExponentialFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmexponential_free(ptr >>> 0, 1));
/**
 * Exponential distribution for WASM
 * WASM用指数分布
 */
export class WasmExponential {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmExponential.prototype);
        obj.__wbg_ptr = ptr;
        WasmExponentialFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmExponentialFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmexponential_free(ptr, 0);
    }
    /**
     * Create new exponential distribution
     * @param {number} rate
     * @param {number} seed
     */
    constructor(rate, seed) {
        const ret = wasm.wasmexponential_new(rate, seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmExponentialFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create standard exponential distribution (rate=1)
     * @param {number} seed
     * @returns {WasmExponential}
     */
    static standard(seed) {
        const ret = wasm.wasmexponential_standard(seed);
        return WasmExponential.__wrap(ret);
    }
    /**
     * Sample single value using inverse transform sampling
     * @returns {number}
     */
    sample() {
        const ret = wasm.wasmexponential_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * Sample multiple values
     * @param {number} n
     * @returns {Float32Array}
     */
    sample_n(n) {
        const ret = wasm.wasmexponential_sample_n(this.__wbg_ptr, n);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Probability density function
     * @param {number} x
     * @returns {number}
     */
    pdf(x) {
        const ret = wasm.wasmexponential_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Log probability density function
     * @param {number} x
     * @returns {number}
     */
    log_pdf(x) {
        const ret = wasm.wasmexponential_log_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Cumulative distribution function
     * @param {number} x
     * @returns {number}
     */
    cdf(x) {
        const ret = wasm.wasmexponential_cdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Get mean
     * @returns {number}
     */
    mean() {
        const ret = wasm.wasmexponential_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get variance
     * @returns {number}
     */
    variance() {
        const ret = wasm.wasmexponential_variance(this.__wbg_ptr);
        return ret;
    }
}

const WasmGroupNormFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgroupnorm_free(ptr >>> 0, 1));
/**
 * Group Normalization for WASM
 * WASM用のグループ正規化
 */
export class WasmGroupNorm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGroupNormFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgroupnorm_free(ptr, 0);
    }
    /**
     * Create a new Group Normalization layer
     * 新しいグループ正規化レイヤーを作成
     * @param {number} num_groups
     * @param {number} num_channels
     * @param {number} epsilon
     */
    constructor(num_groups, num_channels, epsilon) {
        const ret = wasm.wasmgroupnorm_new(num_groups, num_channels, epsilon);
        this.__wbg_ptr = ret >>> 0;
        WasmGroupNormFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set scale (gamma) parameters
     * スケール（ガンマ）パラメータを設定
     * @param {Float32Array} gamma
     */
    set_gamma(gamma) {
        const ptr0 = passArrayF32ToWasm0(gamma, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmgroupnorm_set_gamma(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set shift (beta) parameters
     * シフト（ベータ）パラメータを設定
     * @param {Float32Array} beta
     */
    set_beta(beta) {
        const ptr0 = passArrayF32ToWasm0(beta, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmgroupnorm_set_beta(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Forward pass through group normalization
     * グループ正規化の順伝播
     * @param {Float32Array} input
     * @param {number} batch_size
     * @param {number} height
     * @param {number} width
     * @returns {Float32Array}
     */
    forward(input, batch_size, height, width) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmgroupnorm_forward(this.__wbg_ptr, ptr0, len0, batch_size, height, width);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}

const WasmLRSchedulerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlrscheduler_free(ptr >>> 0, 1));
/**
 * Learning rate scheduler for WASM optimizers
 * WASMオプティマイザ用学習率スケジューラ
 */
export class WasmLRScheduler {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLRScheduler.prototype);
        obj.__wbg_ptr = ptr;
        WasmLRSchedulerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLRSchedulerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlrscheduler_free(ptr, 0);
    }
    /**
     * Create StepLR scheduler
     * @param {number} initial_lr
     * @param {number} step_size
     * @param {number} gamma
     * @returns {WasmLRScheduler}
     */
    static step_lr(initial_lr, step_size, gamma) {
        const ret = wasm.wasmlrscheduler_step_lr(initial_lr, step_size, gamma);
        return WasmLRScheduler.__wrap(ret);
    }
    /**
     * Create ExponentialLR scheduler
     * @param {number} initial_lr
     * @param {number} gamma
     * @returns {WasmLRScheduler}
     */
    static exponential_lr(initial_lr, gamma) {
        const ret = wasm.wasmlrscheduler_exponential_lr(initial_lr, gamma);
        return WasmLRScheduler.__wrap(ret);
    }
    /**
     * Create CosineAnnealingLR scheduler
     * @param {number} initial_lr
     * @param {number} t_max
     * @param {number} eta_min
     * @returns {WasmLRScheduler}
     */
    static cosine_annealing_lr(initial_lr, t_max, eta_min) {
        const ret = wasm.wasmlrscheduler_cosine_annealing_lr(initial_lr, t_max, eta_min);
        return WasmLRScheduler.__wrap(ret);
    }
    /**
     * Step the scheduler and get updated learning rate
     * @returns {number}
     */
    step() {
        const ret = wasm.wasmlrscheduler_step(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get current learning rate
     * @returns {number}
     */
    get_lr() {
        const ret = wasm.wasmlrscheduler_get_lr(this.__wbg_ptr);
        return ret;
    }
    /**
     * Reset scheduler
     */
    reset() {
        wasm.wasmlrscheduler_reset(this.__wbg_ptr);
    }
}

const WasmLayerNormFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlayernorm_free(ptr >>> 0, 1));
/**
 * Layer Normalization for WASM
 * WASM用のレイヤー正規化
 */
export class WasmLayerNorm {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLayerNormFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlayernorm_free(ptr, 0);
    }
    /**
     * Create a new Layer Normalization layer
     * 新しいレイヤー正規化レイヤーを作成
     * @param {Uint32Array} normalized_shape
     * @param {number} epsilon
     */
    constructor(normalized_shape, epsilon) {
        const ptr0 = passArray32ToWasm0(normalized_shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlayernorm_new(ptr0, len0, epsilon);
        this.__wbg_ptr = ret >>> 0;
        WasmLayerNormFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set scale (gamma) parameters
     * スケール（ガンマ）パラメータを設定
     * @param {Float32Array} gamma
     */
    set_gamma(gamma) {
        const ptr0 = passArrayF32ToWasm0(gamma, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlayernorm_set_gamma(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set shift (beta) parameters
     * シフト（ベータ）パラメータを設定
     * @param {Float32Array} beta
     */
    set_beta(beta) {
        const ptr0 = passArrayF32ToWasm0(beta, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlayernorm_set_beta(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Forward pass through layer normalization
     * レイヤー正規化の順伝播
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    forward(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlayernorm_forward(this.__wbg_ptr, ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}

const WasmLinearFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlinear_free(ptr >>> 0, 1));
/**
 * Complete linear layer for WASM neural networks
 * WASM用完全な線形レイヤー
 */
export class WasmLinear {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLinear.prototype);
        obj.__wbg_ptr = ptr;
        WasmLinearFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLinearFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlinear_free(ptr, 0);
    }
    /**
     * Create new linear layer with Xavier/Glorot initialization
     * Xavier/Glorot初期化による新しい線形レイヤーを作成
     * @param {number} in_features
     * @param {number} out_features
     * @param {boolean} bias
     */
    constructor(in_features, out_features, bias) {
        const ret = wasm.wasmlinear_new(in_features, out_features, bias);
        this.__wbg_ptr = ret >>> 0;
        WasmLinearFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create linear layer with custom initialization
     * カスタム初期化による線形レイヤーを作成
     * @param {number} in_features
     * @param {number} out_features
     * @param {Float32Array} weights
     * @param {Float32Array | null} [bias]
     * @returns {WasmLinear}
     */
    static with_weights(in_features, out_features, weights, bias) {
        const ptr0 = passArrayF32ToWasm0(weights, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        var ptr1 = isLikeNone(bias) ? 0 : passArrayF32ToWasm0(bias, wasm.__wbindgen_malloc);
        var len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlinear_with_weights(in_features, out_features, ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmLinear.__wrap(ret[0]);
    }
    /**
     * Forward pass through linear layer
     * 線形レイヤーの順伝播
     * @param {Float32Array} input
     * @param {number} batch_size
     * @returns {Float32Array}
     */
    forward(input, batch_size) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlinear_forward(this.__wbg_ptr, ptr0, len0, batch_size);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get layer parameters for training
     * 訓練用のレイヤーパラメータを取得
     * @returns {Float32Array}
     */
    get_weights() {
        const ret = wasm.wasmlinear_get_weights(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get bias parameters
     * バイアスパラメータを取得
     * @returns {Float32Array | undefined}
     */
    get_bias() {
        const ret = wasm.wasmlinear_get_bias(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * Update weights with new values
     * 新しい値で重みを更新
     * @param {Float32Array} new_weights
     */
    update_weights(new_weights) {
        const ptr0 = passArrayF32ToWasm0(new_weights, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlinear_update_weights(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Update bias with new values
     * 新しい値でバイアスを更新
     * @param {Float32Array} new_bias
     */
    update_bias(new_bias) {
        const ptr0 = passArrayF32ToWasm0(new_bias, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmlinear_update_bias(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get input features count
     * @returns {number}
     */
    in_features() {
        const ret = wasm.wasmlinear_in_features(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get output features count
     * @returns {number}
     */
    out_features() {
        const ret = wasm.wasmlinear_out_features(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Check if layer has bias
     * @returns {boolean}
     */
    has_bias() {
        const ret = wasm.wasmlinear_has_bias(this.__wbg_ptr);
        return ret !== 0;
    }
}

const WasmLoggerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlogger_free(ptr >>> 0, 1));
/**
 * WASM-specific logging utilities
 */
export class WasmLogger {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLoggerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlogger_free(ptr, 0);
    }
    /**
     * Log info message
     * @param {string} message
     */
    static info(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlogger_info(ptr0, len0);
    }
    /**
     * Log warning message
     * @param {string} message
     */
    static warn(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlogger_warn(ptr0, len0);
    }
    /**
     * Log error message
     * @param {string} message
     */
    static error(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlogger_error(ptr0, len0);
    }
    /**
     * Log debug message
     * @param {string} message
     */
    static debug(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmlogger_debug(ptr0, len0);
    }
}

const WasmLossFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmloss_free(ptr >>> 0, 1));
/**
 * WASM-compatible loss functions
 * WASM互換損失関数
 */
export class WasmLoss {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLossFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmloss_free(ptr, 0);
    }
    /**
     * Mean Squared Error (MSE) loss
     * MSE(y_pred, y_true) = mean((y_pred - y_true)²)
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static mse_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_mse_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Mean Absolute Error (MAE) loss
     * MAE(y_pred, y_true) = mean(|y_pred - y_true|)
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static mae_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_mae_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Huber loss (smooth L1 loss)
     * Combines MSE and MAE for robustness
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @param {number} delta
     * @returns {number}
     */
    static huber_loss(predictions, targets, delta) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_huber_loss(ptr0, len0, ptr1, len1, delta);
        return ret;
    }
    /**
     * Cross-entropy loss for binary classification
     * Binary Cross-Entropy: -mean(y*log(p) + (1-y)*log(1-p))
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static binary_cross_entropy_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_binary_cross_entropy_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Cross-entropy loss for multiclass classification
     * Input: logits (raw scores), targets (one-hot or class indices)
     * @param {Float32Array} logits
     * @param {Float32Array} targets
     * @returns {number}
     */
    static cross_entropy_loss(logits, targets) {
        const ptr0 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_cross_entropy_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Sparse cross-entropy loss (targets as class indices instead of one-hot)
     * logits: \[batch_size * num_classes\], targets: \[batch_size\] (class indices)
     * @param {Float32Array} logits
     * @param {Uint32Array} targets
     * @param {number} num_classes
     * @returns {number}
     */
    static sparse_cross_entropy_loss(logits, targets, num_classes) {
        const ptr0 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_sparse_cross_entropy_loss(ptr0, len0, ptr1, len1, num_classes);
        return ret;
    }
    /**
     * KL Divergence loss
     * KL(P||Q) = sum(P * log(P/Q))
     * @param {Float32Array} p_distribution
     * @param {Float32Array} q_distribution
     * @returns {number}
     */
    static kl_divergence_loss(p_distribution, q_distribution) {
        const ptr0 = passArrayF32ToWasm0(p_distribution, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(q_distribution, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_kl_divergence_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Focal loss for handling class imbalance
     * FL(pt) = -α(1-pt)^γ log(pt)
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @param {number} alpha
     * @param {number} gamma
     * @returns {number}
     */
    static focal_loss(predictions, targets, alpha, gamma) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_focal_loss(ptr0, len0, ptr1, len1, alpha, gamma);
        return ret;
    }
    /**
     * Cosine similarity loss
     * Loss = 1 - cosine_similarity(pred, target)
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static cosine_similarity_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_cosine_similarity_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Hinge loss for SVM-style classification
     * Hinge(y, f(x)) = max(0, 1 - y * f(x))
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static hinge_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_hinge_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Squared hinge loss (smooth version of hinge loss)
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static squared_hinge_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_squared_hinge_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Log-cosh loss (smooth version of MAE)
     * LogCosh(y_pred, y_true) = mean(log(cosh(y_pred - y_true)))
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static log_cosh_loss(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_log_cosh_loss(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Combined loss function selector
     * 損失関数セレクター
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @param {string} loss_type
     * @returns {number}
     */
    static compute_loss(predictions, targets, loss_type) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(loss_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_compute_loss(ptr0, len0, ptr1, len1, ptr2, len2);
        return ret;
    }
    /**
     * Get loss function gradient for backpropagation
     * 逆伝播用の損失関数勾配を取得
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @param {string} loss_type
     * @returns {Float32Array}
     */
    static loss_gradient(predictions, targets, loss_type) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(loss_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmloss_loss_gradient(ptr0, len0, ptr1, len1, ptr2, len2);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
}

const WasmMemoryMonitorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmemorymonitor_free(ptr >>> 0, 1));
/**
 * Memory usage monitor for WASM
 */
export class WasmMemoryMonitor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMemoryMonitorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmemorymonitor_free(ptr, 0);
    }
    /**
     * Create a new memory usage monitor
     * 新しいメモリ使用量モニターを作成
     */
    constructor() {
        const ret = wasm.wasmmemorymonitor_new();
        this.__wbg_ptr = ret >>> 0;
        WasmMemoryMonitorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Record memory allocation
     * @param {number} size
     */
    record_allocation(size) {
        wasm.wasmmemorymonitor_record_allocation(this.__wbg_ptr, size);
    }
    /**
     * Record memory deallocation
     * @param {number} size
     */
    record_deallocation(size) {
        wasm.wasmmemorymonitor_record_deallocation(this.__wbg_ptr, size);
    }
    /**
     * Get current memory usage
     * @returns {number}
     */
    current_usage() {
        const ret = wasm.wasmmemorymonitor_current_usage(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get peak memory usage
     * @returns {number}
     */
    peak_usage() {
        const ret = wasm.wasmmemorymonitor_peak_usage(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reset statistics
     */
    reset() {
        wasm.wasmmemorymonitor_reset(this.__wbg_ptr);
    }
}

const WasmMemoryPoolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmemorypool_free(ptr >>> 0, 1));
/**
 * Memory pool for efficient tensor allocation
 */
export class WasmMemoryPool {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMemoryPoolFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmemorypool_free(ptr, 0);
    }
    /**
     * Create new memory pool for efficient buffer management
     */
    constructor() {
        const ret = wasm.wasmmemorypool_new();
        this.__wbg_ptr = ret >>> 0;
        WasmMemoryPoolFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get a buffer from the pool or allocate new one
     * @param {number} size
     * @returns {Float32Array}
     */
    get_buffer(size) {
        const ret = wasm.wasmmemorypool_get_buffer(this.__wbg_ptr, size);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Return a buffer to the pool
     * @param {Float32Array} buffer
     */
    return_buffer(buffer) {
        const ptr0 = passArrayF32ToWasm0(buffer, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmmemorypool_return_buffer(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get pool statistics
     * @returns {string}
     */
    get_stats() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmmemorypool_get_stats(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Clear all pools
     */
    clear() {
        wasm.wasmmemorypool_clear(this.__wbg_ptr);
    }
}

const WasmMetricsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmetrics_free(ptr >>> 0, 1));
/**
 * Model evaluation metrics calculator
 * モデル評価メトリクス計算機
 */
export class WasmMetrics {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMetricsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmetrics_free(ptr, 0);
    }
    /**
     * Calculate accuracy for classification tasks
     * 分類タスクの精度を計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @returns {number}
     */
    static accuracy(predictions, targets) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_accuracy(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Calculate precision for binary classification
     * バイナリ分類の適合率を計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @param {number} positive_class
     * @returns {number}
     */
    static precision(predictions, targets, positive_class) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_precision(ptr0, len0, ptr1, len1, positive_class);
        return ret;
    }
    /**
     * Calculate recall for binary classification
     * バイナリ分類の再現率を計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @param {number} positive_class
     * @returns {number}
     */
    static recall(predictions, targets, positive_class) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_recall(ptr0, len0, ptr1, len1, positive_class);
        return ret;
    }
    /**
     * Calculate F1 score
     * F1スコアを計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @param {number} positive_class
     * @returns {number}
     */
    static f1_score(predictions, targets, positive_class) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_f1_score(ptr0, len0, ptr1, len1, positive_class);
        return ret;
    }
    /**
     * Calculate confusion matrix for multi-class classification
     * 多クラス分類の混同行列を計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @param {number} num_classes
     * @returns {Uint32Array}
     */
    static confusion_matrix(predictions, targets, num_classes) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_confusion_matrix(ptr0, len0, ptr1, len1, num_classes);
        var v3 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Calculate Mean Absolute Error (MAE) for regression
     * 回帰のための平均絶対誤差（MAE）を計算
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static mae(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_mae(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Calculate Mean Squared Error (MSE) for regression
     * 回帰のための平均二乗誤差（MSE）を計算
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static mse(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_mse(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Calculate Root Mean Squared Error (RMSE) for regression
     * 回帰のための平方根平均二乗誤差（RMSE）を計算
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static rmse(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_rmse(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Calculate R-squared coefficient for regression
     * 回帰のための決定係数（R二乗）を計算
     * @param {Float32Array} predictions
     * @param {Float32Array} targets
     * @returns {number}
     */
    static r2_score(predictions, targets) {
        const ptr0 = passArrayF32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_r2_score(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Calculate top-k accuracy for multi-class classification
     * 多クラス分類のためのtop-k精度を計算
     * @param {Float32Array} logits
     * @param {Uint32Array} targets
     * @param {number} num_classes
     * @param {number} k
     * @returns {number}
     */
    static top_k_accuracy(logits, targets, num_classes, k) {
        const ptr0 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_top_k_accuracy(ptr0, len0, ptr1, len1, num_classes, k);
        return ret;
    }
    /**
     * Calculate comprehensive classification report
     * 包括的な分類レポートを計算
     * @param {Uint32Array} predictions
     * @param {Uint32Array} targets
     * @param {number} num_classes
     * @returns {object}
     */
    static classification_report(predictions, targets, num_classes) {
        const ptr0 = passArray32ToWasm0(predictions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmetrics_classification_report(ptr0, len0, ptr1, len1, num_classes);
        return ret;
    }
}

const WasmModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmodel_free(ptr >>> 0, 1));
/**
 * Simple neural network model for WASM
 */
export class WasmModel {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmModelFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmodel_free(ptr, 0);
    }
    /**
     * Create new neural network model
     */
    constructor() {
        const ret = wasm.wasmmodel_new();
        this.__wbg_ptr = ret >>> 0;
        WasmModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add linear layer
     * @param {number} in_features
     * @param {number} out_features
     * @param {boolean} _bias
     */
    add_linear(in_features, out_features, _bias) {
        wasm.wasmmodel_add_linear(this.__wbg_ptr, in_features, out_features, _bias);
    }
    /**
     * Add ReLU activation
     */
    add_relu() {
        wasm.wasmmodel_add_relu(this.__wbg_ptr);
    }
    /**
     * Get number of layers
     * @returns {number}
     */
    num_layers() {
        const ret = wasm.wasmmodel_num_layers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Simple forward pass (placeholder)
     * @param {WasmTensor} input
     * @returns {WasmTensor}
     */
    forward(input) {
        _assertClass(input, WasmTensor);
        const ret = wasm.wasmmodel_forward(this.__wbg_ptr, input.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
}

const WasmNormalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmnormal_free(ptr >>> 0, 1));
/**
 * Normal (Gaussian) distribution for WASM
 * WASM用正規（ガウス）分布
 */
export class WasmNormal {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmNormal.prototype);
        obj.__wbg_ptr = ptr;
        WasmNormalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmNormalFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmnormal_free(ptr, 0);
    }
    /**
     * Create new normal distribution
     * @param {number} mean
     * @param {number} std_dev
     * @param {number} seed
     */
    constructor(mean, std_dev, seed) {
        const ret = wasm.wasmnormal_new(mean, std_dev, seed);
        this.__wbg_ptr = ret >>> 0;
        WasmNormalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create standard normal distribution (mean=0, std=1)
     * @param {number} seed
     * @returns {WasmNormal}
     */
    static standard(seed) {
        const ret = wasm.wasmnormal_standard(seed);
        return WasmNormal.__wrap(ret);
    }
    /**
     * Sample single value using Box-Muller transform
     * @returns {number}
     */
    sample() {
        const ret = wasm.wasmnormal_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * Sample multiple values
     * @param {number} n
     * @returns {Float32Array}
     */
    sample_n(n) {
        const ret = wasm.wasmnormal_sample_n(this.__wbg_ptr, n);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Probability density function
     * @param {number} x
     * @returns {number}
     */
    pdf(x) {
        const ret = wasm.wasmnormal_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Log probability density function
     * @param {number} x
     * @returns {number}
     */
    log_pdf(x) {
        const ret = wasm.wasmnormal_log_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Cumulative distribution function (using error function approximation)
     * @param {number} x
     * @returns {number}
     */
    cdf(x) {
        const ret = wasm.wasmnormal_cdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Get mean
     * @returns {number}
     */
    mean() {
        const ret = wasm.wasmbernoulli_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get standard deviation
     * @returns {number}
     */
    std_dev() {
        const ret = wasm.wasmnormal_std_dev(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get variance
     * @returns {number}
     */
    variance() {
        const ret = wasm.wasmnormal_variance(this.__wbg_ptr);
        return ret;
    }
}

const WasmNormalizeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmnormalize_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Normalize transformation
 */
export class WasmNormalize {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmNormalize.prototype);
        obj.__wbg_ptr = ptr;
        WasmNormalizeFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmNormalizeFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmnormalize_free(ptr, 0);
    }
    /**
     * Create new normalization transform
     * @param {Float32Array} mean
     * @param {Float32Array} std
     */
    constructor(mean, std) {
        const ptr0 = passArrayF32ToWasm0(mean, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(std, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmnormalize_new(ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmNormalizeFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply normalization to tensor
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmnormalize_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmnormalize_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmOptimizerFactoryFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmoptimizerfactory_free(ptr >>> 0, 1));
/**
 * Optimizer factory for creating different optimizers
 * 異なるオプティマイザを作成するファクトリ
 */
export class WasmOptimizerFactory {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmOptimizerFactoryFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmoptimizerfactory_free(ptr, 0);
    }
    /**
     * Create optimizer by name
     * @param {number} learning_rate
     * @param {number} momentum
     * @param {number} weight_decay
     * @returns {WasmSGD}
     */
    static create_sgd(learning_rate, momentum, weight_decay) {
        const ret = wasm.wasmoptimizerfactory_create_sgd(learning_rate, momentum, weight_decay);
        return WasmSGD.__wrap(ret);
    }
    /**
     * Create Adam optimizer
     * @param {number} learning_rate
     * @param {number} beta1
     * @param {number} beta2
     * @param {number} epsilon
     * @param {number} weight_decay
     * @returns {WasmAdam}
     */
    static create_adam(learning_rate, beta1, beta2, epsilon, weight_decay) {
        const ret = wasm.wasmoptimizerfactory_create_adam(learning_rate, beta1, beta2, epsilon, weight_decay);
        return WasmAdam.__wrap(ret);
    }
    /**
     * Create AdaGrad optimizer
     * @param {number} learning_rate
     * @param {number} epsilon
     * @returns {WasmAdaGrad}
     */
    static create_adagrad(learning_rate, epsilon) {
        const ret = wasm.wasmoptimizerfactory_create_adagrad(learning_rate, epsilon);
        return WasmAdaGrad.__wrap(ret);
    }
    /**
     * Create RMSprop optimizer
     * @param {number} learning_rate
     * @param {number} alpha
     * @param {number} epsilon
     * @param {number} momentum
     * @returns {WasmRMSprop}
     */
    static create_rmsprop(learning_rate, alpha, epsilon, momentum) {
        const ret = wasm.wasmoptimizerfactory_create_rmsprop(learning_rate, alpha, epsilon, momentum);
        return WasmRMSprop.__wrap(ret);
    }
}

const WasmPerformanceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmperformance_free(ptr >>> 0, 1));
/**
 * Performance monitoring for WASM
 */
export class WasmPerformance {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmPerformanceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmperformance_free(ptr, 0);
    }
    /**
     * Create a new performance monitor
     * 新しいパフォーマンスモニターを作成
     */
    constructor() {
        const ret = wasm.wasmperformance_new();
        this.__wbg_ptr = ret >>> 0;
        WasmPerformanceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Start performance measurement
     */
    start() {
        wasm.wasmperformance_start(this.__wbg_ptr);
    }
    /**
     * Get elapsed time in milliseconds
     * @returns {number}
     */
    elapsed() {
        const ret = wasm.wasmperformance_elapsed(this.__wbg_ptr);
        return ret;
    }
    /**
     * Log performance result
     * @param {string} operation_name
     */
    log(operation_name) {
        const ptr0 = passStringToWasm0(operation_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmperformance_log(this.__wbg_ptr, ptr0, len0);
    }
}

const WasmPreprocessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmpreprocessor_free(ptr >>> 0, 1));
/**
 * Data preprocessing utilities for neural networks
 * ニューラルネットワーク用のデータ前処理ユーティリティ
 */
export class WasmPreprocessor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmPreprocessorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmpreprocessor_free(ptr, 0);
    }
    /**
     * Normalize data using min-max normalization: (x - min) / (max - min)
     * min-max正規化を使用してデータを正規化: (x - min) / (max - min)
     * @param {Float32Array} data
     * @param {number} min_val
     * @param {number} max_val
     * @returns {Float32Array}
     */
    static min_max_normalize(data, min_val, max_val) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_min_max_normalize(ptr0, len0, min_val, max_val);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Standardize data using z-score normalization: (x - mean) / std
     * z-score正規化を使用してデータを標準化: (x - mean) / std
     * @param {Float32Array} data
     * @param {number} mean
     * @param {number} std
     * @returns {Float32Array}
     */
    static z_score_normalize(data, mean, std) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_z_score_normalize(ptr0, len0, mean, std);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Compute statistics (mean, std, min, max) for normalization
     * 正規化用の統計値（平均、標準偏差、最小値、最大値）を計算
     * @param {Float32Array} data
     * @returns {Float32Array}
     */
    static compute_stats(data) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_compute_stats(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * One-hot encoding for categorical data
     * カテゴリカルデータのワンホットエンコーディング
     * @param {Uint32Array} labels
     * @param {number} num_classes
     * @returns {Float32Array}
     */
    static one_hot_encode(labels, num_classes) {
        const ptr0 = passArray32ToWasm0(labels, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_one_hot_encode(ptr0, len0, num_classes);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Convert one-hot encoding back to labels
     * ワンホットエンコーディングをラベルに戻す
     * @param {Float32Array} one_hot
     * @param {number} num_classes
     * @returns {Uint32Array}
     */
    static one_hot_decode(one_hot, num_classes) {
        const ptr0 = passArrayF32ToWasm0(one_hot, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_one_hot_decode(ptr0, len0, num_classes);
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Data augmentation: add Gaussian noise
     * データ拡張: ガウシアンノイズの追加
     * @param {Float32Array} data
     * @param {number} mean
     * @param {number} std
     * @param {number} seed
     * @returns {Float32Array}
     */
    static add_gaussian_noise(data, mean, std, seed) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_add_gaussian_noise(ptr0, len0, mean, std, seed);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Train-test split for datasets
     * データセットの訓練・テスト分割
     * @param {Float32Array} features
     * @param {Float32Array} targets
     * @param {number} feature_size
     * @param {number} test_ratio
     * @param {number} seed
     * @returns {object}
     */
    static train_test_split(features, targets, feature_size, test_ratio, seed) {
        const ptr0 = passArrayF32ToWasm0(features, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_train_test_split(ptr0, len0, ptr1, len1, feature_size, test_ratio, seed);
        return ret;
    }
    /**
     * Batch data for training
     * 訓練用のデータバッチ化
     * @param {Float32Array} features
     * @param {Float32Array} targets
     * @param {number} feature_size
     * @param {number} batch_size
     * @returns {Array<any>}
     */
    static create_batches(features, targets, feature_size, batch_size) {
        const ptr0 = passArrayF32ToWasm0(features, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(targets, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmpreprocessor_create_batches(ptr0, len0, ptr1, len1, feature_size, batch_size);
        return ret;
    }
}

const WasmProcessingPipelineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmprocessingpipeline_free(ptr >>> 0, 1));
/**
 * Processing pipeline for analysis operations
 */
export class WasmProcessingPipeline {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmProcessingPipelineFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmprocessingpipeline_free(ptr, 0);
    }
    /**
     * Create new processing pipeline
     * @param {boolean} parallel_execution
     */
    constructor(parallel_execution) {
        const ret = wasm.wasmprocessingpipeline_new(parallel_execution);
        this.__wbg_ptr = ret >>> 0;
        WasmProcessingPipelineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add operation to pipeline
     * @param {string} operation_name
     */
    add_operation(operation_name) {
        const ptr0 = passStringToWasm0(operation_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmprocessingpipeline_add_operation(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get operation count
     * @returns {number}
     */
    operation_count() {
        const ret = wasm.wasmmodel_num_layers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get pipeline configuration
     * @returns {string}
     */
    get_config() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmprocessingpipeline_get_config(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmQualityMetricsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmqualitymetrics_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Quality Metrics
 */
export class WasmQualityMetrics {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmQualityMetrics.prototype);
        obj.__wbg_ptr = ptr;
        WasmQualityMetricsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmQualityMetricsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmqualitymetrics_free(ptr, 0);
    }
    /**
     * Create new quality metrics analyzer
     * @param {number} threshold
     */
    constructor(threshold) {
        const ret = wasm.wasmqualitymetrics_new(threshold);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmQualityMetricsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Calculate data completeness (percentage of non-NaN values)
     * @param {WasmTensor} tensor
     * @returns {number}
     */
    completeness(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_completeness(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate data accuracy (values within expected range)
     * @param {WasmTensor} tensor
     * @param {number} min_val
     * @param {number} max_val
     * @returns {number}
     */
    accuracy(tensor, min_val, max_val) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_accuracy(this.__wbg_ptr, tensor.__wbg_ptr, min_val, max_val);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate data consistency (low variance indicator)
     * @param {WasmTensor} tensor
     * @returns {number}
     */
    consistency(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_consistency(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate data validity (percentage of finite values)
     * @param {WasmTensor} tensor
     * @returns {number}
     */
    validity(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_validity(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate data uniqueness (ratio of unique values)
     * @param {WasmTensor} tensor
     * @returns {number}
     */
    uniqueness(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_uniqueness(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Comprehensive quality score
     * @param {WasmTensor} tensor
     * @returns {number}
     */
    overall_quality(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmqualitymetrics_overall_quality(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Get quality report as JSON string
     * @param {WasmTensor} tensor
     * @returns {string}
     */
    quality_report(tensor) {
        let deferred2_0;
        let deferred2_1;
        try {
            _assertClass(tensor, WasmTensor);
            const ret = wasm.wasmqualitymetrics_quality_report(this.__wbg_ptr, tensor.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
}

const WasmRMSpropFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrmsprop_free(ptr >>> 0, 1));
/**
 * RMSprop optimizer for WASM
 * WASM用RMSpropオプティマイザ
 */
export class WasmRMSprop {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmRMSprop.prototype);
        obj.__wbg_ptr = ptr;
        WasmRMSpropFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRMSpropFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrmsprop_free(ptr, 0);
    }
    /**
     * Create new RMSprop optimizer
     * @param {number} learning_rate
     * @param {number} alpha
     * @param {number} epsilon
     */
    constructor(learning_rate, alpha, epsilon) {
        const ret = wasm.wasmrmsprop_new(learning_rate, alpha, epsilon);
        this.__wbg_ptr = ret >>> 0;
        WasmRMSpropFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create RMSprop with momentum
     * @param {number} learning_rate
     * @param {number} alpha
     * @param {number} epsilon
     * @param {number} momentum
     * @returns {WasmRMSprop}
     */
    static with_momentum(learning_rate, alpha, epsilon, momentum) {
        const ret = wasm.wasmrmsprop_with_momentum(learning_rate, alpha, epsilon, momentum);
        return WasmRMSprop.__wrap(ret);
    }
    /**
     * Update parameters with RMSprop algorithm
     * @param {string} param_id
     * @param {Float32Array} parameters
     * @param {Float32Array} gradients
     * @returns {Float32Array}
     */
    step(param_id, parameters, gradients) {
        const ptr0 = passStringToWasm0(param_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(parameters, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmrmsprop_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
}

const WasmRandomCropFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrandomcrop_free(ptr >>> 0, 1));
/**
 * WASM wrapper for RandomCrop transformation
 */
export class WasmRandomCrop {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRandomCropFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrandomcrop_free(ptr, 0);
    }
    /**
     * Create new random crop transform
     * @param {number} height
     * @param {number} width
     * @param {number | null} [padding]
     */
    constructor(height, width, padding) {
        const ret = wasm.wasmrandomcrop_new(height, width, isLikeNone(padding) ? 0x100000001 : (padding) >>> 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmRandomCropFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply random crop to tensor
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmrandomcrop_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmrandomcrop_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmReLUFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrelu_free(ptr >>> 0, 1));
/**
 * Simple ReLU activation for WASM
 */
export class WasmReLU {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReLUFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrelu_free(ptr, 0);
    }
    /**
     * Create new ReLU activation layer
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        WasmReLUFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply ReLU activation function
     * @param {WasmTensor} input
     * @returns {WasmTensor}
     */
    forward(input) {
        _assertClass(input, WasmTensor);
        const ret = wasm.wasmrelu_forward(this.__wbg_ptr, input.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
}

const WasmResizeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmresize_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Resize transformation
 */
export class WasmResize {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmResizeFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmresize_free(ptr, 0);
    }
    /**
     * Create new resize transform
     * @param {number} height
     * @param {number} width
     * @param {string} interpolation
     */
    constructor(height, width, interpolation) {
        const ptr0 = passStringToWasm0(interpolation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmresize_new(height, width, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmResizeFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply resize to tensor
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmresize_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmresize_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmRngFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrng_free(ptr >>> 0, 1));
/**
 * WASM-compatible random number generator using Linear Congruential Generator
 * WASM互換の線形合同法乱数生成器
 */
export class WasmRng {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRngFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrng_free(ptr, 0);
    }
    /**
     * Create new RNG with seed
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.wasmrng_new(seed);
        this.__wbg_ptr = ret >>> 0;
        WasmRngFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Generate next random u32
     * @returns {number}
     */
    next_u32() {
        const ret = wasm.wasmrng_next_u32(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Generate random f32 in [0, 1)
     * @returns {number}
     */
    next_f32() {
        const ret = wasm.wasmrng_next_f32(this.__wbg_ptr);
        return ret;
    }
    /**
     * Generate random f32 in [0, 1) (alternative name for consistency)
     * @returns {number}
     */
    uniform() {
        const ret = wasm.wasmrng_next_f32(this.__wbg_ptr);
        return ret;
    }
}

const WasmSGDFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsgd_free(ptr >>> 0, 1));
/**
 * SGD (Stochastic Gradient Descent) optimizer for WASM
 * WASM用SGD（確率的勾配降下法）オプティマイザ
 */
export class WasmSGD {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmSGD.prototype);
        obj.__wbg_ptr = ptr;
        WasmSGDFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSGDFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsgd_free(ptr, 0);
    }
    /**
     * Create new SGD optimizer
     * @param {number} learning_rate
     */
    constructor(learning_rate) {
        const ret = wasm.wasmsgd_new(learning_rate);
        this.__wbg_ptr = ret >>> 0;
        WasmSGDFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create SGD with momentum
     * @param {number} learning_rate
     * @param {number} momentum
     * @returns {WasmSGD}
     */
    static with_momentum(learning_rate, momentum) {
        const ret = wasm.wasmsgd_with_momentum(learning_rate, momentum);
        return WasmSGD.__wrap(ret);
    }
    /**
     * Create SGD with weight decay
     * @param {number} learning_rate
     * @param {number} momentum
     * @param {number} weight_decay
     * @returns {WasmSGD}
     */
    static with_weight_decay(learning_rate, momentum, weight_decay) {
        const ret = wasm.wasmsgd_with_weight_decay(learning_rate, momentum, weight_decay);
        return WasmSGD.__wrap(ret);
    }
    /**
     * Update parameters with gradients
     * @param {string} param_id
     * @param {Float32Array} parameters
     * @param {Float32Array} gradients
     * @returns {Float32Array}
     */
    step(param_id, parameters, gradients) {
        const ptr0 = passStringToWasm0(param_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(parameters, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsgd_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * Get learning rate
     * @returns {number}
     */
    get_learning_rate() {
        const ret = wasm.wasmsgd_get_learning_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set learning rate
     * @param {number} lr
     */
    set_learning_rate(lr) {
        wasm.wasmsgd_set_learning_rate(this.__wbg_ptr, lr);
    }
    /**
     * Clear momentum buffers
     */
    zero_grad() {
        wasm.wasmsgd_zero_grad(this.__wbg_ptr);
    }
}

const WasmSignalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsignal_free(ptr >>> 0, 1));
/**
 * WASM-compatible FFT implementation using Cooley-Tukey algorithm
 * WASM互換のCooley-TukeyアルゴリズムFFT実装
 */
export class WasmSignal {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSignalFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsignal_free(ptr, 0);
    }
    /**
     * Discrete Fourier Transform (DFT) - basic O(N²) implementation
     * 離散フーリエ変換(DFT) - 基本O(N²)実装
     * @param {Float32Array} real_input
     * @returns {object}
     */
    static dft(real_input) {
        const ptr0 = passArrayF32ToWasm0(real_input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_dft(ptr0, len0);
        return ret;
    }
    /**
     * Inverse Discrete Fourier Transform (IDFT)
     * 逆離散フーリエ変換(IDFT)
     * @param {Float32Array} real_input
     * @param {Float32Array} imag_input
     * @returns {object}
     */
    static idft(real_input, imag_input) {
        const ptr0 = passArrayF32ToWasm0(real_input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(imag_input, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_idft(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Real Fast Fourier Transform (RFFT) - optimized for real inputs
     * 実数高速フーリエ変換(RFFT) - 実数入力用最適化
     * @param {Float32Array} real_input
     * @returns {object}
     */
    static rfft(real_input) {
        const ptr0 = passArrayF32ToWasm0(real_input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_rfft(ptr0, len0);
        return ret;
    }
    /**
     * Compute power spectral density
     * パワースペクトル密度を計算
     * @param {Float32Array} real_input
     * @returns {Float32Array}
     */
    static power_spectrum(real_input) {
        const ptr0 = passArrayF32ToWasm0(real_input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_power_spectrum(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply Hamming window to signal
     * ハミング窓をシグナルに適用
     * @param {Float32Array} signal
     * @returns {Float32Array}
     */
    static hamming_window(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_hamming_window(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply Hanning window to signal
     * ハン窓をシグナルに適用
     * @param {Float32Array} signal
     * @returns {Float32Array}
     */
    static hanning_window(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_hanning_window(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply Blackman window to signal
     * ブラックマン窓をシグナルに適用
     * @param {Float32Array} signal
     * @returns {Float32Array}
     */
    static blackman_window(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_blackman_window(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Compute magnitude spectrum
     * 振幅スペクトルを計算
     * @param {Float32Array} real_fft
     * @param {Float32Array} imag_fft
     * @returns {Float32Array}
     */
    static magnitude_spectrum(real_fft, imag_fft) {
        const ptr0 = passArrayF32ToWasm0(real_fft, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(imag_fft, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_magnitude_spectrum(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Compute phase spectrum
     * 位相スペクトルを計算
     * @param {Float32Array} real_fft
     * @param {Float32Array} imag_fft
     * @returns {Float32Array}
     */
    static phase_spectrum(real_fft, imag_fft) {
        const ptr0 = passArrayF32ToWasm0(real_fft, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(imag_fft, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_phase_spectrum(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Generate frequency bins for FFT result
     * FFT結果の周波数ビンを生成
     * @param {number} n
     * @param {number} sample_rate
     * @returns {Float32Array}
     */
    static fft_frequencies(n, sample_rate) {
        const ret = wasm.wasmsignal_fft_frequencies(n, sample_rate);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Generate frequency bins for RFFT result
     * RFFT結果の周波数ビンを生成
     * @param {number} n
     * @param {number} sample_rate
     * @returns {Float32Array}
     */
    static rfft_frequencies(n, sample_rate) {
        const ret = wasm.wasmsignal_rfft_frequencies(n, sample_rate);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Apply low-pass filter (simple moving average)
     * ローパスフィルタを適用（単純移動平均）
     * @param {Float32Array} signal
     * @param {number} window_size
     * @returns {Float32Array}
     */
    static low_pass_filter(signal, window_size) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_low_pass_filter(ptr0, len0, window_size);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply high-pass filter (difference from moving average)
     * ハイパスフィルタを適用（移動平均との差分）
     * @param {Float32Array} signal
     * @param {number} window_size
     * @returns {Float32Array}
     */
    static high_pass_filter(signal, window_size) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_high_pass_filter(ptr0, len0, window_size);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Compute cross-correlation between two signals
     * 2つの信号間の相互相関を計算
     * @param {Float32Array} signal_a
     * @param {Float32Array} signal_b
     * @returns {Float32Array}
     */
    static cross_correlation(signal_a, signal_b) {
        const ptr0 = passArrayF32ToWasm0(signal_a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(signal_b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_cross_correlation(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Compute autocorrelation of a signal
     * 信号の自己相関を計算
     * @param {Float32Array} signal
     * @returns {Float32Array}
     */
    static autocorrelation(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_autocorrelation(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Generate sine wave
     * 正弦波を生成
     * @param {number} frequency
     * @param {number} sample_rate
     * @param {number} duration
     * @param {number} amplitude
     * @param {number} phase
     * @returns {Float32Array}
     */
    static generate_sine_wave(frequency, sample_rate, duration, amplitude, phase) {
        const ret = wasm.wasmsignal_generate_sine_wave(frequency, sample_rate, duration, amplitude, phase);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Generate cosine wave
     * 余弦波を生成
     * @param {number} frequency
     * @param {number} sample_rate
     * @param {number} duration
     * @param {number} amplitude
     * @param {number} phase
     * @returns {Float32Array}
     */
    static generate_cosine_wave(frequency, sample_rate, duration, amplitude, phase) {
        const ret = wasm.wasmsignal_generate_cosine_wave(frequency, sample_rate, duration, amplitude, phase);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Generate white noise
     * ホワイトノイズを生成
     * @param {number} num_samples
     * @param {number} amplitude
     * @param {number} seed
     * @returns {Float32Array}
     */
    static generate_white_noise(num_samples, amplitude, seed) {
        const ret = wasm.wasmsignal_generate_white_noise(num_samples, amplitude, seed);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Compute signal energy
     * 信号エネルギーを計算
     * @param {Float32Array} signal
     * @returns {number}
     */
    static signal_energy(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_signal_energy(ptr0, len0);
        return ret;
    }
    /**
     * Compute signal power (average energy)
     * 信号パワー（平均エネルギー）を計算
     * @param {Float32Array} signal
     * @returns {number}
     */
    static signal_power(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_signal_power(ptr0, len0);
        return ret;
    }
    /**
     * Compute root mean square (RMS) amplitude
     * 実効値（RMS）振幅を計算
     * @param {Float32Array} signal
     * @returns {number}
     */
    static rms_amplitude(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_rms_amplitude(ptr0, len0);
        return ret;
    }
    /**
     * Find peaks in signal
     * 信号のピークを検出
     * @param {Float32Array} signal
     * @param {number} threshold
     * @returns {Uint32Array}
     */
    static find_peaks(signal, threshold) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_find_peaks(ptr0, len0, threshold);
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply gain (amplification) to signal
     * 信号にゲイン（増幅）を適用
     * @param {Float32Array} signal
     * @param {number} gain
     * @returns {Float32Array}
     */
    static apply_gain(signal, gain) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_apply_gain(ptr0, len0, gain);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Normalize signal to range [-1, 1]
     * 信号を[-1, 1]範囲に正規化
     * @param {Float32Array} signal
     * @returns {Float32Array}
     */
    static normalize_signal(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_normalize_signal(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Compute zero-crossing rate
     * ゼロクロッシング率を計算
     * @param {Float32Array} signal
     * @returns {number}
     */
    static zero_crossing_rate(signal) {
        const ptr0 = passArrayF32ToWasm0(signal, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsignal_zero_crossing_rate(ptr0, len0);
        return ret;
    }
}

const WasmSpecialFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmspecial_free(ptr >>> 0, 1));
/**
 * Gamma function implementation for WASM
 * WASM用ガンマ関数実装
 */
export class WasmSpecial {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSpecialFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmspecial_free(ptr, 0);
    }
    /**
     * Gamma function Γ(x)
     * Using Lanczos approximation for accuracy
     * @param {number} x
     * @returns {number}
     */
    static gamma(x) {
        const ret = wasm.wasmspecial_gamma(x);
        return ret;
    }
    /**
     * Natural logarithm of gamma function ln(Γ(x))
     * @param {number} x
     * @returns {number}
     */
    static lgamma(x) {
        const ret = wasm.wasmspecial_lgamma(x);
        return ret;
    }
    /**
     * Digamma function ψ(x) = d/dx ln(Γ(x))
     * @param {number} x
     * @returns {number}
     */
    static digamma(x) {
        const ret = wasm.wasmspecial_digamma(x);
        return ret;
    }
    /**
     * Error function erf(x)
     * @param {number} x
     * @returns {number}
     */
    static erf(x) {
        const ret = wasm.wasmspecial_erf(x);
        return ret;
    }
    /**
     * Complementary error function erfc(x) = 1 - erf(x)
     * @param {number} x
     * @returns {number}
     */
    static erfc(x) {
        const ret = wasm.wasmspecial_erfc(x);
        return ret;
    }
    /**
     * Beta function B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
     * @param {number} a
     * @param {number} b
     * @returns {number}
     */
    static beta(a, b) {
        const ret = wasm.wasmspecial_beta(a, b);
        return ret;
    }
    /**
     * Bessel function of the first kind J₀(x)
     * @param {number} x
     * @returns {number}
     */
    static bessel_j0(x) {
        const ret = wasm.wasmspecial_bessel_j0(x);
        return ret;
    }
    /**
     * Bessel function of the first kind J₁(x)
     * @param {number} x
     * @returns {number}
     */
    static bessel_j1(x) {
        const ret = wasm.wasmspecial_bessel_j1(x);
        return ret;
    }
    /**
     * Modified Bessel function of the first kind I₀(x)
     * @param {number} x
     * @returns {number}
     */
    static bessel_i0(x) {
        const ret = wasm.wasmspecial_bessel_i0(x);
        return ret;
    }
}

const WasmStatisticalAnalyzerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmstatisticalanalyzer_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Statistical Analysis
 */
export class WasmStatisticalAnalyzer {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmStatisticalAnalyzerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmstatisticalanalyzer_free(ptr, 0);
    }
    /**
     * Create new statistical analyzer
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        WasmStatisticalAnalyzerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Calculate basic statistics
     * @param {WasmTensor} tensor
     * @returns {string}
     */
    basic_stats(tensor) {
        let deferred2_0;
        let deferred2_1;
        try {
            _assertClass(tensor, WasmTensor);
            const ret = wasm.wasmstatisticalanalyzer_basic_stats(this.__wbg_ptr, tensor.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Calculate percentiles
     * @param {WasmTensor} tensor
     * @param {Float32Array} percentiles
     * @returns {Array<any>}
     */
    percentiles(tensor, percentiles) {
        _assertClass(tensor, WasmTensor);
        const ptr0 = passArrayF32ToWasm0(percentiles, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmstatisticalanalyzer_percentiles(this.__wbg_ptr, tensor.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Detect outliers using IQR method
     * @param {WasmTensor} tensor
     * @returns {Array<any>}
     */
    detect_outliers(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmstatisticalanalyzer_detect_outliers(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const WasmStatisticalFunctionsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmstatisticalfunctions_free(ptr >>> 0, 1));
/**
 * Advanced statistical functions for web applications
 */
export class WasmStatisticalFunctions {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmStatisticalFunctionsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmstatisticalfunctions_free(ptr, 0);
    }
    /**
     * Create new statistical functions instance
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        WasmStatisticalFunctionsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Calculate correlation coefficient between two tensors
     * @param {WasmTensor} x
     * @param {WasmTensor} y
     * @returns {number}
     */
    correlation(x, y) {
        _assertClass(x, WasmTensor);
        _assertClass(y, WasmTensor);
        const ret = wasm.wasmstatisticalfunctions_correlation(this.__wbg_ptr, x.__wbg_ptr, y.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate covariance between two tensors
     * @param {WasmTensor} x
     * @param {WasmTensor} y
     * @returns {number}
     */
    covariance(x, y) {
        _assertClass(x, WasmTensor);
        _assertClass(y, WasmTensor);
        const ret = wasm.wasmstatisticalfunctions_covariance(this.__wbg_ptr, x.__wbg_ptr, y.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate percentile
     * @param {WasmTensor} tensor
     * @param {number} percentile
     * @returns {number}
     */
    percentile(tensor, percentile) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmstatisticalfunctions_percentile(this.__wbg_ptr, tensor.__wbg_ptr, percentile);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Calculate quantiles
     * @param {WasmTensor} tensor
     * @param {Float32Array} q
     * @returns {Array<any>}
     */
    quantiles(tensor, q) {
        _assertClass(tensor, WasmTensor);
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmstatisticalfunctions_quantiles(this.__wbg_ptr, tensor.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}

const WasmTensorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtensor_free(ptr >>> 0, 1));
/**
 * WASM-compatible tensor wrapper
 * WASM互換テンソルラッパー
 */
export class WasmTensor {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmTensor.prototype);
        obj.__wbg_ptr = ptr;
        WasmTensorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTensorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtensor_free(ptr, 0);
    }
    /**
     * Create a new WASM tensor
     * @param {Float32Array} data
     * @param {Uint32Array} shape
     */
    constructor(data, shape) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensor_new(ptr0, len0, ptr1, len1);
        this.__wbg_ptr = ret >>> 0;
        WasmTensorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get tensor data
     * @returns {Float32Array}
     */
    get data() {
        const ret = wasm.wasmtensor_data(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get tensor shape
     * @returns {Uint32Array}
     */
    get shape() {
        const ret = wasm.wasmtensor_shape(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Element-wise addition
     * @param {WasmTensor} other
     * @returns {WasmTensor}
     */
    add(other) {
        _assertClass(other, WasmTensor);
        const ret = wasm.wasmtensor_add(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Element-wise multiplication
     * @param {WasmTensor} other
     * @returns {WasmTensor}
     */
    multiply(other) {
        _assertClass(other, WasmTensor);
        const ret = wasm.wasmtensor_multiply(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * ReLU activation
     * @returns {WasmTensor}
     */
    relu() {
        const ret = wasm.wasmtensor_relu(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Sigmoid activation
     * @returns {WasmTensor}
     */
    sigmoid() {
        const ret = wasm.wasmtensor_sigmoid(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Matrix multiplication (2D only)
     * @param {WasmTensor} other
     * @returns {WasmTensor}
     */
    matmul(other) {
        _assertClass(other, WasmTensor);
        const ret = wasm.wasmtensor_matmul(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Create tensor filled with zeros
     * @param {Uint32Array} shape
     * @returns {WasmTensor}
     */
    static zeros(shape) {
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensor_zeros(ptr0, len0);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Create tensor filled with ones
     * @param {Uint32Array} shape
     * @returns {WasmTensor}
     */
    static ones(shape) {
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensor_ones(ptr0, len0);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Create tensor with random values
     * @param {Uint32Array} shape
     * @returns {WasmTensor}
     */
    static random(shape) {
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensor_random(ptr0, len0);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Reshape tensor
     * @param {Uint32Array} new_shape
     * @returns {WasmTensor}
     */
    reshape(new_shape) {
        const ptr0 = passArray32ToWasm0(new_shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensor_reshape(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get tensor size (total number of elements)
     * @returns {number}
     */
    size() {
        const ret = wasm.wasmtensor_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get tensor dimensions (number of axes)
     * @returns {number}
     */
    ndim() {
        const ret = wasm.wasmtensor_ndim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Transpose 2D tensor
     * @returns {WasmTensor}
     */
    transpose() {
        const ret = wasm.wasmtensor_transpose(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Element-wise subtraction
     * @param {WasmTensor} other
     * @returns {WasmTensor}
     */
    subtract(other) {
        _assertClass(other, WasmTensor);
        const ret = wasm.wasmtensor_subtract(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Element-wise division
     * @param {WasmTensor} other
     * @returns {WasmTensor}
     */
    divide(other) {
        _assertClass(other, WasmTensor);
        const ret = wasm.wasmtensor_divide(this.__wbg_ptr, other.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Scalar addition
     * @param {number} scalar
     * @returns {WasmTensor}
     */
    add_scalar(scalar) {
        const ret = wasm.wasmtensor_add_scalar(this.__wbg_ptr, scalar);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Scalar multiplication
     * @param {number} scalar
     * @returns {WasmTensor}
     */
    mul_scalar(scalar) {
        const ret = wasm.wasmtensor_mul_scalar(this.__wbg_ptr, scalar);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Power function
     * @param {number} exponent
     * @returns {WasmTensor}
     */
    pow(exponent) {
        const ret = wasm.wasmtensor_pow(this.__wbg_ptr, exponent);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Square root
     * @returns {WasmTensor}
     */
    sqrt() {
        const ret = wasm.wasmtensor_sqrt(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Exponential function
     * @returns {WasmTensor}
     */
    exp() {
        const ret = wasm.wasmtensor_exp(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Natural logarithm
     * @returns {WasmTensor}
     */
    log() {
        const ret = wasm.wasmtensor_log(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Sum all elements
     * @returns {number}
     */
    sum() {
        const ret = wasm.wasmtensor_sum(this.__wbg_ptr);
        return ret;
    }
    /**
     * Mean of all elements
     * @returns {number}
     */
    mean() {
        const ret = wasm.wasmtensor_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Maximum element
     * @returns {number}
     */
    max() {
        const ret = wasm.wasmtensor_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * Minimum element
     * @returns {number}
     */
    min() {
        const ret = wasm.wasmtensor_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * Tanh activation
     * @returns {WasmTensor}
     */
    tanh() {
        const ret = wasm.wasmtensor_tanh(this.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
}

const WasmTensorBufferFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtensorbuffer_free(ptr >>> 0, 1));
/**
 * Memory-aware tensor buffer for WASM
 */
export class WasmTensorBuffer {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmTensorBuffer.prototype);
        obj.__wbg_ptr = ptr;
        WasmTensorBufferFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTensorBufferFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtensorbuffer_free(ptr, 0);
    }
    /**
     * Create new tensor buffer
     * @param {Float32Array} data
     * @param {Uint32Array} shape
     */
    constructor(data, shape) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorbuffer_new(ptr0, len0, ptr1, len1);
        this.__wbg_ptr = ret >>> 0;
        WasmTensorBufferFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create tensor buffer from memory pool
     * @param {WasmTensorPool} pool
     * @param {Uint32Array} shape
     * @returns {WasmTensorBuffer | undefined}
     */
    static from_pool(pool, shape) {
        _assertClass(pool, WasmTensorPool);
        const ptr0 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorbuffer_from_pool(pool.__wbg_ptr, ptr0, len0);
        return ret === 0 ? undefined : WasmTensorBuffer.__wrap(ret);
    }
    /**
     * Get buffer data
     * @returns {Float32Array}
     */
    get data() {
        const ret = wasm.wasmtensorbuffer_data(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get buffer shape
     * @returns {Uint32Array}
     */
    get shape() {
        const ret = wasm.wasmtensorbuffer_shape(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get memory ID if allocated from pool
     * @returns {number | undefined}
     */
    get memory_id() {
        const ret = wasm.wasmtensorbuffer_memory_id(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Get buffer size in bytes
     * @returns {number}
     */
    size_bytes() {
        const ret = wasm.wasmtensorbuffer_size_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Release buffer back to pool
     * @param {WasmTensorPool} pool
     * @returns {boolean}
     */
    release_to_pool(pool) {
        _assertClass(pool, WasmTensorPool);
        const ret = wasm.wasmtensorbuffer_release_to_pool(this.__wbg_ptr, pool.__wbg_ptr);
        return ret !== 0;
    }
}

const WasmTensorOpsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtensorops_free(ptr >>> 0, 1));
/**
 * Advanced tensor operations for neural networks
 * ニューラルネットワーク用の高度なテンソル操作
 */
export class WasmTensorOps {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTensorOpsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtensorops_free(ptr, 0);
    }
    /**
     * Matrix multiplication: A @ B
     * 行列積: A @ B
     * @param {Float32Array} a
     * @param {number} a_rows
     * @param {number} a_cols
     * @param {Float32Array} b
     * @param {number} b_rows
     * @param {number} b_cols
     * @returns {Float32Array}
     */
    static matmul(a, a_rows, a_cols, b, b_rows, b_cols) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_matmul(ptr0, len0, a_rows, a_cols, ptr1, len1, b_rows, b_cols);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Transpose a 2D matrix
     * 2D行列の転置
     * @param {Float32Array} matrix
     * @param {number} rows
     * @param {number} cols
     * @returns {Float32Array}
     */
    static transpose(matrix, rows, cols) {
        const ptr0 = passArrayF32ToWasm0(matrix, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_transpose(ptr0, len0, rows, cols);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Reshape tensor while preserving total elements
     * 総要素数を保持しながらテンソルをリシェイプ
     * @param {Float32Array} data
     * @param {Uint32Array} new_shape
     * @returns {Float32Array}
     */
    static reshape(data, new_shape) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(new_shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_reshape(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Concatenate tensors along specified axis
     * 指定軸でテンソルを連結
     * @param {Array<any>} tensors
     * @param {Array<any>} shapes
     * @param {number} axis
     * @returns {object}
     */
    static concatenate(tensors, shapes, axis) {
        const ret = wasm.wasmtensorops_concatenate(tensors, shapes, axis);
        return ret;
    }
    /**
     * Split tensor along specified axis
     * 指定軸でテンソルを分割
     * @param {Float32Array} data
     * @param {Uint32Array} shape
     * @param {number} axis
     * @param {Uint32Array} split_sizes
     * @returns {Array<any>}
     */
    static split(data, shape, axis, split_sizes) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(split_sizes, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_split(ptr0, len0, ptr1, len1, axis, ptr2, len2);
        return ret;
    }
    /**
     * Compute tensor dot product (Einstein summation)
     * テンソル内積の計算（アインシュタイン記法）
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {number}
     */
    static dot_product(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_dot_product(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * Element-wise operations
     * 要素ごとの操作
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
    static element_wise_add(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_element_wise_add(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Element-wise multiplication
     * 要素ごとの乗算
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
    static element_wise_mul(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_element_wise_mul(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Element-wise subtraction
     * 要素ごとの減算
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
    static element_wise_sub(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_element_wise_sub(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Element-wise division
     * 要素ごとの除算
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {Float32Array}
     */
    static element_wise_div(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_element_wise_div(ptr0, len0, ptr1, len1);
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * Reduce operations
     * リダクション操作
     * @param {Float32Array} data
     * @param {number | null | undefined} axis
     * @param {Uint32Array} shape
     * @returns {object}
     */
    static reduce_sum(data, axis, shape) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_reduce_sum(ptr0, len0, isLikeNone(axis) ? 0x100000001 : (axis) >>> 0, ptr1, len1);
        return ret;
    }
    /**
     * Reduce mean
     * 平均値の計算
     * @param {Float32Array} data
     * @param {number | null | undefined} axis
     * @param {Uint32Array} shape
     * @returns {object}
     */
    static reduce_mean(data, axis, shape) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_reduce_mean(ptr0, len0, isLikeNone(axis) ? 0x100000001 : (axis) >>> 0, ptr1, len1);
        return ret;
    }
    /**
     * Broadcasting addition for tensors of different shapes
     * 異なる形状のテンソルのブロードキャスト加算
     * @param {Float32Array} a
     * @param {Uint32Array} a_shape
     * @param {Float32Array} b
     * @param {Uint32Array} b_shape
     * @returns {object}
     */
    static broadcast_add(a, a_shape, b, b_shape) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(a_shape, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArray32ToWasm0(b_shape, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_broadcast_add(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        return ret;
    }
    /**
     * Compute gradient clipping (useful for training)
     * 勾配クリッピングを計算（訓練に有用）
     * @param {Float32Array} gradients
     * @param {number} max_norm
     * @returns {Float32Array}
     */
    static clip_gradients(gradients, max_norm) {
        const ptr0 = passArrayF32ToWasm0(gradients, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_clip_gradients(ptr0, len0, max_norm);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply dropout during training (sets random elements to zero)
     * 訓練中のドロップアウトを適用（ランダム要素をゼロに設定）
     * @param {Float32Array} input
     * @param {number} dropout_rate
     * @param {boolean} training
     * @param {number} seed
     * @returns {Float32Array}
     */
    static dropout(input, dropout_rate, training, seed) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtensorops_dropout(ptr0, len0, dropout_rate, training, seed);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}

const WasmTensorPoolFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtensorpool_free(ptr >>> 0, 1));
/**
 * Memory pool for WASM tensor operations
 */
export class WasmTensorPool {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTensorPoolFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtensorpool_free(ptr, 0);
    }
    /**
     * Create new memory pool with specified capacity
     * @param {number} capacity_bytes
     */
    constructor(capacity_bytes) {
        const ret = wasm.wasmtensorpool_new(capacity_bytes);
        this.__wbg_ptr = ret >>> 0;
        WasmTensorPoolFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Allocate memory block
     * @param {number} size
     * @returns {number | undefined}
     */
    allocate(size) {
        const ret = wasm.wasmtensorpool_allocate(this.__wbg_ptr, size);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Deallocate memory block
     * @param {number} index
     * @returns {boolean}
     */
    deallocate(index) {
        const ret = wasm.wasmtensorpool_deallocate(this.__wbg_ptr, index);
        return ret !== 0;
    }
    /**
     * Get total allocated memory in elements
     * @returns {number}
     */
    get_total_allocated() {
        const ret = wasm.wasmtensorpool_get_total_allocated(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get memory usage statistics
     * @returns {object}
     */
    get_usage_stats() {
        const ret = wasm.wasmtensorpool_get_usage_stats(this.__wbg_ptr);
        return ret;
    }
    /**
     * Force garbage collection of unused blocks
     * @returns {number}
     */
    garbage_collect() {
        const ret = wasm.wasmtensorpool_garbage_collect(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Clear all allocated memory
     */
    clear() {
        wasm.wasmtensorpool_clear(this.__wbg_ptr);
    }
}

const WasmTensorSpecialFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtensorspecial_free(ptr >>> 0, 1));
/**
 * Tensor-based special functions for WASM
 */
export class WasmTensorSpecial {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTensorSpecialFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtensorspecial_free(ptr, 0);
    }
    /**
     * Apply gamma function to tensor elements
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    static tensor_gamma(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmtensorspecial_tensor_gamma(tensor.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Apply lgamma function to tensor elements
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    static tensor_lgamma(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmtensorspecial_tensor_lgamma(tensor.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Apply erf function to tensor elements
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    static tensor_erf(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmtensorspecial_tensor_erf(tensor.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
    /**
     * Apply bessel_j0 function to tensor elements
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    static tensor_bessel_j0(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmtensorspecial_tensor_bessel_j0(tensor.__wbg_ptr);
        return WasmTensor.__wrap(ret);
    }
}

const WasmTimeSeriesDetectorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtimeseriesdetector_free(ptr >>> 0, 1));
/**
 * WASM wrapper for Time Series Anomaly Detector
 */
export class WasmTimeSeriesDetector {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmTimeSeriesDetector.prototype);
        obj.__wbg_ptr = ptr;
        WasmTimeSeriesDetectorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTimeSeriesDetectorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtimeseriesdetector_free(ptr, 0);
    }
    /**
     * Create new time series anomaly detector
     * @param {number} window_size
     * @param {number | null} [seasonal_period]
     */
    constructor(window_size, seasonal_period) {
        const ret = wasm.wasmtimeseriesdetector_new(window_size, isLikeNone(seasonal_period) ? 0x100000001 : (seasonal_period) >>> 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        WasmTimeSeriesDetectorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add new data point and check for anomalies
     * @param {number} timestamp
     * @param {number} value
     * @returns {any}
     */
    add_point(timestamp, value) {
        const ret = wasm.wasmtimeseriesdetector_add_point(this.__wbg_ptr, timestamp, value);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Get trend analysis
     * @returns {string}
     */
    get_trend_analysis() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmtimeseriesdetector_get_trend_analysis(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Get seasonal analysis
     * @returns {string}
     */
    get_seasonal_analysis() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmtimeseriesdetector_get_seasonal_analysis(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
}

const WasmToTensorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtotensor_free(ptr >>> 0, 1));
/**
 * WASM wrapper for ToTensor transformation
 */
export class WasmToTensor {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmToTensorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtotensor_free(ptr, 0);
    }
    /**
     * Create new to tensor transform
     */
    constructor() {
        const ret = wasm.wasmrelu_new();
        this.__wbg_ptr = ret >>> 0;
        WasmToTensorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Apply to tensor transformation (identity operation)
     * @param {WasmTensor} tensor
     * @returns {WasmTensor}
     */
    apply(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.wasmtotensor_apply(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get transformation name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmtotensor_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmTransformPipelineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtransformpipeline_free(ptr >>> 0, 1));
/**
 * Simple pipeline for chaining transformations
 */
export class WasmTransformPipeline {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTransformPipelineFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtransformpipeline_free(ptr, 0);
    }
    /**
     * Create new pipeline
     * @param {boolean} cache_enabled
     */
    constructor(cache_enabled) {
        const ret = wasm.wasmprocessingpipeline_new(cache_enabled);
        this.__wbg_ptr = ret >>> 0;
        WasmTransformPipelineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add transform to pipeline
     * @param {string} transform_name
     */
    add_transform(transform_name) {
        const ptr0 = passStringToWasm0(transform_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmtransformpipeline_add_transform(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get number of transforms in pipeline
     * @returns {number}
     */
    length() {
        const ret = wasm.wasmmodel_num_layers(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Clear all transforms
     */
    clear() {
        wasm.wasmtransformpipeline_clear(this.__wbg_ptr);
    }
    /**
     * Execute pipeline on tensor (simplified)
     * @param {WasmTensor} input
     * @returns {WasmTensor}
     */
    execute(input) {
        _assertClass(input, WasmTensor);
        const ret = wasm.wasmtransformpipeline_execute(this.__wbg_ptr, input.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmTensor.__wrap(ret[0]);
    }
    /**
     * Get pipeline statistics
     * @returns {string}
     */
    get_stats() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmtransformpipeline_get_stats(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}

const WasmUniformFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmuniform_free(ptr >>> 0, 1));
/**
 * Uniform distribution for WASM
 * WASM用一様分布
 */
export class WasmUniform {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmUniform.prototype);
        obj.__wbg_ptr = ptr;
        WasmUniformFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmUniformFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmuniform_free(ptr, 0);
    }
    /**
     * Create new uniform distribution
     * @param {number} low
     * @param {number} high
     * @param {number} seed
     */
    constructor(low, high, seed) {
        const ret = wasm.wasmuniform_new(low, high, seed);
        this.__wbg_ptr = ret >>> 0;
        WasmUniformFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create standard uniform distribution [0, 1)
     * @param {number} seed
     * @returns {WasmUniform}
     */
    static standard(seed) {
        const ret = wasm.wasmuniform_standard(seed);
        return WasmUniform.__wrap(ret);
    }
    /**
     * Sample single value
     * @returns {number}
     */
    sample() {
        const ret = wasm.wasmuniform_sample(this.__wbg_ptr);
        return ret;
    }
    /**
     * Sample multiple values
     * @param {number} n
     * @returns {Float32Array}
     */
    sample_n(n) {
        const ret = wasm.wasmuniform_sample_n(this.__wbg_ptr, n);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Probability density function
     * @param {number} x
     * @returns {number}
     */
    pdf(x) {
        const ret = wasm.wasmuniform_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Log probability density function
     * @param {number} x
     * @returns {number}
     */
    log_pdf(x) {
        const ret = wasm.wasmuniform_log_pdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Cumulative distribution function
     * @param {number} x
     * @returns {number}
     */
    cdf(x) {
        const ret = wasm.wasmuniform_cdf(this.__wbg_ptr, x);
        return ret;
    }
    /**
     * Get mean
     * @returns {number}
     */
    mean() {
        const ret = wasm.wasmuniform_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get variance
     * @returns {number}
     */
    variance() {
        const ret = wasm.wasmuniform_variance(this.__wbg_ptr);
        return ret;
    }
}

const WasmVisionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmvision_free(ptr >>> 0, 1));
/**
 * Vision utilities for WASM
 * WASM用画像処理ユーティリティ
 */
export class WasmVision {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmVisionFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmvision_free(ptr, 0);
    }
    /**
     * Resize image using bilinear interpolation
     * バイリニア補間による画像リサイズ
     * @param {Float32Array} image_data
     * @param {number} original_height
     * @param {number} original_width
     * @param {number} new_height
     * @param {number} new_width
     * @param {number} channels
     * @returns {Float32Array}
     */
    static resize(image_data, original_height, original_width, new_height, new_width, channels) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_resize(ptr0, len0, original_height, original_width, new_height, new_width, channels);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Normalize image with mean and standard deviation
     * 平均と標準偏差による画像正規化
     * @param {Float32Array} image_data
     * @param {Float32Array} mean
     * @param {Float32Array} std
     * @param {number} channels
     * @returns {Float32Array}
     */
    static normalize(image_data, mean, std, channels) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(mean, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(std, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_normalize(ptr0, len0, ptr1, len1, ptr2, len2, channels);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * Convert RGB to grayscale
     * RGBからグレースケールに変換
     * @param {Float32Array} rgb_data
     * @param {number} height
     * @param {number} width
     * @returns {Float32Array}
     */
    static rgb_to_grayscale(rgb_data, height, width) {
        const ptr0 = passArrayF32ToWasm0(rgb_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_rgb_to_grayscale(ptr0, len0, height, width);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply Gaussian blur
     * ガウシアンブラーを適用
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @param {number} sigma
     * @returns {Float32Array}
     */
    static gaussian_blur(image_data, height, width, channels, sigma) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_gaussian_blur(ptr0, len0, height, width, channels, sigma);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Crop image to specified region
     * 指定領域に画像をクロップ
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @param {number} start_y
     * @param {number} start_x
     * @param {number} crop_height
     * @param {number} crop_width
     * @returns {Float32Array}
     */
    static crop(image_data, height, width, channels, start_y, start_x, crop_height, crop_width) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_crop(ptr0, len0, height, width, channels, start_y, start_x, crop_height, crop_width);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Flip image horizontally
     * 画像を水平反転
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @returns {Float32Array}
     */
    static flip_horizontal(image_data, height, width, channels) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_flip_horizontal(ptr0, len0, height, width, channels);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Flip image vertically
     * 画像を垂直反転
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @returns {Float32Array}
     */
    static flip_vertical(image_data, height, width, channels) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_flip_vertical(ptr0, len0, height, width, channels);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Rotate image by 90 degrees clockwise
     * 画像を時計回りに90度回転
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @returns {Float32Array}
     */
    static rotate_90_cw(image_data, height, width, channels) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_rotate_90_cw(ptr0, len0, height, width, channels);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply center crop (crop from center of image)
     * センタークロップ（画像中央からクロップ）
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @param {number} crop_size
     * @returns {Float32Array}
     */
    static center_crop(image_data, height, width, channels, crop_size) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_center_crop(ptr0, len0, height, width, channels, crop_size);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Adjust image brightness
     * 画像の明度を調整
     * @param {Float32Array} image_data
     * @param {number} factor
     * @returns {Float32Array}
     */
    static adjust_brightness(image_data, factor) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_adjust_brightness(ptr0, len0, factor);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Adjust image contrast
     * 画像のコントラストを調整
     * @param {Float32Array} image_data
     * @param {number} factor
     * @returns {Float32Array}
     */
    static adjust_contrast(image_data, factor) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_adjust_contrast(ptr0, len0, factor);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Add Gaussian noise to image (data augmentation)
     * 画像にガウシアンノイズを追加（データ拡張）
     * @param {Float32Array} image_data
     * @param {number} std_dev
     * @returns {Float32Array}
     */
    static add_gaussian_noise(image_data, std_dev) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_add_gaussian_noise(ptr0, len0, std_dev);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply random rotation (for data augmentation)
     * ランダム回転を適用（データ拡張用）
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @param {number} max_angle_deg
     * @returns {Float32Array}
     */
    static random_rotation(image_data, height, width, channels, max_angle_deg) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_random_rotation(ptr0, len0, height, width, channels, max_angle_deg);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply edge detection (Sobel filter)
     * エッジ検出（Sobelフィルター）
     * @param {Float32Array} image_data
     * @param {number} height
     * @param {number} width
     * @returns {Float32Array}
     */
    static edge_detection(image_data, height, width) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_edge_detection(ptr0, len0, height, width);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Convert image from 0-255 range to 0-1 range
     * 画像を0-255範囲から0-1範囲に変換
     * @param {Uint8Array} image_data
     * @returns {Float32Array}
     */
    static to_float(image_data) {
        const ptr0 = passArray8ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_to_float(ptr0, len0);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Convert image from 0-1 range to 0-255 range
     * 画像を0-1範囲から0-255範囲に変換
     * @param {Float32Array} image_data
     * @returns {Uint8Array}
     */
    static to_uint8(image_data) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_to_uint8(ptr0, len0);
        var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v2;
    }
    /**
     * Calculate image histogram
     * 画像のヒストグラムを計算
     * @param {Float32Array} image_data
     * @param {number} bins
     * @returns {Uint32Array}
     */
    static histogram(image_data, bins) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_histogram(ptr0, len0, bins);
        var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Apply histogram equalization
     * ヒストグラム均等化を適用
     * @param {Float32Array} image_data
     * @param {number} bins
     * @returns {Float32Array}
     */
    static histogram_equalization(image_data, bins) {
        const ptr0 = passArrayF32ToWasm0(image_data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmvision_histogram_equalization(ptr0, len0, bins);
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}

const WorkerManagerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_workermanager_free(ptr >>> 0, 1));
/**
 * Web Worker utilities for background computation
 */
export class WorkerManager {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WorkerManagerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_workermanager_free(ptr, 0);
    }
    /**
     * Create new web worker manager
     */
    constructor() {
        const ret = wasm.workermanager_new();
        this.__wbg_ptr = ret >>> 0;
        WorkerManagerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create and start a web worker
     * @param {string} script_url
     */
    create_worker(script_url) {
        const ptr0 = passStringToWasm0(script_url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.workermanager_create_worker(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Send tensor data to worker
     * @param {WasmTensor} tensor
     */
    send_tensor(tensor) {
        _assertClass(tensor, WasmTensor);
        const ret = wasm.workermanager_send_tensor(this.__wbg_ptr, tensor.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Terminate worker
     */
    terminate() {
        wasm.workermanager_terminate(this.__wbg_ptr);
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_buffer_609cc3eee51ed158 = function(arg0) {
        const ret = arg0.buffer;
        return ret;
    };
    imports.wbg.__wbg_call_672a4d21634d4a24 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_call_7cccdd69e0791ae2 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.call(arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_clearRect_8e4ba7ea0e06711a = function(arg0, arg1, arg2, arg3, arg4) {
        arg0.clearRect(arg1, arg2, arg3, arg4);
    };
    imports.wbg.__wbg_createElement_8c9931a732ee2fea = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.createElement(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_crypto_574e78ad8b13b65f = function(arg0) {
        const ret = arg0.crypto;
        return ret;
    };
    imports.wbg.__wbg_debug_3cb59063b29f58c1 = function(arg0) {
        console.debug(arg0);
    };
    imports.wbg.__wbg_document_d249400bd7bd996d = function(arg0) {
        const ret = arg0.document;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_error_524f506f44df1645 = function(arg0) {
        console.error(arg0);
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_fillRect_c38d5d56492a2368 = function(arg0, arg1, arg2, arg3, arg4) {
        arg0.fillRect(arg1, arg2, arg3, arg4);
    };
    imports.wbg.__wbg_from_2a5d3e218e67aa85 = function(arg0) {
        const ret = Array.from(arg0);
        return ret;
    };
    imports.wbg.__wbg_getContext_e9cf379449413580 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    }, arguments) };
    imports.wbg.__wbg_getElementById_f827f0d6648718a8 = function(arg0, arg1, arg2) {
        const ret = arg0.getElementById(getStringFromWasm0(arg1, arg2));
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_getItem_17f98dee3b43fa7e = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        const ret = arg1.getItem(getStringFromWasm0(arg2, arg3));
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    }, arguments) };
    imports.wbg.__wbg_getRandomValues_b8f5dbd5f3995a9e = function() { return handleError(function (arg0, arg1) {
        arg0.getRandomValues(arg1);
    }, arguments) };
    imports.wbg.__wbg_get_67b2ba62fc30de12 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_get_b9b93047fe3cf45b = function(arg0, arg1) {
        const ret = arg0[arg1 >>> 0];
        return ret;
    };
    imports.wbg.__wbg_has_a5ea9117f258a0ec = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.has(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_height_838cee19ba8597db = function(arg0) {
        const ret = arg0.height;
        return ret;
    };
    imports.wbg.__wbg_instanceof_CanvasRenderingContext2d_df82a4d3437bf1cc = function(arg0) {
        let result;
        try {
            result = arg0 instanceof CanvasRenderingContext2D;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_HtmlCanvasElement_2ea67072a7624ac5 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof HTMLCanvasElement;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_HtmlInputElement_12d71bf2d15dd19e = function(arg0) {
        let result;
        try {
            result = arg0 instanceof HTMLInputElement;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Window_def73ea0955fc569 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Window;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_a1eab7e0d067391b = function(arg0) {
        const ret = Array.isArray(arg0);
        return ret;
    };
    imports.wbg.__wbg_key_c5e0a01cf450dca2 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg1.key(arg2 >>> 0);
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    }, arguments) };
    imports.wbg.__wbg_length_3b4f022188ae8db6 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_e2d2a49132c1b256 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_ed4a84b02b798bda = function() { return handleError(function (arg0) {
        const ret = arg0.length;
        return ret;
    }, arguments) };
    imports.wbg.__wbg_localStorage_1406c99c39728187 = function() { return handleError(function (arg0) {
        const ret = arg0.localStorage;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    }, arguments) };
    imports.wbg.__wbg_log_1ae1e9f741096e91 = function(arg0, arg1) {
        console.log(arg0, arg1);
    };
    imports.wbg.__wbg_log_c222819a41e063d3 = function(arg0) {
        console.log(arg0);
    };
    imports.wbg.__wbg_msCrypto_a61aeb35a24c1329 = function(arg0) {
        const ret = arg0.msCrypto;
        return ret;
    };
    imports.wbg.__wbg_new_405e22f390576ce2 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_new_780abee5c1739fd7 = function(arg0) {
        const ret = new Float32Array(arg0);
        return ret;
    };
    imports.wbg.__wbg_new_78feb108b6472713 = function() {
        const ret = new Array();
        return ret;
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return ret;
    };
    imports.wbg.__wbg_new_a12002a7f91c75be = function(arg0) {
        const ret = new Uint8Array(arg0);
        return ret;
    };
    imports.wbg.__wbg_new_b1a33e5095abf678 = function() { return handleError(function (arg0, arg1) {
        const ret = new Worker(getStringFromWasm0(arg0, arg1));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_newnoargs_105ed471475aaf50 = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_newwithbyteoffsetandlength_d97e637ebe145a9a = function(arg0, arg1, arg2) {
        const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_newwithbyteoffsetandlength_e6b7e69acd4c7354 = function(arg0, arg1, arg2) {
        const ret = new Float32Array(arg0, arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_newwithlength_a381634e90c276d4 = function(arg0) {
        const ret = new Uint8Array(arg0 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_node_905d3e251edff8a2 = function(arg0) {
        const ret = arg0.node;
        return ret;
    };
    imports.wbg.__wbg_now_807e54c39636c349 = function() {
        const ret = Date.now();
        return ret;
    };
    imports.wbg.__wbg_now_d18023d54d4e5500 = function(arg0) {
        const ret = arg0.now();
        return ret;
    };
    imports.wbg.__wbg_performance_c185c0cdc2766575 = function(arg0) {
        const ret = arg0.performance;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_postMessage_6edafa8f7b9c2f52 = function() { return handleError(function (arg0, arg1) {
        arg0.postMessage(arg1);
    }, arguments) };
    imports.wbg.__wbg_process_dc0fbacc7c1c06f7 = function(arg0) {
        const ret = arg0.process;
        return ret;
    };
    imports.wbg.__wbg_push_737cfc8c1432c2c6 = function(arg0, arg1) {
        const ret = arg0.push(arg1);
        return ret;
    };
    imports.wbg.__wbg_randomFillSync_ac0988aba3254290 = function() { return handleError(function (arg0, arg1) {
        arg0.randomFillSync(arg1);
    }, arguments) };
    imports.wbg.__wbg_random_3ad904d98382defe = function() {
        const ret = Math.random();
        return ret;
    };
    imports.wbg.__wbg_removeItem_9d2669ee3bba6f7d = function() { return handleError(function (arg0, arg1, arg2) {
        arg0.removeItem(getStringFromWasm0(arg1, arg2));
    }, arguments) };
    imports.wbg.__wbg_require_60cc747a6bc5215a = function() { return handleError(function () {
        const ret = module.require;
        return ret;
    }, arguments) };
    imports.wbg.__wbg_setItem_212ecc915942ab0a = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.setItem(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
    }, arguments) };
    imports.wbg.__wbg_set_10bad9bee0e9c58b = function(arg0, arg1, arg2) {
        arg0.set(arg1, arg2 >>> 0);
    };
    imports.wbg.__wbg_set_65595bdd868b3009 = function(arg0, arg1, arg2) {
        arg0.set(arg1, arg2 >>> 0);
    };
    imports.wbg.__wbg_set_bb8cecf6a62b9f46 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = Reflect.set(arg0, arg1, arg2);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_setaccept_ff32b9ffcfbd061d = function(arg0, arg1, arg2) {
        arg0.accept = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_setfillStyle_2205fca942c641ba = function(arg0, arg1, arg2) {
        arg0.fillStyle = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_setheight_da683a33fa99843c = function(arg0, arg1) {
        arg0.height = arg1 >>> 0;
    };
    imports.wbg.__wbg_setlength_a668e53981184590 = function(arg0, arg1) {
        arg0.length = arg1 >>> 0;
    };
    imports.wbg.__wbg_setmultiple_1b3b3f243cda56b2 = function(arg0, arg1) {
        arg0.multiple = arg1 !== 0;
    };
    imports.wbg.__wbg_settype_2a902a4a235bb64a = function(arg0, arg1, arg2) {
        arg0.type = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_setwidth_c5fed9f5e7f0b406 = function(arg0, arg1) {
        arg0.width = arg1 >>> 0;
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = arg1.stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_88a902d13a557d07 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_56578be7e9f832b0 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_37c5d418e4bf5819 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_5de37043a91a9c40 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_subarray_aa9065fa9dc5df96 = function(arg0, arg1, arg2) {
        const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_terminate_e8eab2977ce01111 = function(arg0) {
        arg0.terminate();
    };
    imports.wbg.__wbg_timeEnd_c619922f7c81b96d = function(arg0, arg1) {
        console.timeEnd(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_time_45bf36fd575512a4 = function(arg0, arg1) {
        console.time(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_versions_c01dfd4722a88165 = function(arg0) {
        const ret = arg0.versions;
        return ret;
    };
    imports.wbg.__wbg_warn_4ca3906c248c47c4 = function(arg0) {
        console.warn(arg0);
    };
    imports.wbg.__wbg_width_5dde457d606ba683 = function(arg0) {
        const ret = arg0.width;
        return ret;
    };
    imports.wbg.__wbindgen_copy_to_typed_array = function(arg0, arg1, arg2) {
        new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
    };
    imports.wbg.__wbindgen_debug_string = function(arg0, arg1) {
        const ret = debugString(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_float32_array_new = function(arg0, arg1) {
        var v0 = getArrayF32FromWasm0(arg0, arg1).slice();
        wasm.__wbindgen_free(arg0, arg1 * 4, 4);
        const ret = v0;
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_export_2;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };
    imports.wbg.__wbindgen_is_function = function(arg0) {
        const ret = typeof(arg0) === 'function';
        return ret;
    };
    imports.wbg.__wbindgen_is_object = function(arg0) {
        const val = arg0;
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbindgen_is_string = function(arg0) {
        const ret = typeof(arg0) === 'string';
        return ret;
    };
    imports.wbg.__wbindgen_is_undefined = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbindgen_memory = function() {
        const ret = wasm.memory;
        return ret;
    };
    imports.wbg.__wbindgen_number_get = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbindgen_number_new = function(arg0) {
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('rustorch_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
