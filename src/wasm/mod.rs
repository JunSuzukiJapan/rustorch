//! WebAssembly bindings for RusTorch
//! RusTorch用WebAssemblyバインディング

#[cfg(feature = "wasm")]
pub mod tensor;
#[cfg(feature = "wasm")]
pub mod bindings;
#[cfg(feature = "wasm")]
pub mod interop;
#[cfg(feature = "wasm")]
pub mod browser;
#[cfg(feature = "wasm")]
pub mod optimized;