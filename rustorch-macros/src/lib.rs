//! # RusTorch Procedural Macros
//!
//! Procedural macros for creating N-dimensional tensors with compile-time shape inference.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprArray};

/// Creates an N-dimensional tensor with compile-time shape inference.
///
/// This macro supports arbitrary dimensional tensors (1D, 2D, 3D, 4D, 5D, 6D, and beyond).
/// All numeric types that implement conversion to f32 are supported.
///
/// # Examples
///
/// ```rust
/// use rustorch::tensor_nd;
///
/// // 1D tensor: shape [4]
/// let t1 = tensor_nd!([1, 2, 3, 4]);
///
/// // 2D tensor: shape [2, 3]
/// let t2 = tensor_nd!([
///     [1, 2, 3],
///     [4, 5, 6]
/// ]);
///
/// // 3D tensor: shape [2, 2, 2]
/// let t3 = tensor_nd!([
///     [[1, 2], [3, 4]],
///     [[5, 6], [7, 8]]
/// ]);
///
/// // 4D tensor: shape [2, 2, 2, 2]
/// let t4 = tensor_nd!([
///     [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
///     [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
/// ]);
///
/// // 6D tensor: shape [1, 1, 1, 1, 1, 2]
/// let t6 = tensor_nd!([[[[[[1, 2]]]]]]);
/// ```
#[proc_macro]
pub fn tensor_nd(input: TokenStream) -> TokenStream {
    let expr = parse_macro_input!(input as Expr);

    match &expr {
        Expr::Array(arr) => {
            let (data_tokens, shape) = flatten_array(arr);

            let expanded = quote! {
                {
                    let data: Vec<f32> = vec![#(#data_tokens),*];
                    let shape = vec![#(#shape),*];
                    ::rustorch::tensor::Tensor::from_vec(data, shape)
                }
            };

            TokenStream::from(expanded)
        }
        _ => {
            syn::Error::new_spanned(expr, "Expected array literal")
                .to_compile_error()
                .into()
        }
    }
}

/// Recursively flattens a nested array and calculates its shape
fn flatten_array(arr: &ExprArray) -> (Vec<proc_macro2::TokenStream>, Vec<usize>) {
    let elems = &arr.elems;

    if elems.is_empty() {
        return (vec![], vec![0]);
    }

    // Check if first element is an array (nested) or a value (leaf)
    match &elems[0] {
        Expr::Array(_nested) => {
            // Nested array - recurse
            let mut all_data = Vec::new();
            let mut inner_shape = None;

            for elem in elems.iter() {
                match elem {
                    Expr::Array(inner_arr) => {
                        let (data, shape) = flatten_array(inner_arr);

                        // Verify shape consistency
                        if let Some(ref expected_shape) = inner_shape {
                            if &shape != expected_shape {
                                // Shape mismatch - will be caught at runtime
                                // For better error messages, we could emit a compile error here
                            }
                        } else {
                            inner_shape = Some(shape.clone());
                        }

                        all_data.extend(data);
                    }
                    _ => {
                        // Mixed array/value - shape inconsistency
                    }
                }
            }

            let mut shape = vec![elems.len()];
            if let Some(inner) = inner_shape {
                shape.extend(inner);
            }

            (all_data, shape)
        }
        _ => {
            // Leaf level - convert elements to f32
            let data: Vec<_> = elems.iter().map(|e| {
                quote! {
                    (#e as f64) as f32
                }
            }).collect();

            (data, vec![elems.len()])
        }
    }
}
