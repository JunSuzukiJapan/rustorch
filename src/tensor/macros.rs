//! # Tensor Creation Macros
//! テンソル作成マクロ
//!
//! This module provides convenient macros for creating tensors with literal syntax.
//! Supports 1D, 2D, and 3D tensor initialization with automatic shape inference.
//!
//! ## Examples
//!
//! ```rust
//! use rustorch::tensor;
//!
//! // 1D tensor
//! let t1 = tensor!([1, 2, 3, 4]);
//!
//! // 2D tensor
//! let t2 = tensor!([
//!     [1, 2, 3],
//!     [4, 5, 6]
//! ]);
//!
//! // 3D tensor
//! let t3 = tensor!([
//!     [[1, 2], [3, 4]],
//!     [[5, 6], [7, 8]]
//! ]);
//! ```

/// Convenient macro for creating tensors with literal syntax
///
/// This macro supports 1D, 2D, and 3D tensor creation with automatic shape inference.
/// All numeric types that implement `ToPrimitive` are supported and will be converted to f32.
///
/// # Examples
///
/// ```rust
/// use rustorch::tensor;
///
/// // 1D tensor: shape [4]
/// let t1 = tensor!([1, 2, 3, 4]);
///
/// // 2D tensor: shape [2, 3]
/// let t2 = tensor!([
///     [1, 2, 3],
///     [4, 5, 6]
/// ]);
///
/// // 3D tensor: shape [2, 2, 2]
/// let t3 = tensor!([
///     [[1, 2], [3, 4]],
///     [[5, 6], [7, 8]]
/// ]);
/// ```
#[macro_export]
macro_rules! tensor {
    // 3次元テンソル: [[[x, ...], ...], ...]
    ( [ $( [ $( [ $( $x:expr ),* ] ),* ] ),+ ] ) => {
        {
            use ::num_traits::ToPrimitive;
            let data: Vec<f32> = vec![
                $(
                    $(
                        $( ToPrimitive::to_f32(&$x).expect("Failed to convert to f32") ),*
                    ),*
                ),+
            ];

            // Shape calculation for 3D
            let dim0 = $crate::tensor!(@count [ $( [ $( [ $( $x ),* ] ),* ] ),+ ]);
            let dim1 = {
                let first = $crate::tensor!(@count_first_2d [ $( [ $( [ $( $x ),* ] ),* ] ),+ ]);
                first
            };
            let dim2 = {
                let first = $crate::tensor!(@count_first_1d [ $( [ $( [ $( $x ),* ] ),* ] ),+ ]);
                first
            };

            let shape = vec![dim0, dim1, dim2];
            $crate::tensor::Tensor::from_vec(data, shape)
        }
    };

    // 2次元テンソル: [[x, ...], ...]
    ( [ $( [ $( $x:expr ),* ] ),+ ] ) => {
        {
            use ::num_traits::ToPrimitive;
            let data: Vec<f32> = vec![
                $(
                    $( ToPrimitive::to_f32(&$x).expect("Failed to convert to f32") ),*
                ),+
            ];

            // Shape calculation for 2D
            let rows = $crate::tensor!(@count [ $( [ $( $x ),* ] ),+ ]);
            let cols = $crate::tensor!(@count_first_1d [ $( [ $( $x ),* ] ),+ ]);

            let shape = vec![rows, cols];
            $crate::tensor::Tensor::from_vec(data, shape)
        }
    };

    // 1次元テンソル: [x, ...]
    ( [ $( $x:expr ),+ ] ) => {
        {
            use ::num_traits::ToPrimitive;
            let data: Vec<f32> = vec![
                $( ToPrimitive::to_f32(&$x).expect("Failed to convert to f32") ),+
            ];
            let shape = vec![data.len()];
            $crate::tensor::Tensor::from_vec(data, shape)
        }
    };

    // ヘルパー: 配列の要素数をカウント
    (@count [ $( $x:tt ),+ ]) => {
        {
            let count = 0 $( + { let _ = stringify!($x); 1 } )+;
            count
        }
    };

    // ヘルパー: 2次元配列の最初の行の要素数をカウント
    (@count_first_2d [ [ $( [ $( $x:tt ),* ] ),+ ] $(, $_rest:tt )* ]) => {
        {
            let count = 0 $( + { let _ = stringify!([ $( $x ),* ]); 1 } )+;
            count
        }
    };

    // ヘルパー: 配列の最初の要素の長さをカウント
    (@count_first_1d [ [ $( $x:tt ),+ ] $(, $_rest:tt )* ]) => {
        {
            let count = 0 $( + { let _ = stringify!($x); 1 } )+;
            count
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_tensor_macro_1d() {
        let t = tensor!([1, 2, 3, 4]);
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_tensor_macro_2d() {
        let t = tensor!([
            [1, 2, 3],
            [4, 5, 6]
        ]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]));
    }

    #[test]
    fn test_tensor_macro_3d() {
        let t = tensor!([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]);
        assert_eq!(t.shape(), &[2, 2, 2]);
        assert_eq!(t.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][..]));
    }

    #[test]
    fn test_tensor_macro_mixed_types() {
        let t = tensor!([1, 2.5, 3, 4.2]);
        assert_eq!(t.shape(), &[4]);
        assert_eq!(t.as_slice(), Some(&[1.0, 2.5, 3.0, 4.2][..]));
    }
}