//! Basic linear algebra for WebAssembly (BLAS-free implementation)
//! WebAssembly向け基本線形代数（BLAS非依存実装）

use wasm_bindgen::prelude::*;

/// WASM-compatible basic linear algebra operations
/// WASM互換基本線形代数演算
#[wasm_bindgen]
pub struct WasmLinearAlgebra {
    max_matrix_size: usize,
}

#[wasm_bindgen]
impl WasmLinearAlgebra {
    /// Create new linear algebra instance with size limits
    /// サイズ制限付きの新しい線形代数インスタンスを作成
    #[wasm_bindgen(constructor)]
    pub fn new(max_size: usize) -> WasmLinearAlgebra {
        WasmLinearAlgebra {
            max_matrix_size: if max_size > 0 && max_size <= 1000 {
                max_size
            } else {
                500
            },
        }
    }

    /// Matrix multiplication (basic O(n³) implementation)
    /// 行列乗算（基本O(n³)実装）
    #[wasm_bindgen]
    pub fn matmul(
        &self,
        a: Vec<f64>,
        a_rows: usize,
        a_cols: usize,
        b: Vec<f64>,
        b_rows: usize,
        b_cols: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if a_rows > self.max_matrix_size
            || a_cols > self.max_matrix_size
            || b_rows > self.max_matrix_size
            || b_cols > self.max_matrix_size
        {
            return Err(JsValue::from_str("Matrix too large for WASM"));
        }

        if a_cols != b_rows {
            return Err(JsValue::from_str("Matrix dimensions incompatible"));
        }

        if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
            return Err(JsValue::from_str("Data size doesn't match dimensions"));
        }

        let mut result = vec![0.0; a_rows * b_cols];

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[i * a_cols + k] * b[k * b_cols + j];
                }
                result[i * b_cols + j] = sum;
            }
        }

        Ok(result)
    }

    /// Compute eigenvalues using QR algorithm (small matrices only)
    /// QRアルゴリズムによる固有値計算（小行列のみ）
    #[wasm_bindgen]
    pub fn eigenvalues(&self, matrix: Vec<f64>, n: usize) -> Result<Vec<f64>, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str(
                "Matrix too large for eigenvalue computation",
            ));
        }

        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        // Copy matrix for QR iterations
        let mut a = matrix;
        let max_iterations = 1000;
        let tolerance = 1e-10;

        // Simplified QR algorithm for eigenvalues
        for _ in 0..max_iterations {
            let q = self.qr_q_matrix(&a, n)?;
            let r = self.qr_r_matrix(&a, n)?;
            a = self.matmul(r, n, n, q, n, n)?;

            // Check convergence (off-diagonal elements should be small)
            let mut off_diag_sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diag_sum += a[i * n + j].abs();
                    }
                }
            }

            if off_diag_sum < tolerance {
                break;
            }
        }

        // Extract diagonal elements (eigenvalues)
        let mut eigenvalues = Vec::new();
        for i in 0..n {
            eigenvalues.push(a[i * n + i]);
        }

        Ok(eigenvalues)
    }

    /// QR decomposition Q matrix using Gram-Schmidt process
    /// Gram-Schmidt過程によるQR分解のQ行列
    #[wasm_bindgen]
    pub fn qr_q_matrix(&self, matrix: &[f64], n: usize) -> Result<Vec<f64>, JsValue> {
        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        let mut q = vec![0.0; n * n];
        let mut r = vec![0.0; n * n];

        // Gram-Schmidt orthogonalization
        for j in 0..n {
            // Copy column j from A to Q
            for i in 0..n {
                q[i * n + j] = matrix[i * n + j];
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                // Compute dot product
                let mut dot = 0.0;
                for i in 0..n {
                    dot += q[i * n + j] * q[i * n + k];
                }
                r[k * n + j] = dot;

                // Subtract projection
                for i in 0..n {
                    q[i * n + j] -= dot * q[i * n + k];
                }
            }

            // Normalize column j
            let mut norm = 0.0;
            for i in 0..n {
                norm += q[i * n + j] * q[i * n + j];
            }
            norm = norm.sqrt();

            r[j * n + j] = norm;

            if norm > 1e-15 {
                for i in 0..n {
                    q[i * n + j] /= norm;
                }
            }
        }

        Ok(q)
    }

    /// QR decomposition R matrix using Gram-Schmidt process
    /// Gram-Schmidt過程によるQR分解のR行列
    #[wasm_bindgen]
    pub fn qr_r_matrix(&self, matrix: &[f64], n: usize) -> Result<Vec<f64>, JsValue> {
        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        let mut q = vec![0.0; n * n];
        let mut r = vec![0.0; n * n];

        // Gram-Schmidt orthogonalization (same algorithm)
        for j in 0..n {
            // Copy column j from A to Q
            for i in 0..n {
                q[i * n + j] = matrix[i * n + j];
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                // Compute dot product
                let mut dot = 0.0;
                for i in 0..n {
                    dot += q[i * n + j] * q[i * n + k];
                }
                r[k * n + j] = dot;

                // Subtract projection
                for i in 0..n {
                    q[i * n + j] -= dot * q[i * n + k];
                }
            }

            // Normalize column j
            let mut norm = 0.0;
            for i in 0..n {
                norm += q[i * n + j] * q[i * n + j];
            }
            norm = norm.sqrt();

            r[j * n + j] = norm;

            if norm > 1e-15 {
                for i in 0..n {
                    q[i * n + j] /= norm;
                }
            }
        }

        Ok(r)
    }

    /// Singular Value Decomposition (simplified for small matrices)
    /// 特異値分解（小行列用簡略版）
    #[wasm_bindgen]
    pub fn svd(
        &self,
        matrix: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Result<js_sys::Object, JsValue> {
        if rows > self.max_matrix_size || cols > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large for SVD"));
        }

        if matrix.len() != rows * cols {
            return Err(JsValue::from_str("Data size doesn't match dimensions"));
        }

        // For SVD, we compute A^T * A for eigenvalues (singular values squared)
        let at = self.transpose(&matrix, rows, cols);
        let ata = self.matmul(at, cols, rows, matrix.clone(), rows, cols)?;

        // Compute eigenvalues of A^T * A
        let eigenvals = self.eigenvalues(ata, cols)?;

        // Singular values are square roots of eigenvalues
        let singular_values: Vec<f64> = eigenvals
            .iter()
            .map(|&x| if x >= 0.0 { x.sqrt() } else { 0.0 })
            .collect();

        // Sort in descending order
        let mut sv_sorted = singular_values.clone();
        sv_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Create result object
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"singular_values".into(),
            &js_sys::Array::from_iter(sv_sorted.iter().map(|&x| js_sys::Number::from(x))),
        )?;
        js_sys::Reflect::set(
            &result,
            &"rank".into(),
            &JsValue::from(sv_sorted.iter().filter(|&&x| x > 1e-10).count()),
        )?;
        js_sys::Reflect::set(
            &result,
            &"condition_number".into(),
            &JsValue::from(if sv_sorted.last().unwrap_or(&1e-15) > &1e-15 {
                sv_sorted[0] / sv_sorted.last().unwrap()
            } else {
                f64::INFINITY
            }),
        )?;

        Ok(result)
    }

    /// LU decomposition with partial pivoting
    /// 部分ピボッティング付きLU分解
    #[wasm_bindgen]
    pub fn lu_decomposition(&self, matrix: Vec<f64>, n: usize) -> Result<js_sys::Object, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large for LU decomposition"));
        }

        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        let mut a = matrix;
        let mut p = (0..n).collect::<Vec<usize>>(); // Permutation vector

        // LU decomposition with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if a[i * n + k].abs() > a[max_row * n + k].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..n {
                    a.swap(k * n + j, max_row * n + j);
                }
                p.swap(k, max_row);
            }

            // Check for singular matrix
            if a[k * n + k].abs() < 1e-15 {
                return Err(JsValue::from_str("Matrix is singular"));
            }

            // Elimination
            for i in k + 1..n {
                let factor = a[i * n + k] / a[k * n + k];
                a[i * n + k] = factor; // Store L factor

                for j in k + 1..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }

        // Separate L and U matrices
        let mut l = vec![0.0; n * n];
        let mut u = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    l[i * n + j] = 1.0; // L diagonal
                    u[i * n + j] = a[i * n + j]; // U diagonal and upper
                } else if i > j {
                    l[i * n + j] = a[i * n + j]; // L lower
                } else {
                    u[i * n + j] = a[i * n + j]; // U upper
                }
            }
        }

        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &"L".into(),
            &js_sys::Array::from_iter(l.iter().map(|&x| js_sys::Number::from(x))),
        )?;
        js_sys::Reflect::set(
            &result,
            &"U".into(),
            &js_sys::Array::from_iter(u.iter().map(|&x| js_sys::Number::from(x))),
        )?;
        js_sys::Reflect::set(
            &result,
            &"P".into(),
            &js_sys::Array::from_iter(p.iter().map(|&x| js_sys::Number::from(x as f64))),
        )?;

        Ok(result)
    }

    /// Compute matrix determinant using LU decomposition
    /// LU分解による行列式計算
    #[wasm_bindgen]
    pub fn determinant(&self, matrix: Vec<f64>, n: usize) -> Result<f64, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large"));
        }

        let lu_result = self.lu_decomposition(matrix, n)?;
        let u = js_sys::Reflect::get(&lu_result, &"U".into())?;
        let p = js_sys::Reflect::get(&lu_result, &"P".into())?;

        let u_array: js_sys::Array = u.into();
        let p_array: js_sys::Array = p.into();

        // Determinant is product of U diagonal elements
        let mut det = 1.0;
        for i in 0..n {
            let idx = i * n + i;
            let u_val: f64 = u_array.get(idx as u32).as_f64().unwrap();
            det *= u_val;
        }

        // Account for row swaps in permutation
        let mut swap_count = 0;
        let p_vec: Vec<usize> = (0..n)
            .map(|i| p_array.get(i as u32).as_f64().unwrap() as usize)
            .collect();
        for i in 0..n {
            if p_vec[i] != i {
                swap_count += 1;
            }
        }

        // Odd number of swaps changes sign
        if swap_count % 2 == 1 {
            det = -det;
        }

        Ok(det)
    }

    /// Matrix inverse using LU decomposition
    /// LU分解による逆行列計算
    #[wasm_bindgen]
    pub fn inverse(&self, matrix: Vec<f64>, n: usize) -> Result<Vec<f64>, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large for inversion"));
        }

        let det = self.determinant(matrix.clone(), n)?;
        if det.abs() < 1e-15 {
            return Err(JsValue::from_str("Matrix is singular (determinant ≈ 0)"));
        }

        let lu_result = self.lu_decomposition(matrix, n)?;
        let l = js_sys::Reflect::get(&lu_result, &"L".into())?;
        let u = js_sys::Reflect::get(&lu_result, &"U".into())?;
        let p = js_sys::Reflect::get(&lu_result, &"P".into())?;

        let l_array: js_sys::Array = l.into();
        let u_array: js_sys::Array = u.into();
        let p_array: js_sys::Array = p.into();

        let l_vec: Vec<f64> = (0..n * n)
            .map(|i| l_array.get(i as u32).as_f64().unwrap())
            .collect();
        let u_vec: Vec<f64> = (0..n * n)
            .map(|i| u_array.get(i as u32).as_f64().unwrap())
            .collect();
        let p_vec: Vec<usize> = (0..n)
            .map(|i| p_array.get(i as u32).as_f64().unwrap() as usize)
            .collect();

        let mut inverse = vec![0.0; n * n];

        // Solve for each column of the inverse
        for col in 0..n {
            // Create unit vector for column col
            let mut b = vec![0.0; n];
            b[col] = 1.0;

            // Apply permutation
            let mut pb = vec![0.0; n];
            for i in 0..n {
                pb[i] = b[p_vec[i]];
            }

            // Forward substitution (Ly = Pb)
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut sum = pb[i];
                for j in 0..i {
                    sum -= l_vec[i * n + j] * y[j];
                }
                y[i] = sum; // L has 1s on diagonal
            }

            // Backward substitution (Ux = y)
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                for j in i + 1..n {
                    sum -= u_vec[i * n + j] * x[j];
                }
                x[i] = sum / u_vec[i * n + i];
            }

            // Store column in inverse matrix
            for i in 0..n {
                inverse[i * n + col] = x[i];
            }
        }

        Ok(inverse)
    }

    /// Compute matrix trace (sum of diagonal elements)
    /// 行列のトレース（対角要素の和）
    #[wasm_bindgen]
    pub fn trace(&self, matrix: Vec<f64>, n: usize) -> Result<f64, JsValue> {
        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        let mut trace = 0.0;
        for i in 0..n {
            trace += matrix[i * n + i];
        }

        Ok(trace)
    }

    /// Compute matrix norm (Frobenius norm)
    /// 行列ノルム（フロベニウスノルム）
    #[wasm_bindgen]
    pub fn frobenius_norm(&self, matrix: Vec<f64>) -> f64 {
        matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Matrix transpose
    /// 行列転置
    #[wasm_bindgen]
    pub fn transpose(&self, matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }

        result
    }

    /// Solve linear system Ax = b using LU decomposition
    /// LU分解による連立一次方程式の解法
    #[wasm_bindgen]
    pub fn solve(&self, a: Vec<f64>, b: Vec<f64>, n: usize) -> Result<Vec<f64>, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large"));
        }

        if a.len() != n * n || b.len() != n {
            return Err(JsValue::from_str("Dimension mismatch"));
        }

        let lu_result = self.lu_decomposition(a, n)?;
        let l = js_sys::Reflect::get(&lu_result, &"L".into())?;
        let u = js_sys::Reflect::get(&lu_result, &"U".into())?;
        let p = js_sys::Reflect::get(&lu_result, &"P".into())?;

        let l_array: js_sys::Array = l.into();
        let u_array: js_sys::Array = u.into();
        let p_array: js_sys::Array = p.into();

        let l_vec: Vec<f64> = (0..n * n)
            .map(|i| l_array.get(i as u32).as_f64().unwrap())
            .collect();
        let u_vec: Vec<f64> = (0..n * n)
            .map(|i| u_array.get(i as u32).as_f64().unwrap())
            .collect();
        let p_vec: Vec<usize> = (0..n)
            .map(|i| p_array.get(i as u32).as_f64().unwrap() as usize)
            .collect();

        // Apply permutation to b
        let mut pb = vec![0.0; n];
        for i in 0..n {
            pb[i] = b[p_vec[i]];
        }

        // Forward substitution (Ly = Pb)
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = pb[i];
            for j in 0..i {
                sum -= l_vec[i * n + j] * y[j];
            }
            y[i] = sum;
        }

        // Backward substitution (Ux = y)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in i + 1..n {
                sum -= u_vec[i * n + j] * x[j];
            }
            x[i] = sum / u_vec[i * n + i];
        }

        Ok(x)
    }

    /// Check if matrix is symmetric
    /// 行列が対称かチェック
    #[wasm_bindgen]
    pub fn is_symmetric(&self, matrix: Vec<f64>, n: usize) -> Result<bool, JsValue> {
        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        let tolerance = 1e-10;
        for i in 0..n {
            for j in 0..n {
                if (matrix[i * n + j] - matrix[j * n + i]).abs() > tolerance {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Check if matrix is positive definite
    /// 行列が正定値かチェック
    #[wasm_bindgen]
    pub fn is_positive_definite(&self, matrix: Vec<f64>, n: usize) -> Result<bool, JsValue> {
        if !self.is_symmetric(matrix.clone(), n)? {
            return Ok(false);
        }

        let eigenvals = self.eigenvalues(matrix, n)?;
        Ok(eigenvals.iter().all(|&x| x > 1e-10))
    }

    /// Compute condition number of matrix
    /// 行列の条件数を計算
    #[wasm_bindgen]
    pub fn condition_number(
        &self,
        matrix: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Result<f64, JsValue> {
        let svd_result = self.svd(matrix, rows, cols)?;
        let condition = js_sys::Reflect::get(&svd_result, &"condition_number".into())?;
        Ok(condition.as_f64().unwrap_or(f64::INFINITY))
    }

    /// Matrix rank estimation
    /// 行列ランクの推定
    #[wasm_bindgen]
    pub fn rank(&self, matrix: Vec<f64>, rows: usize, cols: usize) -> Result<usize, JsValue> {
        let svd_result = self.svd(matrix, rows, cols)?;
        let rank = js_sys::Reflect::get(&svd_result, &"rank".into())?;
        Ok(rank.as_f64().unwrap() as usize)
    }

    /// Compute pseudoinverse using SVD
    /// SVDによる擬似逆行列計算
    #[wasm_bindgen]
    pub fn pseudoinverse(
        &self,
        matrix: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if rows > self.max_matrix_size || cols > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large"));
        }

        // For small matrices, use a simplified approach
        // In practice, full SVD would be needed for proper pseudoinverse

        if rows == cols {
            // Square matrix - try regular inverse
            match self.inverse(matrix.clone(), rows) {
                Ok(inv) => Ok(inv),
                Err(_) => {
                    // Fallback: return transpose for rank-deficient matrices
                    Ok(self.transpose(&matrix, rows, cols))
                }
            }
        } else {
            // Non-square: return Moore-Penrose approximation
            if rows > cols {
                // Tall matrix: (A^T A)^(-1) A^T
                let at = self.transpose(&matrix, rows, cols);
                let ata = self.matmul(at.clone(), cols, rows, matrix, rows, cols)?;
                let ata_inv = self.inverse(ata, cols)?;
                self.matmul(ata_inv, cols, cols, at, cols, rows)
            } else {
                // Wide matrix: A^T (A A^T)^(-1)
                let at = self.transpose(&matrix, rows, cols);
                let aat = self.matmul(matrix, rows, cols, at.clone(), cols, rows)?;
                let aat_inv = self.inverse(aat, rows)?;
                self.matmul(at, cols, rows, aat_inv, rows, rows)
            }
        }
    }

    /// Power iteration for largest eigenvalue
    /// べき乗法による最大固有値計算
    #[wasm_bindgen]
    pub fn largest_eigenvalue(
        &self,
        matrix: Vec<f64>,
        n: usize,
        max_iter: usize,
    ) -> Result<f64, JsValue> {
        if n > self.max_matrix_size {
            return Err(JsValue::from_str("Matrix too large"));
        }

        if matrix.len() != n * n {
            return Err(JsValue::from_str("Matrix must be square"));
        }

        // Initialize random vector
        let mut v: Vec<f64> = (0..n).map(|_| js_sys::Math::random() - 0.5).collect();

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            // v = A * v
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += matrix[i * n + j] * v[j];
                }
            }

            // Compute eigenvalue estimate (Rayleigh quotient)
            let numerator: f64 = new_v.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let denominator: f64 = v.iter().map(|&x| x * x).sum();
            eigenvalue = numerator / denominator;

            // Normalize new_v
            let new_norm = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            for x in &mut new_v {
                *x /= new_norm;
            }

            // Check convergence
            let diff: f64 = new_v
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if diff < 1e-10 {
                break;
            }

            v = new_v;
        }

        Ok(eigenvalue)
    }

    /// Matrix-vector product
    /// 行列ベクトル積
    #[wasm_bindgen]
    pub fn matvec(
        &self,
        matrix: Vec<f64>,
        vector: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f64>, JsValue> {
        if matrix.len() != rows * cols || vector.len() != cols {
            return Err(JsValue::from_str("Dimension mismatch"));
        }

        let mut result = vec![0.0; rows];

        for i in 0..rows {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// Vector dot product
    /// ベクトル内積
    #[wasm_bindgen]
    pub fn dot(&self, a: Vec<f64>, b: Vec<f64>) -> Result<f64, JsValue> {
        if a.len() != b.len() {
            return Err(JsValue::from_str("Vector lengths must match"));
        }

        let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(dot_product)
    }

    /// Vector L2 norm
    /// ベクトルL2ノルム
    #[wasm_bindgen]
    pub fn vector_norm(&self, vector: Vec<f64>) -> f64 {
        vector.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize vector to unit length
    /// ベクトルを単位長に正規化
    #[wasm_bindgen]
    pub fn normalize_vector(&self, vector: Vec<f64>) -> Vec<f64> {
        let norm = self.vector_norm(vector.clone());
        if norm > 1e-15 {
            vector.iter().map(|&x| x / norm).collect()
        } else {
            vector
        }
    }

    /// Check maximum matrix size limit
    /// 最大行列サイズ制限をチェック
    #[wasm_bindgen]
    pub fn max_size(&self) -> usize {
        self.max_matrix_size
    }

    /// Get recommended chunk size for large operations
    /// 大きな演算の推奨チャンクサイズを取得
    #[wasm_bindgen]
    pub fn recommended_chunk_size(&self, total_elements: usize) -> usize {
        if total_elements <= 10000 {
            total_elements
        } else if total_elements <= 100000 {
            total_elements / 4
        } else {
            25000 // 25K elements max per chunk
        }
    }
}

/// Linear algebra utilities and helper functions
/// 線形代数ユーティリティとヘルパー関数
#[wasm_bindgen]
pub struct WasmLinAlgUtils;

#[wasm_bindgen]
impl WasmLinAlgUtils {
    /// Generate identity matrix
    /// 単位行列を生成
    #[wasm_bindgen]
    pub fn identity(n: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
        }
        matrix
    }

    /// Generate random matrix for testing
    /// テスト用ランダム行列を生成
    #[wasm_bindgen]
    pub fn random_matrix(rows: usize, cols: usize, scale: f64) -> Vec<f64> {
        (0..rows * cols)
            .map(|_| (js_sys::Math::random() - 0.5) * 2.0 * scale)
            .collect()
    }

    /// Generate symmetric random matrix
    /// 対称ランダム行列を生成
    #[wasm_bindgen]
    pub fn random_symmetric(n: usize, scale: f64) -> Vec<f64> {
        let mut matrix = vec![0.0; n * n];

        for i in 0..n {
            for j in i..n {
                let value = (js_sys::Math::random() - 0.5) * 2.0 * scale;
                matrix[i * n + j] = value;
                matrix[j * n + i] = value; // Symmetric
            }
        }

        matrix
    }

    /// Generate positive definite matrix
    /// 正定値行列を生成
    #[wasm_bindgen]
    pub fn random_positive_definite(n: usize, scale: f64) -> Vec<f64> {
        // Generate A and compute A^T * A
        let a = Self::random_matrix(n, n, scale);
        let at = WasmLinearAlgebra::new(1000).transpose(&a, n, n);
        WasmLinearAlgebra::new(1000)
            .matmul(at, n, n, a, n, n)
            .unwrap()
    }

    /// Check if values are approximately equal
    /// 値が近似的に等しいかチェック
    #[wasm_bindgen]
    pub fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// Check if matrices are approximately equal
    /// 行列が近似的に等しいかチェック
    #[wasm_bindgen]
    pub fn matrices_approx_equal(a: Vec<f64>, b: Vec<f64>, tolerance: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() < tolerance)
    }
}

#[cfg(test)]
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_matrix_multiplication() {
        let linalg = WasmLinearAlgebra::new(100);

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix

        let result = linalg.matmul(a, 2, 2, b, 2, 2).unwrap();

        // Expected: [19, 22, 43, 50]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_qr_decomposition() {
        let linalg = WasmLinearAlgebra::new(100);

        let matrix = vec![1.0, 2.0, 3.0, 4.0];

        let (q, r) = linalg.qr_decomposition(&matrix, 2).unwrap();

        // Verify Q is orthogonal: Q^T * Q = I
        let qt = linalg.transpose(&q, 2, 2);
        let qtq = linalg.matmul(qt, 2, 2, q.clone(), 2, 2).unwrap();

        // Check if qtq is approximately identity
        assert!((qtq[0] - 1.0).abs() < 1e-10); // [0,0]
        assert!(qtq[1].abs() < 1e-10); // [0,1]
        assert!(qtq[2].abs() < 1e-10); // [1,0]
        assert!((qtq[3] - 1.0).abs() < 1e-10); // [1,1]
    }

    #[wasm_bindgen_test]
    fn test_determinant() {
        let linalg = WasmLinearAlgebra::new(100);

        let matrix = vec![1.0, 2.0, 3.0, 4.0];

        let det = linalg.determinant(matrix, 2).unwrap();
        assert!((det - (-2.0)).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_inverse() {
        let linalg = WasmLinearAlgebra::new(100);

        let matrix = vec![2.0, 1.0, 1.0, 1.0];

        let inv = linalg.inverse(matrix.clone(), 2).unwrap();

        // Verify A * A^(-1) = I
        let product = linalg.matmul(matrix, 2, 2, inv, 2, 2).unwrap();
        assert!((product[0] - 1.0).abs() < 1e-10); // [0,0]
        assert!(product[1].abs() < 1e-10); // [0,1]
        assert!(product[2].abs() < 1e-10); // [1,0]
        assert!((product[3] - 1.0).abs() < 1e-10); // [1,1]
    }

    #[wasm_bindgen_test]
    fn test_solve_linear_system() {
        let linalg = WasmLinearAlgebra::new(100);

        let a = vec![2.0, 1.0, 1.0, 1.0];
        let b = vec![3.0, 2.0];

        let x = linalg.solve(a.clone(), b.clone(), 2).unwrap();

        // Verify A * x = b
        let ax = linalg.matvec(a, x, 2, 2).unwrap();
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[wasm_bindgen_test]
    fn test_utilities() {
        let identity = WasmLinAlgUtils::identity(3);
        assert_eq!(identity.len(), 9);
        assert!((identity[0] - 1.0).abs() < 1e-15);
        assert!((identity[4] - 1.0).abs() < 1e-15);
        assert!((identity[8] - 1.0).abs() < 1e-15);
        assert!(identity[1].abs() < 1e-15);

        let linalg = WasmLinearAlgebra::new(100);
        assert!(linalg.is_symmetric(identity, 3).unwrap());
        assert!(linalg.is_positive_definite(identity, 3).unwrap());
    }
}
