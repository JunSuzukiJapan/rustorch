//! Mathematical operations for tensors
//! テンソルの数学演算

use super::core::Tensor;
// Removed unused imports
use num_traits::Float;
use std::ops;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Create tensor with random values from normal distribution
    /// 正規分布からランダム値でテンソルを作成
    pub fn randn(shape: &[usize]) -> Tensor<T>
    where
        T: From<f32>,
    {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.sample::<f32, _>(StandardNormal)))
            .collect();
            
        Tensor::from_vec(data, shape.to_vec())
    }
    
    /// Get the batch size (first dimension)
    /// バッチサイズを取得（最初の次元）
    pub fn batch_size(&self) -> usize {
        self.shape().get(0).copied().unwrap_or(1)
    }
    
    /// Apply function to each element
    /// 各要素に関数を適用
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mapped_data: Vec<T> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_vec(mapped_data, self.shape().to_vec())
    }
    
    /// Element-wise maximum with another tensor
    /// 別のテンソルとの要素ごとの最大値
    pub fn maximum(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err("Shape mismatch for maximum operation".to_string());
        }
        
        let max_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { a } else { b })
            .collect();
            
        Ok(Tensor::from_vec(max_data, self.shape().to_vec()))
    }
    
    /// Stack tensors along a new axis
    /// 新しい軸に沿ってテンソルをスタック
    pub fn stack(tensors: &[&Tensor<T>]) -> Result<Tensor<T>, String> {
        if tensors.is_empty() {
            return Err("Cannot stack empty tensor list".to_string());
        }
        
        let first_shape = tensors[0].shape();
        for tensor in tensors.iter().skip(1) {
            if tensor.shape() != first_shape {
                return Err("All tensors must have the same shape".to_string());
            }
        }
        
        // Create new shape with additional dimension
        let mut new_shape = vec![tensors.len()];
        new_shape.extend_from_slice(first_shape);
        
        // Collect all data
        let mut all_data = Vec::new();
        for tensor in tensors {
            all_data.extend_from_slice(tensor.data.as_slice().unwrap_or(&[]));
        }
        
        Ok(Tensor::from_vec(all_data, new_shape))
    }
    /// Element-wise addition with another tensor (supports broadcasting).
    /// 別のテンソルとの要素ごとの加算（ブロードキャスト対応）
    pub fn add(&self, other: &Tensor<T>) -> Result<Self, String> {
        // Check if shapes are compatible for broadcasting
        if self.shape() == other.shape() {
            // Direct addition when shapes match
            let result_data = self.as_array() + other.as_array();
            Ok(Tensor::new(result_data))
        } else if self.can_broadcast_with(other) {
            // Simple broadcasting for common case: (N, M) + (1, M)
            let self_shape = self.shape();
            let other_shape = other.shape();
            
            // Case 1: (N, M) + (1, M) - bias addition pattern
            if self_shape.len() == 2 && other_shape.len() == 2 && 
               self_shape[1] == other_shape[1] && other_shape[0] == 1 {
                let batch_size = self_shape[0];
                let feature_size = self_shape[1];
                let mut result_data = vec![T::zero(); batch_size * feature_size];
                
                for b in 0..batch_size {
                    for f in 0..feature_size {
                        let self_idx = b * feature_size + f;
                        let other_idx = f; // broadcast from (1, M)
                        result_data[self_idx] = self.data.as_slice().unwrap()[self_idx] + 
                                               other.data.as_slice().unwrap()[other_idx];
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 2: (N, M) + [1] - scalar broadcast
            else if other_shape.len() == 1 && other_shape[0] == 1 {
                let scalar_val = other.data.as_slice().unwrap()[0];
                let result_data: Vec<T> = self.data.as_slice().unwrap().iter()
                    .map(|&x| x + scalar_val)
                    .collect();
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 3: [1] + (N, M) - scalar broadcast (commutative)  
            else if self_shape.len() == 1 && self_shape[0] == 1 {
                let scalar_val = self.data.as_slice().unwrap()[0];
                let result_data: Vec<T> = other.data.as_slice().unwrap().iter()
                    .map(|&x| scalar_val + x)
                    .collect();
                Ok(Tensor::from_vec(result_data, other_shape.to_vec()))
            }
            // Case 4: 4D tensor broadcasting for batch norm: (N, C, H, W) + (1, C, 1, 1)
            else if self_shape.len() == 4 && other_shape.len() == 4 &&
                    self_shape[1] == other_shape[1] &&
                    other_shape[0] == 1 && other_shape[2] == 1 && other_shape[3] == 1 {
                let (n, c, h, w) = (self_shape[0], self_shape[1], self_shape[2], self_shape[3]);
                let mut result_data = vec![T::zero(); n * c * h * w];
                
                for batch in 0..n {
                    for ch in 0..c {
                        let bias_val = other.data.as_slice().unwrap()[ch];
                        for height in 0..h {
                            for width in 0..w {
                                let idx = batch * (c * h * w) + ch * (h * w) + height * w + width;
                                result_data[idx] = self.data.as_slice().unwrap()[idx] + bias_val;
                            }
                        }
                    }
                }
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            } else {
                Err(format!("Broadcasting not implemented for shapes: {:?} vs {:?}", self_shape, other_shape))
            }
        } else {
            Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()))
        }
    }

    /// Element-wise subtraction with another tensor.
    /// 別のテンソルとの要素ごとの減算
    pub fn sub(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() - other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Element-wise multiplication with another tensor (supports broadcasting).
    /// 別のテンソルとの要素ごとの乗算（ブロードキャスト対応）
    pub fn mul(&self, other: &Tensor<T>) -> Result<Self, String> {
        // Check if shapes are compatible for broadcasting
        if self.shape() == other.shape() {
            // Direct multiplication when shapes match
            let result_data = self.as_array() * other.as_array();
            Ok(Tensor::new(result_data))
        } else if self.can_broadcast_with(other) {
            // Broadcasting for multiplication
            let self_shape = self.shape();
            let other_shape = other.shape();
            
            // Case 1: (N, M) * (1, M) - bias multiplication pattern
            if self_shape.len() == 2 && other_shape.len() == 2 && 
               self_shape[1] == other_shape[1] && other_shape[0] == 1 {
                let batch_size = self_shape[0];
                let feature_size = self_shape[1];
                let mut result_data = vec![T::zero(); batch_size * feature_size];
                
                for b in 0..batch_size {
                    for f in 0..feature_size {
                        let self_idx = b * feature_size + f;
                        let other_idx = f; // broadcast from (1, M)
                        result_data[self_idx] = self.data.as_slice().unwrap()[self_idx] * 
                                               other.data.as_slice().unwrap()[other_idx];
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 2: (N, M) * [1] - scalar broadcast
            else if other_shape.len() == 1 && other_shape[0] == 1 {
                let scalar_val = other.data.as_slice().unwrap()[0];
                let result_data: Vec<T> = self.data.as_slice().unwrap().iter()
                    .map(|&x| x * scalar_val)
                    .collect();
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 3: [1] * (N, M) - scalar broadcast (commutative)  
            else if self_shape.len() == 1 && self_shape[0] == 1 {
                let scalar_val = self.data.as_slice().unwrap()[0];
                let result_data: Vec<T> = other.data.as_slice().unwrap().iter()
                    .map(|&x| scalar_val * x)
                    .collect();
                Ok(Tensor::from_vec(result_data, other_shape.to_vec()))
            } else {
                Err(format!("Broadcasting not implemented for shapes: {:?} vs {:?}", self_shape, other_shape))
            }
        } else {
            Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()))
        }
    }

    /// Element-wise division with another tensor.
    /// 別のテンソルとの要素ごとの除算
    pub fn div(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape(), other.shape()));
        }
        
        let result_data = self.as_array() / other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Scalar addition.
    /// スカラー加算
    pub fn add_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() + scalar;
        Tensor::new(result_data)
    }

    /// Scalar subtraction.
    /// スカラー減算
    pub fn sub_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() - scalar;
        Tensor::new(result_data)
    }

    /// Scalar multiplication.
    /// スカラー乗算
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() * scalar;
        Tensor::new(result_data)
    }

    /// Scalar division.
    /// スカラー除算
    pub fn div_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() / scalar;
        Tensor::new(result_data)
    }

    /// Negation of all elements.
    /// 全要素の符号反転
    pub fn neg(&self) -> Self {
        let result_data = self.as_array().mapv(|x| -x);
        Tensor::new(result_data)
    }

    /// Matrix multiplication.
    /// 行列乗算
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Self, String> {
        // Support 2D x 2D, 3D x 2D, and 4D x 4D operations
        if !(self.ndim() == 2 && other.ndim() == 2) && 
           !(self.ndim() == 3 && other.ndim() == 2) &&
           !(self.ndim() == 4 && other.ndim() == 4) {
            return Err(format!(
                "Unsupported matmul dimensions: {}D @ {}D",
                self.ndim(), other.ndim()
            ));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();
        
        // Handle 2D x 2D case
        if self.ndim() == 2 && other.ndim() == 2 {
            if self_shape[1] != other_shape[0] {
                return Err(format!(
                    "Matrix dimension mismatch: {}x{} @ {}x{}",
                    self_shape[0], self_shape[1], other_shape[0], other_shape[1]
                ));
            }

            let result_shape = vec![self_shape[0], other_shape[1]];
            let mut result_data = vec![T::zero(); result_shape.iter().product()];

            let self_array = &self.data;
            let other_array = &other.data;

            for i in 0..result_shape[0] {
                for j in 0..result_shape[1] {
                    let mut sum = T::zero();
                    for k in 0..self_shape[1] {
                        let self_val = self_array[[i, k]];
                        let other_val = other_array[[k, j]];
                        sum = sum + self_val * other_val;
                    }
                    result_data[i * result_shape[1] + j] = sum;
                }
            }

            Ok(Tensor::from_vec(result_data, result_shape))
        }
        // Handle 3D x 2D (batch) case  
        else if self.ndim() == 3 && other.ndim() == 2 {
            let batch_size = self_shape[0];
            let seq_len = self_shape[1];
            let input_feat = self_shape[2];
            let output_feat = other_shape[1];
            
            if input_feat != other_shape[0] {
                return Err(format!(
                    "Feature dimension mismatch: {} != {}",
                    input_feat, other_shape[0]
                ));
            }
            
            let result_shape = vec![batch_size, seq_len, output_feat];
            let mut result_data = vec![T::zero(); result_shape.iter().product()];
            
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for o in 0..output_feat {
                        let mut sum = T::zero();
                        for i in 0..input_feat {
                            let self_val = self.data[[b, s, i]];
                            let other_val = other.data[[i, o]];
                            sum = sum + self_val * other_val;
                        }
                        result_data[b * seq_len * output_feat + s * output_feat + o] = sum;
                    }
                }
            }
            
            Ok(Tensor::from_vec(result_data, result_shape))
        }
        // Handle 4D x 4D (batch, heads, seq, features) case for attention
        else if self.ndim() == 4 && other.ndim() == 4 {
            let batch_size = self_shape[0];
            let num_heads = self_shape[1];
            let seq_len_q = self_shape[2];
            let feat_q = self_shape[3];
            
            // For attention: Q @ K^T, where K^T has shape (batch, heads, features, seq_len)
            // So we expect other to be transposed: (batch, heads, feat_k, seq_len_k)
            let feat_k = other_shape[2];
            let seq_len_k = other_shape[3];
            
            if feat_q != feat_k {
                return Err(format!(
                    "Feature dimension mismatch: {} != {}",
                    feat_q, feat_k
                ));
            }
            
            let result_shape = vec![batch_size, num_heads, seq_len_q, seq_len_k];
            let mut result_data = vec![T::zero(); result_shape.iter().product()];
            
            for b in 0..batch_size {
                for h in 0..num_heads {
                    for i in 0..seq_len_q {
                        for j in 0..seq_len_k {
                            let mut sum = T::zero();
                            for k in 0..feat_q {
                                let self_val = self.data[[b, h, i, k]];
                                let other_val = other.data[[b, h, k, j]]; // Transposed access
                                sum = sum + self_val * other_val;
                            }
                            let idx = b * num_heads * seq_len_q * seq_len_k +
                                     h * seq_len_q * seq_len_k +
                                     i * seq_len_k + j;
                            result_data[idx] = sum;
                        }
                    }
                }
            }
            
            Ok(Tensor::from_vec(result_data, result_shape))
        }
        else {
            Err("Unsupported tensor dimensions for matmul".to_string())
        }
    }

    /// Transpose the tensor's last two dimensions.
    /// テンソルの最後の2次元を転置
    pub fn transpose_last_two(&self) -> Result<Self, String> {
        if self.ndim() < 2 {
            return Err("transpose_last_two requires at least 2D tensor".to_string());
        }
        
        let mut new_shape = self.shape().to_vec();
        let last = new_shape.len() - 1;
        let second_last = new_shape.len() - 2;
        new_shape.swap(second_last, last);
        
        let mut new_data = vec![T::zero(); self.data.len()];
        
        // For 2D case, use simple transpose
        if self.ndim() == 2 {
            let rows = self.shape()[0];
            let cols = self.shape()[1];
            for i in 0..rows {
                for j in 0..cols {
                    let src_idx = i * cols + j;
                    let dst_idx = j * rows + i;
                    new_data[dst_idx] = self.data.as_slice().unwrap_or(&[])[src_idx];
                }
            }
        } else {
            // For higher dimensions, transpose only last two
            let total_batches: usize = self.shape().iter().take(self.ndim() - 2).product();
            let rows = self.shape()[self.ndim() - 2];
            let cols = self.shape()[self.ndim() - 1];
            
            for batch in 0..total_batches {
                for i in 0..rows {
                    for j in 0..cols {
                        let src_idx = batch * rows * cols + i * cols + j;
                        let dst_idx = batch * rows * cols + j * rows + i;
                        new_data[dst_idx] = self.data.as_slice().unwrap_or(&[])[src_idx];
                    }
                }
            }
        }
        
        Ok(Tensor::from_vec(new_data, new_shape))
    }

    /// Transpose the tensor (swap last two dimensions).
    /// テンソルの転置（最後の2次元を入れ替え）
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() < 2 {
            return Err("Transpose requires at least 2D tensor".to_string());
        }

        let mut axes: Vec<usize> = (0..self.ndim()).collect();
        let ndim = self.ndim();
        axes.swap(ndim - 2, ndim - 1);

        let transposed = self.as_array().clone().permuted_axes(axes.into_iter().collect::<Vec<_>>());
        Ok(Tensor::new(transposed))
    }

    /// Sum of all elements.
    /// 全要素の合計
    pub fn sum(&self) -> T {
        self.as_array().sum()
    }

    /// Mean of all elements.
    /// 全要素の平均
    pub fn mean(&self) -> T {
        let sum = self.sum();
        let count = T::from(self.numel()).unwrap_or(T::one());
        sum / count
    }
    
    /// Get a single scalar value from a tensor (must contain exactly one element)
    /// テンソルから単一のスカラー値を取得（テンソルは正確に1つの要素を含む必要がある）
    pub fn item(&self) -> T {
        if self.numel() != 1 {
            panic!("Tensor must contain exactly one element to call item(), but it contains {}", self.numel());
        }
        self.as_array().iter().next().unwrap().clone()
    }

    /// Sum along a specific axis.
    /// 特定の軸に沿った合計
    pub fn sum_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} out of bounds for {}-dimensional tensor", axis, self.ndim()));
        }

        let sum_result = self.as_array().sum_axis(ndarray::Axis(axis));
        Ok(Tensor::new(sum_result))
    }

    /// Mean along a specific axis.
    /// 特定の軸に沿った平均
    pub fn mean_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} out of bounds for {}-dimensional tensor", axis, self.ndim()));
        }

        let mean_result = self.as_array().mean_axis(ndarray::Axis(axis)).unwrap();
        Ok(Tensor::new(mean_result))
    }

    /// Apply a function element-wise.

    /// Apply exponential function element-wise.
    /// 要素ごとに指数関数を適用
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    /// Apply natural logarithm element-wise.
    /// 要素ごとに自然対数を適用
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Apply sine function element-wise.
    /// 要素ごとにサイン関数を適用
    pub fn sin(&self) -> Self {
        self.map(|x| x.sin())
    }

    /// Apply cosine function element-wise.
    /// 要素ごとにコサイン関数を適用
    pub fn cos(&self) -> Self {
        self.map(|x| x.cos())
    }

    /// Apply tangent function element-wise.
    /// 要素ごとにタンジェント関数を適用
    pub fn tan(&self) -> Self {
        self.map(|x| x.tan())
    }

    /// Apply square root element-wise.
    /// 要素ごとに平方根を適用
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    /// Apply absolute value element-wise.
    /// 要素ごとに絶対値を適用
    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    /// Power operation element-wise.
    /// 要素ごとのべき乗演算
    pub fn pow(&self, exponent: T) -> Self {
        self.map(|x| x.powf(exponent))
    }

    /// Maximum value of the tensor.
    /// テンソルの最大値
    pub fn max(&self) -> Option<T> {
        self.as_slice()?.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }

    /// Minimum value of the tensor.
    /// テンソルの最小値
    pub fn min(&self) -> Option<T> {
        self.as_slice()?.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }
}

// Operator overloading for tensors
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs).expect("Subtraction failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs).expect("Multiplication failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs).expect("Division failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::ops::Neg<Output = T>> ops::Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        let result_data = self.as_array().mapv(|x| -x);
        Tensor::new(result_data)
    }
}

// Scalar operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// In-place operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::AddAssign<&Tensor<T>> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.add(rhs).expect("AddAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::SubAssign<&Tensor<T>> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.sub(rhs).expect("SubAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::MulAssign<&Tensor<T>> for Tensor<T> {
    fn mul_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.mul(rhs).expect("MulAssign failed");
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::DivAssign<&Tensor<T>> for Tensor<T> {
    fn div_assign(&mut self, rhs: &Tensor<T>) {
        *self = self.div(rhs).expect("DivAssign failed");
    }
}

// Value-based operations for direct tensor operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs).expect("Subtraction failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs).expect("Multiplication failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs).expect("Division failed")
    }
}

// Scalar operations with tensor values
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        (&self).add_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Sub<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        (&self).sub_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        (&self).mul_scalar(rhs)
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Div<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        (&self).div_scalar(rhs)
    }
}

// Mixed reference/value operations
impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<&Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        (&self).add(rhs).expect("Addition failed")
    }
}

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> ops::Add<Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        self.add(&rhs).expect("Addition failed")
    }
}