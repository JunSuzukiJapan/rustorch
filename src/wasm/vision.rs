//! Vision and image processing functions for WASM
//! WASM用画像・視覚処理関数

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Vision utilities for WASM
/// WASM用画像処理ユーティリティ
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmVision;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmVision {
    /// Resize image using bilinear interpolation
    /// バイリニア補間による画像リサイズ
    #[wasm_bindgen]
    pub fn resize(
        image_data: Vec<f32>,
        original_height: usize,
        original_width: usize,
        new_height: usize,
        new_width: usize,
        channels: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != original_height * original_width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        let mut resized = vec![0.0; new_height * new_width * channels];

        let x_ratio = original_width as f32 / new_width as f32;
        let y_ratio = original_height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x as f32 * x_ratio;
                let src_y = y as f32 * y_ratio;

                let x1 = src_x as usize;
                let y1 = src_y as usize;
                let x2 = (x1 + 1).min(original_width - 1);
                let y2 = (y1 + 1).min(original_height - 1);

                let dx = src_x - x1 as f32;
                let dy = src_y - y1 as f32;

                for c in 0..channels {
                    // Get four neighboring pixels
                    let idx_11 = (y1 * original_width + x1) * channels + c;
                    let idx_12 = (y1 * original_width + x2) * channels + c;
                    let idx_21 = (y2 * original_width + x1) * channels + c;
                    let idx_22 = (y2 * original_width + x2) * channels + c;

                    let val_11 = image_data[idx_11];
                    let val_12 = image_data[idx_12];
                    let val_21 = image_data[idx_21];
                    let val_22 = image_data[idx_22];

                    // Bilinear interpolation
                    let interpolated = val_11 * (1.0 - dx) * (1.0 - dy)
                        + val_12 * dx * (1.0 - dy)
                        + val_21 * (1.0 - dx) * dy
                        + val_22 * dx * dy;

                    let output_idx = (y * new_width + x) * channels + c;
                    resized[output_idx] = interpolated;
                }
            }
        }

        Ok(resized)
    }

    /// Normalize image with mean and standard deviation
    /// 平均と標準偏差による画像正規化
    #[wasm_bindgen]
    pub fn normalize(
        image_data: Vec<f32>,
        mean: Vec<f32>,
        std: Vec<f32>,
        channels: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if mean.len() != channels || std.len() != channels {
            return Err(JsValue::from_str(
                "Mean and std must match number of channels",
            ));
        }

        if image_data.len() % channels != 0 {
            return Err(JsValue::from_str(
                "Image data size must be divisible by channels",
            ));
        }

        let mut normalized = Vec::with_capacity(image_data.len());

        for (i, &pixel) in image_data.iter().enumerate() {
            let channel = i % channels;
            let normalized_pixel = (pixel - mean[channel]) / std[channel];
            normalized.push(normalized_pixel);
        }

        Ok(normalized)
    }

    /// Convert RGB to grayscale
    /// RGBからグレースケールに変換
    #[wasm_bindgen]
    pub fn rgb_to_grayscale(
        rgb_data: Vec<f32>,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if rgb_data.len() != height * width * 3 {
            return Err(JsValue::from_str("RGB data size mismatch"));
        }

        let mut grayscale = vec![0.0; height * width];

        for i in 0..(height * width) {
            let r = rgb_data[i * 3];
            let g = rgb_data[i * 3 + 1];
            let b = rgb_data[i * 3 + 2];

            // Luminance formula: 0.299*R + 0.587*G + 0.114*B
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }

        Ok(grayscale)
    }

    /// Apply Gaussian blur
    /// ガウシアンブラーを適用
    #[wasm_bindgen]
    pub fn gaussian_blur(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
        sigma: f32,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        // Generate Gaussian kernel (5x5)
        let kernel_size = 5;
        let kernel_radius = kernel_size / 2;
        let mut kernel = vec![0.0; kernel_size * kernel_size];
        let mut kernel_sum = 0.0;

        for ky in 0..kernel_size {
            for kx in 0..kernel_size {
                let x = (kx as i32 - kernel_radius as i32) as f32;
                let y = (ky as i32 - kernel_radius as i32) as f32;
                let value = (-0.5 * (x * x + y * y) / (sigma * sigma)).exp();
                kernel[ky * kernel_size + kx] = value;
                kernel_sum += value;
            }
        }

        // Normalize kernel
        for k in kernel.iter_mut() {
            *k /= kernel_sum;
        }

        let mut blurred = vec![0.0; image_data.len()];

        // Apply convolution
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let mut sum = 0.0;

                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let src_y = y as i32 + ky as i32 - kernel_radius as i32;
                            let src_x = x as i32 + kx as i32 - kernel_radius as i32;

                            // Handle boundaries with clamping
                            let src_y = src_y.max(0).min(height as i32 - 1) as usize;
                            let src_x = src_x.max(0).min(width as i32 - 1) as usize;

                            let src_idx = (src_y * width + src_x) * channels + c;
                            let kernel_val = kernel[ky * kernel_size + kx];

                            sum += image_data[src_idx] * kernel_val;
                        }
                    }

                    let dst_idx = (y * width + x) * channels + c;
                    blurred[dst_idx] = sum;
                }
            }
        }

        Ok(blurred)
    }

    /// Crop image to specified region
    /// 指定領域に画像をクロップ
    #[wasm_bindgen]
    pub fn crop(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
        start_y: usize,
        start_x: usize,
        crop_height: usize,
        crop_width: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        if start_y + crop_height > height || start_x + crop_width > width {
            return Err(JsValue::from_str("Crop region exceeds image bounds"));
        }

        let mut cropped = vec![0.0; crop_height * crop_width * channels];

        for y in 0..crop_height {
            for x in 0..crop_width {
                for c in 0..channels {
                    let src_idx = ((start_y + y) * width + (start_x + x)) * channels + c;
                    let dst_idx = (y * crop_width + x) * channels + c;
                    cropped[dst_idx] = image_data[src_idx];
                }
            }
        }

        Ok(cropped)
    }

    /// Flip image horizontally
    /// 画像を水平反転
    #[wasm_bindgen]
    pub fn flip_horizontal(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        let mut flipped = vec![0.0; image_data.len()];

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let src_idx = (y * width + x) * channels + c;
                    let dst_x = width - 1 - x;
                    let dst_idx = (y * width + dst_x) * channels + c;
                    flipped[dst_idx] = image_data[src_idx];
                }
            }
        }

        Ok(flipped)
    }

    /// Flip image vertically
    /// 画像を垂直反転
    #[wasm_bindgen]
    pub fn flip_vertical(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        let mut flipped = vec![0.0; image_data.len()];

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let src_idx = (y * width + x) * channels + c;
                    let dst_y = height - 1 - y;
                    let dst_idx = (dst_y * width + x) * channels + c;
                    flipped[dst_idx] = image_data[src_idx];
                }
            }
        }

        Ok(flipped)
    }

    /// Rotate image by 90 degrees clockwise
    /// 画像を時計回りに90度回転
    #[wasm_bindgen]
    pub fn rotate_90_cw(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        // After 90° CW rotation: new_height = width, new_width = height
        let mut rotated = vec![0.0; width * height * channels];

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let src_idx = (y * width + x) * channels + c;
                    // Mapping: (y, x) -> (x, height-1-y)
                    let dst_y = x;
                    let dst_x = height - 1 - y;
                    let dst_idx = (dst_y * height + dst_x) * channels + c;
                    rotated[dst_idx] = image_data[src_idx];
                }
            }
        }

        Ok(rotated)
    }

    /// Apply center crop (crop from center of image)
    /// センタークロップ（画像中央からクロップ）
    #[wasm_bindgen]
    pub fn center_crop(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
        crop_size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if crop_size > height || crop_size > width {
            return Err(JsValue::from_str("Crop size larger than image"));
        }

        let start_y = (height - crop_size) / 2;
        let start_x = (width - crop_size) / 2;

        Self::crop(
            image_data, height, width, channels, start_y, start_x, crop_size, crop_size,
        )
    }

    /// Adjust image brightness
    /// 画像の明度を調整
    #[wasm_bindgen]
    pub fn adjust_brightness(image_data: Vec<f32>, factor: f32) -> Vec<f32> {
        image_data
            .into_iter()
            .map(|pixel| (pixel + factor).max(0.0).min(1.0))
            .collect()
    }

    /// Adjust image contrast
    /// 画像のコントラストを調整
    #[wasm_bindgen]
    pub fn adjust_contrast(image_data: Vec<f32>, factor: f32) -> Vec<f32> {
        image_data
            .into_iter()
            .map(|pixel| ((pixel - 0.5) * factor + 0.5).max(0.0).min(1.0))
            .collect()
    }

    /// Add Gaussian noise to image (data augmentation)
    /// 画像にガウシアンノイズを追加（データ拡張）
    #[wasm_bindgen]
    pub fn add_gaussian_noise(image_data: Vec<f32>, std_dev: f32) -> Vec<f32> {
        image_data
            .into_iter()
            .map(|pixel| {
                // Simple Box-Muller transform for Gaussian noise
                let u1 = js_sys::Math::random() as f32;
                let u2 = js_sys::Math::random() as f32;
                let noise =
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std_dev;

                (pixel + noise).max(0.0).min(1.0)
            })
            .collect()
    }

    /// Apply random rotation (for data augmentation)
    /// ランダム回転を適用（データ拡張用）
    #[wasm_bindgen]
    pub fn random_rotation(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
        max_angle_deg: f32,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width * channels {
            return Err(JsValue::from_str("Image data size mismatch"));
        }

        // Random angle in radians
        let angle_rad =
            (js_sys::Math::random() as f32 * 2.0 - 1.0) * max_angle_deg * std::f32::consts::PI
                / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let mut rotated = vec![0.0; image_data.len()];
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        for y in 0..height {
            for x in 0..width {
                // Rotate coordinates around center
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;

                let src_x = dx * cos_a - dy * sin_a + center_x;
                let src_y = dx * sin_a + dy * cos_a + center_y;

                // Bilinear interpolation for sub-pixel sampling
                if src_x >= 0.0
                    && src_x < width as f32 - 1.0
                    && src_y >= 0.0
                    && src_y < height as f32 - 1.0
                {
                    let x1 = src_x as usize;
                    let y1 = src_y as usize;
                    let x2 = x1 + 1;
                    let y2 = y1 + 1;

                    let dx = src_x - x1 as f32;
                    let dy = src_y - y1 as f32;

                    for c in 0..channels {
                        let idx_11 = (y1 * width + x1) * channels + c;
                        let idx_12 = (y1 * width + x2) * channels + c;
                        let idx_21 = (y2 * width + x1) * channels + c;
                        let idx_22 = (y2 * width + x2) * channels + c;

                        let val_11 = image_data[idx_11];
                        let val_12 = image_data[idx_12];
                        let val_21 = image_data[idx_21];
                        let val_22 = image_data[idx_22];

                        let interpolated = val_11 * (1.0 - dx) * (1.0 - dy)
                            + val_12 * dx * (1.0 - dy)
                            + val_21 * (1.0 - dx) * dy
                            + val_22 * dx * dy;

                        let dst_idx = (y * width + x) * channels + c;
                        rotated[dst_idx] = interpolated;
                    }
                }
            }
        }

        Ok(rotated)
    }

    /// Apply edge detection (Sobel filter)
    /// エッジ検出（Sobelフィルター）
    #[wasm_bindgen]
    pub fn edge_detection(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width {
            return Err(JsValue::from_str("Expected grayscale image"));
        }

        let mut edges = vec![0.0; height * width];

        // Sobel kernels
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                // Apply Sobel kernels
                for ky in 0..3 {
                    for kx in 0..3 {
                        let src_y = y + ky - 1;
                        let src_x = x + kx - 1;
                        let src_idx = src_y * width + src_x;
                        let kernel_idx = ky * 3 + kx;

                        gx += image_data[src_idx] * sobel_x[kernel_idx];
                        gy += image_data[src_idx] * sobel_y[kernel_idx];
                    }
                }

                // Compute magnitude
                let magnitude = (gx * gx + gy * gy).sqrt();
                let dst_idx = y * width + x;
                edges[dst_idx] = magnitude;
            }
        }

        Ok(edges)
    }

    /// Apply corner detection (Harris corner detector)
    /// コーナー検出（Harris検出器）
    #[wasm_bindgen]
    pub fn harris_corner_detection(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        threshold: f32,
        k: f32,
    ) -> Result<Vec<f32>, JsValue> {
        if image_data.len() != height * width {
            return Err(JsValue::from_str("Expected grayscale image"));
        }

        let mut response = vec![0.0; height * width];

        // Compute gradients
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut ixx = 0.0f32;
                let mut iyy = 0.0f32;
                let mut ixy = 0.0f32;

                // Harris window (3x3)
                for wy in 0..3 {
                    for wx in 0..3 {
                        let py = y + wy - 1;
                        let px = x + wx - 1;

                        // Compute gradients using Sobel
                        let ix = (image_data[py * width + (px + 1).min(width - 1)]
                            - image_data[py * width + px.max(1) - 1])
                            * 0.5;
                        let iy = (image_data[(py + 1).min(height - 1) * width + px]
                            - image_data[py.max(1) - 1 * width + px])
                            * 0.5;

                        ixx += ix * ix;
                        iyy += iy * iy;
                        ixy += ix * iy;
                    }
                }

                // Harris response: det(M) - k * trace(M)^2
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let harris_response = det - k * trace * trace;

                response[y * width + x] = if harris_response > threshold {
                    harris_response
                } else {
                    0.0
                };
            }
        }

        Ok(response)
    }

    /// Apply morphological operations (opening/closing)
    /// モルフォロジー演算（オープニング/クロージング）
    #[wasm_bindgen]
    pub fn morphological_opening(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let eroded = Self::morphological_erosion_f32(image_data, height, width, kernel_size)?;
        Self::morphological_dilation_f32(eroded, height, width, kernel_size)
    }

    #[wasm_bindgen]
    pub fn morphological_closing(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let dilated = Self::morphological_dilation_f32(image_data, height, width, kernel_size)?;
        Self::morphological_erosion_f32(dilated, height, width, kernel_size)
    }

    /// Compute local binary patterns
    /// 局所二値パターンを計算
    #[wasm_bindgen]
    pub fn local_binary_patterns(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        radius: usize,
    ) -> Result<Vec<u8>, JsValue> {
        if image_data.len() != height * width {
            return Err(JsValue::from_str("Expected grayscale image"));
        }

        let mut lbp = vec![0u8; height * width];

        for y in radius..(height - radius) {
            for x in radius..(width - radius) {
                let center_val = image_data[y * width + x];
                let mut pattern = 0u8;

                // 8-neighbor sampling
                let neighbors = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (1, -1),
                    (0, -1),
                ];

                for (i, (dy, dx)) in neighbors.iter().enumerate() {
                    let ny = (y as i32 + dy * radius as i32) as usize;
                    let nx = (x as i32 + dx * radius as i32) as usize;
                    let neighbor_val = image_data[ny * width + nx];

                    if neighbor_val >= center_val {
                        pattern |= 1 << i;
                    }
                }

                lbp[y * width + x] = pattern;
            }
        }

        Ok(lbp)
    }

    // Helper functions for f32 morphological operations
    fn morphological_dilation_f32(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let mut result = image_data.clone();
        let half_kernel = kernel_size / 2;

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut max_val = 0.0f32;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let img_y = y + ky - half_kernel;
                        let img_x = x + kx - half_kernel;
                        let idx = img_y * width + img_x;
                        max_val = max_val.max(image_data[idx]);
                    }
                }

                result[y * width + x] = max_val;
            }
        }

        Ok(result)
    }

    fn morphological_erosion_f32(
        image_data: Vec<f32>,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let mut result = image_data.clone();
        let half_kernel = kernel_size / 2;

        for y in half_kernel..height - half_kernel {
            for x in half_kernel..width - half_kernel {
                let mut min_val = 1.0f32;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let img_y = y + ky - half_kernel;
                        let img_x = x + kx - half_kernel;
                        let idx = img_y * width + img_x;
                        min_val = min_val.min(image_data[idx]);
                    }
                }

                result[y * width + x] = min_val;
            }
        }

        Ok(result)
    }

    /// Convert image from 0-255 range to 0-1 range
    /// 画像を0-255範囲から0-1範囲に変換
    #[wasm_bindgen]
    pub fn to_float(image_data: Vec<u8>) -> Vec<f32> {
        image_data
            .into_iter()
            .map(|pixel| pixel as f32 / 255.0)
            .collect()
    }

    /// Convert image from 0-1 range to 0-255 range
    /// 画像を0-1範囲から0-255範囲に変換
    #[wasm_bindgen]
    pub fn to_uint8(image_data: Vec<f32>) -> Vec<u8> {
        image_data
            .into_iter()
            .map(|pixel| (pixel * 255.0).round().max(0.0).min(255.0) as u8)
            .collect()
    }

    /// Calculate image histogram
    /// 画像のヒストグラムを計算
    #[wasm_bindgen]
    pub fn histogram(image_data: Vec<f32>, bins: usize) -> Vec<u32> {
        let mut hist = vec![0u32; bins];

        for &pixel in &image_data {
            let bin_idx = ((pixel * bins as f32).floor() as usize).min(bins - 1);
            hist[bin_idx] += 1;
        }

        hist
    }

    /// Apply histogram equalization
    /// ヒストグラム均等化を適用
    #[wasm_bindgen]
    pub fn histogram_equalization(image_data: Vec<f32>, bins: usize) -> Vec<f32> {
        let hist = Self::histogram(image_data.clone(), bins);
        let total_pixels = image_data.len() as f32;

        // Calculate cumulative distribution
        let mut cdf = vec![0.0; bins];
        cdf[0] = hist[0] as f32 / total_pixels;
        for i in 1..bins {
            cdf[i] = cdf[i - 1] + hist[i] as f32 / total_pixels;
        }

        // Apply equalization
        image_data
            .into_iter()
            .map(|pixel| {
                let bin_idx = ((pixel * bins as f32).floor() as usize).min(bins - 1);
                cdf[bin_idx]
            })
            .collect()
    }
}

#[cfg(test)]
#[cfg(feature = "wasm")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_resize() {
        let image = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 grayscale
        let resized = WasmVision::resize(image, 2, 2, 4, 4, 1).unwrap();
        assert_eq!(resized.len(), 16); // 4x4 = 16 pixels
    }

    #[wasm_bindgen_test]
    fn test_normalize() {
        let image = vec![0.0, 128.0, 255.0]; // RGB pixel
        let mean = vec![127.5, 127.5, 127.5];
        let std = vec![127.5, 127.5, 127.5];

        let normalized = WasmVision::normalize(image, mean, std, 3).unwrap();

        // Should be approximately [-1, 0, 1]
        assert!((normalized[0] - (-1.0)).abs() < 0.1);
        assert!(normalized[1].abs() < 0.1);
        assert!((normalized[2] - 1.0).abs() < 0.1);
    }

    #[wasm_bindgen_test]
    fn test_rgb_to_grayscale() {
        let rgb = vec![
            1.0, 0.0, 0.0, // Pure red
            0.0, 1.0, 0.0, // Pure green
            0.0, 0.0, 1.0, // Pure blue
        ];

        let grayscale = WasmVision::rgb_to_grayscale(rgb, 1, 3).unwrap();

        assert_eq!(grayscale.len(), 3);
        assert!((grayscale[0] - 0.299).abs() < 1e-5); // Red component
        assert!((grayscale[1] - 0.587).abs() < 1e-5); // Green component
        assert!((grayscale[2] - 0.114).abs() < 1e-5); // Blue component
    }

    #[wasm_bindgen_test]
    fn test_to_float_uint8_conversion() {
        let uint8_data = vec![0, 128, 255];
        let float_data = WasmVision::to_float(uint8_data);
        let back_to_uint8 = WasmVision::to_uint8(float_data);

        assert_eq!(back_to_uint8, vec![0, 128, 255]);
    }

    #[wasm_bindgen_test]
    fn test_flip_horizontal() {
        let image = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let flipped = WasmVision::flip_horizontal(image, 2, 2, 1).unwrap();

        // Original: [1 2]  Flipped: [2 1]
        //          [3 4]            [4 3]
        assert_eq!(flipped, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[wasm_bindgen_test]
    fn test_crop() {
        let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3 image

        let cropped = WasmVision::crop(image, 3, 3, 1, 1, 1, 2, 2).unwrap();
        assert_eq!(cropped, vec![5.0, 6.0, 8.0, 9.0]); // Center 2x2 region
    }
}
