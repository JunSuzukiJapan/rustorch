//! Image processing example with WASM data transforms
//! WASMデータ変換を使った画像処理の例

#[cfg(feature = "wasm")]
fn main() {
    use rustorch::wasm::tensor::WasmTensor;
    use rustorch::wasm::data_transforms::*;
    use rustorch::wasm::common::MemoryManager;
    
    println!("=== RusTorch WASM Image Processing Demo ===");
    
    // Initialize memory pool for image processing
    MemoryManager::init_pool(100);
    
    // Simulate RGB image data (224x224x3)
    let height = 224;
    let width = 224;
    let channels = 3;
    let total_pixels = height * width * channels;
    
    // Create synthetic image data with spatial patterns
    let mut image_data = Vec::with_capacity(total_pixels);
    for h in 0..height {
        for w in 0..width {
            for c in 0..channels {
                // Create gradient pattern with color variation
                let h_norm = h as f32 / height as f32;
                let w_norm = w as f32 / width as f32;
                let base_value = (h_norm + w_norm) / 2.0;
                
                let pixel_value = match c {
                    0 => base_value * 255.0,           // Red channel
                    1 => (1.0 - base_value) * 255.0,   // Green channel
                    2 => (h_norm * w_norm) * 255.0,    // Blue channel
                    _ => 0.0,
                };
                
                image_data.push(pixel_value);
            }
        }
    }
    
    let original_tensor = WasmTensor::new(image_data, vec![height, width, channels]);
    println!("Original image shape: {:?}", original_tensor.shape());
    println!("Original pixel range: {:.2} - {:.2}", 
             original_tensor.data().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             original_tensor.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Image preprocessing pipeline
    println!("\n--- Image Preprocessing Pipeline ---");
    
    // 1. Resize to 256x256
    println!("1. Resizing to 256x256...");
    if let Ok(resize_transform) = WasmResize::new(256, 256, "bilinear".to_string()) {
        if let Ok(resized) = resize_transform.apply(&original_tensor) {
            println!("   Resized shape: {:?}", resized.shape());
            println!("   Transform: {}", resize_transform.name());
            
            // 2. Center crop to 224x224
            println!("2. Center cropping to 224x224...");
            if let Ok(crop_transform) = WasmCenterCrop::new(224, 224) {
                if let Ok(cropped) = crop_transform.apply(&resized) {
                    println!("   Cropped shape: {:?}", cropped.shape());
                    
                    // 3. Color jitter augmentation
                    println!("3. Applying color jitter...");
                    if let Ok(jitter_transform) = WasmColorJitter::new(0.2, 0.2, 0.1, 0.05) {
                        if let Ok(jittered) = jitter_transform.apply(&cropped) {
                            println!("   Color jitter applied: {}", jitter_transform.name());
                            
                            // 4. Normalize for ImageNet
                            println!("4. Normalizing for ImageNet...");
                            let imagenet_mean = vec![0.485, 0.456, 0.406];
                            let imagenet_std = vec![0.229, 0.224, 0.225];
                            
                            if let Ok(normalize_transform) = WasmNormalize::new(&imagenet_mean, &imagenet_std) {
                                if let Ok(normalized) = normalize_transform.apply(&jittered) {
                                    println!("   Normalized range: {:.3} - {:.3}",
                                             normalized.data().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                                             normalized.data().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                                    
                                    // 5. Convert to tensor format
                                    println!("5. Converting to tensor format...");
                                    if let Ok(tensor_transform) = WasmToTensor::new() {
                                        if let Ok(final_tensor) = tensor_transform.apply(&normalized) {
                                            println!("   Final tensor shape: {:?}", final_tensor.shape());
                                            println!("   Processing complete!");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Alternative: Use pipeline for cleaner code
    println!("\n--- Pipeline Processing ---");
    
    if let Ok(pipeline) = WasmTransformPipeline::new(true) { // Enable caching
        // Build processing pipeline
        pipeline.add_transform("resize_256");
        pipeline.add_transform("center_crop_224");
        pipeline.add_transform("color_jitter");
        pipeline.add_transform("imagenet_normalize");
        pipeline.add_transform("to_tensor");
        
        println!("Pipeline length: {}", pipeline.length());
        
        // Execute pipeline
        if let Ok(final_result) = pipeline.execute(&original_tensor) {
            println!("Pipeline result shape: {:?}", final_result.shape());
            println!("Pipeline stats: {}", pipeline.get_stats());
        }
    }
    
    // Memory statistics
    println!("\n--- Memory Performance ---");
    println!("Pool stats: {}", MemoryManager::get_stats());
    println!("Cache efficiency: {}", MemoryManager::cache_efficiency());
    
    // Demonstrate random crop for data augmentation
    println!("\n--- Data Augmentation ---");
    if let Ok(random_crop) = WasmRandomCrop::new(200, 200, Some(10)) {
        if let Ok(augmented) = random_crop.apply(&original_tensor) {
            println!("Random crop result: {:?}", augmented.shape());
            println!("Augmentation: {}", random_crop.name());
        }
    }
    
    println!("\n=== Image Processing Demo Complete ===");
}

#[cfg(not(feature = "wasm"))]
fn main() {
    println!("WASM feature not enabled. Run with: cargo run --features wasm --example wasm_image_processing");
}