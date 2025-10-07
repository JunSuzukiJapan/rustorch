/// Debug GGUF scale factors (d, dmin) to verify they are read correctly
///
/// Specifically check if d and dmin values match llama.cpp expectations

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn read_u16<R: Read>(reader: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn main() -> std::io::Result<()> {
    println!("🔍 GGUF Scale Factor Debug\n");

    let model_path = std::env::var("HOME").unwrap() +
        "/.rustorch/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    let file = File::open(&model_path)?;
    let mut reader = BufReader::new(file);

    // First, we need to find the data offset
    // For this specific model, we know from previous logs: data_offset = 1709440
    let data_offset: u64 = 1709440;

    println!("📂 Seeking to data offset: {}", data_offset);
    reader.seek(SeekFrom::Start(data_offset))?;

    // We know token_embd.weight is first (Q4_K format)
    // Shape: [2048, 32000] = 65,536,000 elements
    // Q4_K block size: 256 elements/block = 256,000 blocks
    // Each Q4_K block: 144 bytes

    println!("\n═══════════════════════════════════════");
    println!("📊 token_embd.weight (Q4_K) - First Block");
    println!("═══════════════════════════════════════\n");

    // Read first block's d and dmin
    let d_bits = read_u16(&mut reader)?;
    let dmin_bits = read_u16(&mut reader)?;
    let d = half::f16::from_bits(d_bits).to_f32();
    let dmin = half::f16::from_bits(dmin_bits).to_f32();

    println!("First Q4_K block:");
    println!("   d (bits): 0x{:04x}", d_bits);
    println!("   d (f32): {:.10}", d);
    println!("   dmin (bits): 0x{:04x}", dmin_bits);
    println!("   dmin (f32): {:.10}", dmin);

    // Read scales
    let mut scales = [0u8; 12];
    reader.read_exact(&mut scales)?;
    println!("   scales[0]: {} (binary: {:08b})", scales[0], scales[0]);
    println!("   scales[1]: {} (binary: {:08b})", scales[1], scales[1]);
    println!("   scales[4]: {} (binary: {:08b})", scales[4], scales[4]);
    println!("   scales[5]: {} (binary: {:08b})", scales[5], scales[5]);

    // Extract scale/min like RusTorch does
    let sc0 = scales[0] & 63;
    let mn0 = scales[0 + 4] & 63;
    println!("\n   Extracted (j=0):");
    println!("      scale: {} (0-63 range)", sc0);
    println!("      min: {} (0-63 range)", mn0);
    println!("      d1 = d * sc = {:.10} * {} = {:.10}", d, sc0, d * sc0 as f32);
    println!("      m1 = dmin * mn = {:.10} * {} = {:.10}", dmin, mn0, dmin * mn0 as f32);

    // Read first few quantized values
    let mut qs = vec![0u8; 128];
    reader.read_exact(&mut qs)?;
    println!("\n   First quantized byte: {} (binary: {:08b})", qs[0], qs[0]);
    let lower_nibble = qs[0] & 0x0F;
    let upper_nibble = qs[0] >> 4;
    println!("      lower nibble (q[0]): {}", lower_nibble);
    println!("      upper nibble (q[1]): {}", upper_nibble);

    // Calculate dequantized value
    let d1 = d * sc0 as f32;
    let m1 = dmin * mn0 as f32;
    let dequant_0 = d1 * lower_nibble as f32 - m1;
    let dequant_1 = d1 * upper_nibble as f32 - m1;
    println!("\n   Dequantized values:");
    println!("      element[0]: {:.10}", dequant_0);
    println!("      element[1]: {:.10}", dequant_1);

    println!("\n═══════════════════════════════════════");
    println!("📊 Analysis");
    println!("═══════════════════════════════════════\n");

    if d.abs() < 0.0001 {
        println!("❌ d value is too small! Expected ~0.01-0.1 range");
        println!("   This would cause all weights to be ~1000x too small");
    } else if d.abs() > 1.0 {
        println!("⚠️  d value is very large! Expected ~0.01-0.1 range");
    } else {
        println!("✅ d value seems reasonable ({:.10})", d);
    }

    if dequant_0.abs() < 0.00001 {
        println!("❌ Dequantized value is too small!");
        println!("   Expected embedding values ~0.01-0.1 range");
    } else {
        println!("✅ Dequantized value magnitude seems OK");
    }

    Ok(())
}
