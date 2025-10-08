/// Fixed Q6_K dequantization matching llama.cpp exactly
///
/// This implementation matches dequantize_row_q6_K from llama.cpp/ggml-quants.c

use std::fs::File;
use std::io::{BufReader, Read};

const QK_K: usize = 256;

pub fn dequantize_q6_k_fixed(
    reader: &mut BufReader<File>,
    num_elements: usize,
) -> Result<Vec<f64>, String> {
    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let mut output = vec![0.0f64; num_elements];

    for block_idx in 0..num_blocks {
        let block_start = block_idx * QK_K;

        // Read super-block data (210 bytes total)
        // - ql[128]: lower 4 bits
        // - qh[64]: upper 2 bits
        // - sc[16]: scales
        // - d (f16): super-scale

        let mut ql = vec![0u8; 128];
        reader.read_exact(&mut ql).map_err(|e| format!("Read ql failed: {}", e))?;

        let mut qh = vec![0u8; 64];
        reader.read_exact(&mut qh).map_err(|e| format!("Read qh failed: {}", e))?;

        let mut sc = vec![0i8; 16];
        for scale in &mut sc {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf).map_err(|e| format!("Read scale failed: {}", e))?;
            *scale = buf[0] as i8;
        }

        let mut d_buf = [0u8; 2];
        reader.read_exact(&mut d_buf).map_err(|e| format!("Read d failed: {}", e))?;
        let d_bits = u16::from_le_bytes(d_buf);
        let d = half::f16::from_bits(d_bits).to_f32();

        // Dequantize following llama.cpp exactly
        let mut y_idx = block_start;
        let mut ql_idx = 0;
        let mut qh_idx = 0;
        let mut sc_idx = 0;

        // Process in 2 chunks of 128 elements each
        for _n in 0..2 {
            // Process 32 iterations, each producing 4 values
            for l in 0..32 {
                let is = l / 16;

                // Extract 4 quantized values
                let q1 = (((ql[ql_idx + l] & 0xF) | (((qh[qh_idx + l] >> 0) & 3) << 4)) as i8) - 32;
                let q2 = (((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) as i8) - 32;
                let q3 = (((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) as i8) - 32;
                let q4 = (((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) as i8) - 32;

                // Write to output at correct positions
                if y_idx + l < num_elements {
                    output[y_idx + l] = (d * sc[sc_idx + is] as f32 * q1 as f32) as f64;
                }
                if y_idx + l + 32 < num_elements {
                    output[y_idx + l + 32] = (d * sc[sc_idx + is + 2] as f32 * q2 as f32) as f64;
                }
                if y_idx + l + 64 < num_elements {
                    output[y_idx + l + 64] = (d * sc[sc_idx + is + 4] as f32 * q3 as f32) as f64;
                }
                if y_idx + l + 96 < num_elements {
                    output[y_idx + l + 96] = (d * sc[sc_idx + is + 6] as f32 * q4 as f32) as f64;
                }
            }

            // Advance indices like llama.cpp pointers
            y_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }

    Ok(output)
}
