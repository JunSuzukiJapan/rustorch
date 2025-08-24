//! TensorBoard event file writer
//! TensorBoardイベントファイルライター

use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use crc32fast::Hasher;
use super::summary::Summary;

/// Event writer for TensorBoard format
/// TensorBoard形式のイベントライター
pub struct EventWriter {
    /// File writer
    writer: BufWriter<File>,
    /// Event counter
    event_count: u64,
}

impl EventWriter {
    /// Create new event writer
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write TensorBoard file header
        let header = b"";  // TensorBoard expects empty header for now
        writer.write_all(header)?;
        
        Ok(Self {
            writer,
            event_count: 0,
        })
    }
    
    /// Write a summary to the event file
    pub fn write_summary(&mut self, summary: Summary) {
        let event = Event {
            wall_time: get_wall_time(),
            step: summary.step as i64,
            summary: Some(summary),
        };
        
        self.write_event(event);
    }
    
    /// Write an event to the file
    fn write_event(&mut self, event: Event) {
        // Serialize event to bytes (simplified - real implementation needs protobuf)
        let event_bytes = serialize_event(&event);
        
        // Write TFRecord format
        self.write_record(&event_bytes);
        
        self.event_count += 1;
    }
    
    /// Write a record in TFRecord format
    fn write_record(&mut self, data: &[u8]) {
        // TFRecord format:
        // uint64 length
        // uint32 masked_crc32 of length
        // byte data[length]
        // uint32 masked_crc32 of data
        
        let length = data.len() as u64;
        
        // Write length
        self.writer.write_all(&length.to_le_bytes()).unwrap();
        
        // Write CRC of length
        let length_crc = masked_crc32(&length.to_le_bytes());
        self.writer.write_all(&length_crc.to_le_bytes()).unwrap();
        
        // Write data
        self.writer.write_all(data).unwrap();
        
        // Write CRC of data
        let data_crc = masked_crc32(data);
        self.writer.write_all(&data_crc.to_le_bytes()).unwrap();
    }
    
    /// Flush the writer
    pub fn flush(&mut self) {
        self.writer.flush().unwrap();
    }
}

/// Event structure for TensorBoard
#[derive(Debug, Clone)]
struct Event {
    wall_time: f64,
    step: i64,
    summary: Option<Summary>,
}

/// Get current wall time in seconds
fn get_wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Calculate masked CRC32 (TensorBoard format)
fn masked_crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    let crc = hasher.finalize();
    
    // Mask the CRC as per TensorBoard format
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
}

/// Serialize event to bytes (simplified JSON format)
/// 実際の実装ではprotobufを使用する必要があります
fn serialize_event(event: &Event) -> Vec<u8> {
    // Simplified JSON serialization for demonstration
    // Real implementation should use protobuf
    let json = serde_json::json!({
        "wall_time": event.wall_time,
        "step": event.step,
        "summary": event.summary.as_ref().map(|s| {
            serde_json::json!({
                "tag": s.tag,
                "step": s.step,
                "value": match &s.value {
                    super::summary::SummaryValue::Scalar(v) => {
                        serde_json::json!({ "scalar": v })
                    },
                    super::summary::SummaryValue::Histogram(h) => {
                        serde_json::json!({ "histogram": h })
                    },
                    super::summary::SummaryValue::Image(img) => {
                        serde_json::json!({ 
                            "image": {
                                "height": img.height,
                                "width": img.width,
                                "colorspace": img.channels,
                            }
                        })
                    },
                    super::summary::SummaryValue::Text(t) => {
                        serde_json::json!({ "text": t })
                    },
                    _ => serde_json::json!(null),
                }
            })
        })
    });
    
    json.to_string().into_bytes()
}

impl Drop for EventWriter {
    fn drop(&mut self) {
        self.flush();
    }
}