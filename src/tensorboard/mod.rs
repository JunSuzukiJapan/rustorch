//! TensorBoard integration for RusTorch
//! RusTorch用TensorBoard統合

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub mod event_writer;
pub mod summary;

use self::event_writer::EventWriter;
use self::summary::Summary;

/// TensorBoard writer for logging training metrics
/// 訓練メトリクスログ用TensorBoardライター
pub struct SummaryWriter {
    /// Log directory
    /// ログディレクトリ
    log_dir: PathBuf,
    /// Event writer
    /// イベントライター
    event_writer: EventWriter,
    /// Global step counter
    /// グローバルステップカウンタ
    global_step: usize,
    /// Flush interval
    /// フラッシュ間隔
    flush_interval: usize,
    /// Pending summaries
    /// 保留中のサマリー
    pending_summaries: Vec<Summary>,
}

impl SummaryWriter {
    /// Create a new summary writer
    /// 新しいサマリーライターを作成
    pub fn new(log_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        
        // Create log directory if it doesn't exist
        fs::create_dir_all(&log_dir)?;
        
        // Create event file
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        #[cfg(not(target_arch = "wasm32"))]
        let hostname = hostname::get()
            .unwrap_or_else(|_| std::ffi::OsString::from("unknown"))
            .to_string_lossy()
            .to_string();
        
        #[cfg(target_arch = "wasm32")]
        let hostname = "wasm-browser".to_string();
        
        let filename = format!(
            "events.out.tfevents.{}.{}",
            timestamp, hostname
        );
        
        let event_path = log_dir.join(filename);
        let event_writer = EventWriter::new(event_path)?;
        
        Ok(Self {
            log_dir,
            event_writer,
            global_step: 0,
            flush_interval: 10,
            pending_summaries: Vec::new(),
        })
    }
    
    /// Add scalar value
    /// スカラー値を追加
    pub fn add_scalar(&mut self, tag: &str, value: f32, step: Option<usize>) {
        let step = step.unwrap_or(self.global_step);
        let summary = Summary::scalar(tag, value, step);
        self.pending_summaries.push(summary);
        
        if self.pending_summaries.len() >= self.flush_interval {
            self.flush();
        }
    }
    
    /// Add histogram
    /// ヒストグラムを追加
    pub fn add_histogram(&mut self, tag: &str, values: &[f32], step: Option<usize>) {
        let step = step.unwrap_or(self.global_step);
        let summary = Summary::histogram(tag, values, step);
        self.pending_summaries.push(summary);
        
        if self.pending_summaries.len() >= self.flush_interval {
            self.flush();
        }
    }
    
    /// Add image
    /// 画像を追加
    pub fn add_image(&mut self, tag: &str, image: &ImageData, step: Option<usize>) {
        let step = step.unwrap_or(self.global_step);
        let summary = Summary::image(tag, image, step);
        self.pending_summaries.push(summary);
        
        if self.pending_summaries.len() >= self.flush_interval {
            self.flush();
        }
    }
    
    /// Add text
    /// テキストを追加
    pub fn add_text(&mut self, tag: &str, text: &str, step: Option<usize>) {
        let step = step.unwrap_or(self.global_step);
        let summary = Summary::text(tag, text, step);
        self.pending_summaries.push(summary);
        
        if self.pending_summaries.len() >= self.flush_interval {
            self.flush();
        }
    }
    
    /// Add graph (computational graph)
    /// グラフ（計算グラフ）を追加
    pub fn add_graph(&mut self, graph: &GraphDef) {
        let summary = Summary::graph(graph);
        self.pending_summaries.push(summary);
        self.flush();
    }
    
    /// Add embedding projector data
    /// 埋め込みプロジェクタデータを追加
    pub fn add_embedding(
        &mut self,
        mat: &[Vec<f32>],
        metadata: Option<Vec<String>>,
        tag: Option<&str>,
    ) -> std::io::Result<()> {
        let tag = tag.unwrap_or("default");
        let projector_dir = self.log_dir.join("projector");
        fs::create_dir_all(&projector_dir)?;
        
        // Write tensor TSV
        let tensor_path = projector_dir.join(format!("{}_tensor.tsv", tag));
        let mut tensor_file = File::create(tensor_path)?;
        
        for vec in mat {
            let line: String = vec.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("\t");
            writeln!(tensor_file, "{}", line)?;
        }
        
        // Write metadata TSV if provided
        if let Some(metadata) = metadata {
            let metadata_path = projector_dir.join(format!("{}_metadata.tsv", tag));
            let mut metadata_file = File::create(metadata_path)?;
            for label in metadata {
                writeln!(metadata_file, "{}", label)?;
            }
        }
        
        // Create projector config
        let config = ProjectorConfig {
            embeddings: vec![EmbeddingInfo {
                tensor_name: format!("{}_tensor.tsv", tag),
                metadata_path: Some(format!("{}_metadata.tsv", tag)),
            }],
        };
        
        let config_path = projector_dir.join("projector_config.pbtxt");
        let config_content = format_projector_config(&config);
        fs::write(config_path, config_content)?;
        
        Ok(())
    }
    
    /// Add PR curve
    /// PR曲線を追加
    pub fn add_pr_curve(
        &mut self,
        tag: &str,
        labels: &[bool],
        predictions: &[f32],
        step: Option<usize>,
    ) {
        let step = step.unwrap_or(self.global_step);
        let summary = Summary::pr_curve(tag, labels, predictions, step);
        self.pending_summaries.push(summary);
        
        if self.pending_summaries.len() >= self.flush_interval {
            self.flush();
        }
    }
    
    /// Flush pending summaries
    /// 保留中のサマリーをフラッシュ
    pub fn flush(&mut self) {
        for summary in self.pending_summaries.drain(..) {
            self.event_writer.write_summary(summary);
        }
        self.event_writer.flush();
    }
    
    /// Increment global step
    /// グローバルステップをインクリメント
    pub fn step(&mut self) {
        self.global_step += 1;
    }
    
    /// Close the writer
    /// ライターを閉じる
    pub fn close(mut self) {
        self.flush();
    }
}

/// Image data for TensorBoard
/// TensorBoard用画像データ
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Height
    pub height: u32,
    /// Width
    pub width: u32,
    /// Channels (1 for grayscale, 3 for RGB, 4 for RGBA)
    pub channels: u32,
    /// Pixel data (flattened)
    pub data: Vec<u8>,
}

/// Graph definition for computational graphs
/// 計算グラフのグラフ定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDef {
    /// Nodes in the graph
    pub nodes: Vec<NodeDef>,
    /// Edges in the graph
    pub edges: Vec<EdgeDef>,
}

/// Node definition in graph
/// グラフ内のノード定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDef {
    /// Node name
    pub name: String,
    /// Operation type
    pub op: String,
    /// Input nodes
    pub inputs: Vec<String>,
    /// Node attributes
    pub attrs: HashMap<String, String>,
}

/// Edge definition in graph
/// グラフ内のエッジ定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDef {
    /// Source node
    pub source: String,
    /// Target node
    pub target: String,
    /// Edge label
    pub label: Option<String>,
}

/// Projector configuration
/// プロジェクタ設定
#[derive(Debug, Clone)]
struct ProjectorConfig {
    embeddings: Vec<EmbeddingInfo>,
}

/// Embedding information
/// 埋め込み情報
#[derive(Debug, Clone)]
struct EmbeddingInfo {
    tensor_name: String,
    metadata_path: Option<String>,
}

/// Format projector config to protobuf text format
/// プロジェクタ設定をprotobufテキスト形式にフォーマット
fn format_projector_config(config: &ProjectorConfig) -> String {
    let mut result = String::new();
    
    for embedding in &config.embeddings {
        result.push_str("embeddings {\n");
        result.push_str(&format!("  tensor_name: \"{}\"\n", embedding.tensor_name));
        
        if let Some(metadata_path) = &embedding.metadata_path {
            result.push_str(&format!("  metadata_path: \"{}\"\n", metadata_path));
        }
        
        result.push_str("}\n");
    }
    
    result
}

/// Python-compatible API for seamless integration
/// シームレス統合用Python互換API
pub mod python_compat {
    use super::*;
    
    /// Create writer with automatic directory naming
    /// 自動ディレクトリ命名でライターを作成
    pub fn create_writer(base_dir: Option<&str>) -> std::io::Result<SummaryWriter> {
        let base_dir = base_dir.unwrap_or("runs");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let log_dir = format!("{}/experiment_{}", base_dir, timestamp);
        SummaryWriter::new(log_dir)
    }
    
    /// Quick logging function
    /// クイックログ関数
    pub fn log_scalar(writer: &mut SummaryWriter, tag: &str, value: f32) {
        writer.add_scalar(tag, value, None);
        writer.step();
    }
}

/// Macro for easy TensorBoard logging
/// 簡単なTensorBoardログ用マクロ
#[macro_export]
macro_rules! tb_log {
    ($writer:expr, scalar: $tag:expr, $value:expr) => {
        $writer.add_scalar($tag, $value, None);
    };
    
    ($writer:expr, histogram: $tag:expr, $values:expr) => {
        $writer.add_histogram($tag, $values, None);
    };
    
    ($writer:expr, text: $tag:expr, $text:expr) => {
        $writer.add_text($tag, $text, None);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_summary_writer_creation() {
        let dir = tempdir().unwrap();
        let _writer = SummaryWriter::new(dir.path()).unwrap();
        
        // Check that event file was created
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        
        assert_eq!(entries.len(), 1);
        assert!(entries[0].file_name().to_string_lossy().starts_with("events.out.tfevents"));
    }
    
    #[test]
    fn test_scalar_logging() {
        let dir = tempdir().unwrap();
        let mut writer = SummaryWriter::new(dir.path()).unwrap();
        
        writer.add_scalar("loss", 0.5, Some(0));
        writer.add_scalar("accuracy", 0.95, Some(0));
        writer.flush();
        
        // Verify file exists and has content
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        
        assert!(entries[0].metadata().unwrap().len() > 0);
    }
    
    #[test]
    fn test_histogram_logging() {
        let dir = tempdir().unwrap();
        let mut writer = SummaryWriter::new(dir.path()).unwrap();
        
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        writer.add_histogram("weights", &values, Some(0));
        writer.flush();
    }
}