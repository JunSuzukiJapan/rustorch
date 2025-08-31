//! HTTP model downloader with progress tracking
//! 進捗追跡付きHTTPモデルダウンローダー

use crate::error::{RusTorchError, RusTorchResult};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Download progress information
/// ダウンロード進捗情報
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Total bytes to download
    /// ダウンロード総バイト数
    pub total_bytes: u64,
    /// Downloaded bytes
    /// ダウンロード済みバイト数
    pub downloaded_bytes: u64,
    /// Download speed in bytes per second
    /// ダウンロード速度（バイト/秒）
    pub speed_bps: f64,
    /// Estimated time remaining in seconds
    /// 推定残り時間（秒）
    pub eta_seconds: f64,
}

impl DownloadProgress {
    /// Get download percentage (0.0 to 100.0)
    /// ダウンロード完了率を取得（0.0～100.0）
    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.downloaded_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }

    /// Format speed for display
    /// 表示用速度フォーマット
    pub fn format_speed(&self) -> String {
        if self.speed_bps < 1024.0 {
            format!("{:.1} B/s", self.speed_bps)
        } else if self.speed_bps < 1024.0 * 1024.0 {
            format!("{:.1} KB/s", self.speed_bps / 1024.0)
        } else {
            format!("{:.1} MB/s", self.speed_bps / (1024.0 * 1024.0))
        }
    }

    /// Format size for display
    /// 表示用サイズフォーマット
    pub fn format_size(&self) -> String {
        let total_mb = self.total_bytes as f64 / (1024.0 * 1024.0);
        let downloaded_mb = self.downloaded_bytes as f64 / (1024.0 * 1024.0);
        format!("{:.1}/{:.1} MB", downloaded_mb, total_mb)
    }
}

/// Download error types
/// ダウンロードエラータイプ
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    #[error("Download interrupted")]
    Interrupted,
    #[error("File verification failed")]
    VerificationFailed,
}

impl From<DownloadError> for RusTorchError {
    fn from(error: DownloadError) -> Self {
        RusTorchError::DownloadError(error.to_string())
    }
}

/// HTTP model downloader
/// HTTPモデルダウンローダー
pub struct ModelDownloader {
    /// HTTP client for downloads
    client: reqwest::Client,
    /// Maximum number of retry attempts
    /// 最大再試行回数
    max_retries: usize,
    /// Timeout for each request in seconds
    /// 各リクエストのタイムアウト（秒）
    timeout_seconds: u64,
}

impl ModelDownloader {
    /// Create new downloader with default settings
    /// デフォルト設定で新しいダウンローダーを作成
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minutes default
            .user_agent("RusTorch/0.5.2")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            max_retries: 3,
            timeout_seconds: 300,
        }
    }

    /// Create downloader with custom settings
    /// カスタム設定でダウンローダーを作成
    pub fn with_config(max_retries: usize, timeout_seconds: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .user_agent("RusTorch/0.5.2")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            max_retries,
            timeout_seconds,
        }
    }

    /// Download file with progress callback
    /// 進捗コールバック付きファイルダウンロード
    pub async fn download_with_progress<F, P>(
        &self,
        url: &str,
        output_path: P,
        mut progress_callback: F,
    ) -> Result<(), DownloadError>
    where
        F: FnMut(DownloadProgress),
        P: AsRef<Path>,
    {
        let output_path = output_path.as_ref();

        // Create parent directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= self.max_retries {
            match self
                .try_download_with_progress(url, output_path, &mut progress_callback)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                    if attempts <= self.max_retries {
                        println!(
                            "Download attempt {} failed, retrying... ({}/{})",
                            attempts, attempts, self.max_retries
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or(DownloadError::Interrupted))
    }

    /// Single download attempt with progress
    /// 進捗付き単一ダウンロード試行
    async fn try_download_with_progress<F, P>(
        &self,
        url: &str,
        output_path: P,
        progress_callback: &mut F,
    ) -> Result<(), DownloadError>
    where
        F: FnMut(DownloadProgress),
        P: AsRef<Path>,
    {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| DownloadError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DownloadError::HttpError(format!(
                "HTTP {}: {}",
                response.status(),
                response
                    .status()
                    .canonical_reason()
                    .unwrap_or("Unknown error")
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let start_time = Instant::now();

        let mut file = std::fs::File::create(&output_path)?;
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| DownloadError::HttpError(e.to_string()))?;
            file.write_all(&chunk)?;

            downloaded += chunk.len() as u64;
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                downloaded as f64 / elapsed
            } else {
                0.0
            };
            let eta = if speed > 0.0 && total_size > 0 {
                (total_size - downloaded) as f64 / speed
            } else {
                0.0
            };

            let progress = DownloadProgress {
                total_bytes: total_size,
                downloaded_bytes: downloaded,
                speed_bps: speed,
                eta_seconds: eta,
            };

            progress_callback(progress);
        }

        file.flush()?;
        Ok(())
    }

    /// Simple download without progress tracking
    /// 進捗追跡なしのシンプルダウンロード
    pub async fn download<P: AsRef<Path>>(
        &self,
        url: &str,
        output_path: P,
    ) -> Result<(), DownloadError> {
        self.download_with_progress(url, output_path, |_| {}).await
    }

    /// Check if URL is accessible
    /// URLがアクセス可能かチェック
    pub async fn check_url(&self, url: &str) -> Result<(u16, u64), DownloadError> {
        let response = self
            .client
            .head(url)
            .send()
            .await
            .map_err(|e| DownloadError::HttpError(e.to_string()))?;

        let status = response.status().as_u16();
        let content_length = response.content_length().unwrap_or(0);

        Ok((status, content_length))
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_download_progress() {
        let progress = DownloadProgress {
            total_bytes: 1000000,
            downloaded_bytes: 500000,
            speed_bps: 1024.0 * 512.0, // 512 KB/s
            eta_seconds: 976.5,
        };

        assert_eq!(progress.percentage(), 50.0);
        assert_eq!(progress.format_speed(), "512.0 KB/s");
        assert!(progress.format_size().contains("0.5/1.0 MB"));
    }

    #[test]
    fn test_downloader_creation() {
        let downloader = ModelDownloader::new();
        assert_eq!(downloader.max_retries, 3);
        assert_eq!(downloader.timeout_seconds, 300);
    }

    #[test]
    fn test_downloader_custom_config() {
        let downloader = ModelDownloader::with_config(5, 600);
        assert_eq!(downloader.max_retries, 5);
        assert_eq!(downloader.timeout_seconds, 600);
    }

    #[tokio::test]
    async fn test_url_check() {
        let downloader = ModelDownloader::new();

        // Test with a reliable URL (Google's public DNS)
        // This should work in most environments
        let result = downloader.check_url("https://www.google.com").await;

        // Don't assert success since network conditions vary
        // Just ensure the function doesn't panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_invalid_url() {
        let downloader = ModelDownloader::new();
        let result = downloader.check_url("not-a-valid-url").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_download_to_temp_file() {
        let downloader = ModelDownloader::new();
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_download.txt");

        // Test with a small, reliable URL
        // Using httpbin.org which provides reliable test endpoints
        let url = "https://httpbin.org/bytes/1024"; // 1KB test file

        let result = downloader.download(url, &output_path).await;

        // Don't assert success since network conditions vary
        // Just ensure the function doesn't panic and creates file structure
        if result.is_ok() {
            assert!(output_path.exists());
        }
    }
}
