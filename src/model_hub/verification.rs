//! Model file verification and integrity checking
//! モデルファイル検証と整合性チェック

use std::path::Path;
use std::io::Read;
use crate::error::{RusTorchError, RusTorchResult};
use serde::{Serialize, Deserialize};
use digest::Digest;

/// Checksum types for model verification
/// モデル検証用チェックサムタイプ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Checksum {
    /// SHA-256 hash
    Sha256(String),
    /// MD5 hash (less secure, for compatibility)
    Md5(String),
    /// CRC32 (fastest, for basic integrity)
    Crc32(u32),
}

impl Checksum {
    /// Create SHA-256 checksum from hex string
    /// 16進文字列からSHA-256チェックサムを作成
    pub fn sha256(hex: &str) -> Self {
        Self::Sha256(hex.to_lowercase())
    }

    /// Create MD5 checksum from hex string
    /// 16進文字列からMD5チェックサムを作成
    pub fn md5(hex: &str) -> Self {
        Self::Md5(hex.to_lowercase())
    }

    /// Create CRC32 checksum from value
    /// 値からCRC32チェックサムを作成
    pub fn crc32(value: u32) -> Self {
        Self::Crc32(value)
    }
}

/// Model file verifier
/// モデルファイル検証器
pub struct ModelVerifier {
    /// Enable strict verification
    /// 厳密検証を有効化
    strict_mode: bool,
}

impl ModelVerifier {
    /// Create new verifier
    /// 新しい検証器を作成
    pub fn new() -> Self {
        Self {
            strict_mode: true,
        }
    }

    /// Create verifier with strict mode disabled
    /// 厳密モード無効で検証器を作成
    pub fn with_relaxed_mode() -> Self {
        Self {
            strict_mode: false,
        }
    }

    /// Verify file against expected checksum
    /// 期待されるチェックサムに対してファイルを検証
    pub fn verify_file<P: AsRef<Path>>(&self, path: P, expected: &Checksum) -> RusTorchResult<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(RusTorchError::FileNotFound(
                format!("File not found: {}", path.display())
            ));
        }

        let calculated = self.calculate_checksum(path, expected)?;
        
        if !self.checksums_match(&calculated, expected) {
            if self.strict_mode {
                return Err(RusTorchError::VerificationError(
                    format!("Checksum mismatch for {}: expected {:?}, got {:?}", 
                           path.display(), expected, calculated)
                ));
            } else {
                println!("Warning: Checksum mismatch for {} (continuing in relaxed mode)", 
                        path.display());
            }
        }

        Ok(())
    }

    /// Calculate checksum for file
    /// ファイルのチェックサムを計算
    pub fn calculate_checksum<P: AsRef<Path>>(&self, path: P, checksum_type: &Checksum) -> RusTorchResult<Checksum> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;

        match checksum_type {
            Checksum::Sha256(_) => {
                let mut hasher = sha2::Sha256::new();
                let mut buffer = [0; 8192];

                loop {
                    let bytes_read = file.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..bytes_read]);
                }

                use sha2::Digest;
                let hash = hasher.finalize();
                Ok(Checksum::Sha256(format!("{:x}", hash)))
            }
            Checksum::Md5(_) => {
                let mut context = md5::Context::new();
                let mut buffer = [0; 8192];

                loop {
                    let bytes_read = file.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    context.consume(&buffer[..bytes_read]);
                }

                let hash = context.compute();
                Ok(Checksum::Md5(format!("{:x}", hash)))
            }
            Checksum::Crc32(_) => {
                let mut hasher = crc32fast::Hasher::new();
                let mut buffer = [0; 8192];

                loop {
                    let bytes_read = file.read(&mut buffer)?;
                    if bytes_read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..bytes_read]);
                }

                let hash = hasher.finalize();
                Ok(Checksum::Crc32(hash))
            }
        }
    }

    /// Check if two checksums match
    /// 2つのチェックサムが一致するかチェック
    fn checksums_match(&self, a: &Checksum, b: &Checksum) -> bool {
        match (a, b) {
            (Checksum::Sha256(a), Checksum::Sha256(b)) => a == b,
            (Checksum::Md5(a), Checksum::Md5(b)) => a == b,
            (Checksum::Crc32(a), Checksum::Crc32(b)) => a == b,
            _ => false, // Different checksum types
        }
    }

    /// Verify PyTorch model file format
    /// PyTorchモデルファイル形式を検証
    pub fn verify_pytorch_format<P: AsRef<Path>>(&self, path: P) -> RusTorchResult<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 4];
        
        file.read_exact(&mut header)?;

        // Check for pickle magic number or ZIP magic (PyTorch uses both)
        let is_pickle = header[0] == 0x80; // Pickle protocol marker
        let is_zip = &header == b"PK\x03\x04" || &header == b"PK\x05\x06"; // ZIP file markers

        if !is_pickle && !is_zip {
            return Err(RusTorchError::InvalidModel(
                "File does not appear to be a valid PyTorch model (not pickle or ZIP format)"
            ));
        }

        Ok(())
    }

    /// Verify ONNX model file format
    /// ONNXモデルファイル形式を検証
    pub fn verify_onnx_format<P: AsRef<Path>>(&self, path: P) -> RusTorchResult<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 8];
        
        file.read_exact(&mut header)?;

        // ONNX files are protobuf format, check for protobuf magic
        // Protobuf doesn't have a fixed magic number, but ONNX models typically start with specific patterns
        let has_onnx_pattern = header.windows(4).any(|window| {
            // Common ONNX patterns in protobuf
            window == b"\x08\x01\x12" || // Common field encoding
            window == b"\x0a\x02\x08" || // Another common pattern
            window == b"ONNX"            // Sometimes contains literal "ONNX"
        });

        if !has_onnx_pattern {
            if self.strict_mode {
                return Err(RusTorchError::InvalidModel(
                    "File does not appear to be a valid ONNX model"
                ));
            } else {
                println!("Warning: ONNX format verification uncertain (continuing in relaxed mode)");
            }
        }

        Ok(())
    }

    /// Verify file is not corrupted (basic checks)
    /// ファイルが破損していないことを検証（基本チェック）
    pub fn verify_file_integrity<P: AsRef<Path>>(&self, path: P) -> RusTorchResult<()> {
        let path = path.as_ref();
        
        // Check file exists and is readable
        let metadata = std::fs::metadata(path)?;
        
        // Check file is not empty
        if metadata.len() == 0 {
            return Err(RusTorchError::InvalidModel("File is empty"));
        }

        // Check file is not too small to be a valid model
        if metadata.len() < 1024 {
            if self.strict_mode {
                return Err(RusTorchError::InvalidModel("File too small to be a valid model"));
            } else {
                println!("Warning: File seems very small for a model (continuing in relaxed mode)");
            }
        }

        // Try to read first and last bytes to ensure file is readable
        let mut file = std::fs::File::open(path)?;
        let mut first_byte = [0u8; 1];
        let mut last_byte = [0u8; 1];
        
        file.read_exact(&mut first_byte)?;
        
        if metadata.len() > 1 {
            use std::io::Seek;
            file.seek(std::io::SeekFrom::End(-1))?;
            file.read_exact(&mut last_byte)?;
        }

        Ok(())
    }
}

impl Default for ModelVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    #[test]
    fn test_checksum_creation() {
        let sha256 = Checksum::sha256("a1b2c3d4e5f6");
        matches!(sha256, Checksum::Sha256(_));

        let md5 = Checksum::md5("1234567890abcdef");
        matches!(md5, Checksum::Md5(_));

        let crc32 = Checksum::crc32(0x12345678);
        matches!(crc32, Checksum::Crc32(0x12345678));
    }

    #[test]
    fn test_verifier_creation() {
        let verifier = ModelVerifier::new();
        assert!(verifier.strict_mode);

        let relaxed = ModelVerifier::with_relaxed_mode();
        assert!(!relaxed.strict_mode);
    }

    #[test]
    fn test_calculate_sha256() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, b"hello world").unwrap();

        let verifier = ModelVerifier::new();
        let expected = Checksum::Sha256("dummy".to_string());
        let result = verifier.calculate_checksum(&test_file, &expected);
        
        assert!(result.is_ok());
        if let Ok(Checksum::Sha256(hash)) = result {
            // "hello world" SHA-256 is known
            assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
        }
    }

    #[test]
    fn test_calculate_crc32() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, b"hello world").unwrap();

        let verifier = ModelVerifier::new();
        let expected = Checksum::Crc32(0);
        let result = verifier.calculate_checksum(&test_file, &expected);
        
        assert!(result.is_ok());
        if let Ok(Checksum::Crc32(hash)) = result {
            // "hello world" CRC32 is known
            assert_eq!(hash, 0x0d4a1185);
        }
    }

    #[test]
    fn test_file_integrity_checks() {
        let temp_dir = TempDir::new().unwrap();
        let verifier = ModelVerifier::new();

        // Test with valid file
        let valid_file = temp_dir.path().join("valid.txt");
        std::fs::write(&valid_file, b"valid model data with sufficient size").unwrap();
        assert!(verifier.verify_file_integrity(&valid_file).is_ok());

        // Test with empty file (should fail in strict mode)
        let empty_file = temp_dir.path().join("empty.txt");
        std::fs::write(&empty_file, b"").unwrap();
        assert!(verifier.verify_file_integrity(&empty_file).is_err());

        // Test with small file (should fail in strict mode)
        let small_file = temp_dir.path().join("small.txt");
        std::fs::write(&small_file, b"tiny").unwrap();
        assert!(verifier.verify_file_integrity(&small_file).is_err());

        // Test with relaxed verifier
        let relaxed = ModelVerifier::with_relaxed_mode();
        assert!(relaxed.verify_file_integrity(&small_file).is_ok());
    }

    #[test]
    fn test_pytorch_format_verification() {
        let temp_dir = TempDir::new().unwrap();
        let verifier = ModelVerifier::new();

        // Test with pickle-like header
        let pickle_file = temp_dir.path().join("pickle.pth");
        let mut file = std::fs::File::create(&pickle_file).unwrap();
        file.write_all(&[0x80, 0x02]).unwrap(); // Pickle protocol 2
        file.write_all(b"rest of pickle data").unwrap();
        drop(file);

        assert!(verifier.verify_pytorch_format(&pickle_file).is_ok());

        // Test with ZIP-like header (PyTorch also uses ZIP format)
        let zip_file = temp_dir.path().join("zip.pth");
        let mut file = std::fs::File::create(&zip_file).unwrap();
        file.write_all(b"PK\x03\x04").unwrap(); // ZIP header
        file.write_all(b"rest of zip data").unwrap();
        drop(file);

        assert!(verifier.verify_pytorch_format(&zip_file).is_ok());

        // Test with invalid header
        let invalid_file = temp_dir.path().join("invalid.pth");
        std::fs::write(&invalid_file, b"not a pytorch file").unwrap();
        assert!(verifier.verify_pytorch_format(&invalid_file).is_err());
    }

    #[test]
    fn test_verify_file_with_checksum() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, b"hello world").unwrap();

        let verifier = ModelVerifier::new();
        
        // Test with correct SHA-256
        let correct_sha256 = Checksum::sha256("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
        assert!(verifier.verify_file(&test_file, &correct_sha256).is_ok());

        // Test with incorrect SHA-256
        let incorrect_sha256 = Checksum::sha256("incorrect_hash");
        assert!(verifier.verify_file(&test_file, &incorrect_sha256).is_err());

        // Test with correct CRC32
        let correct_crc32 = Checksum::crc32(0x0d4a1185);
        assert!(verifier.verify_file(&test_file, &correct_crc32).is_ok());

        // Test with incorrect CRC32
        let incorrect_crc32 = Checksum::crc32(0x12345678);
        assert!(verifier.verify_file(&test_file, &incorrect_crc32).is_err());
    }

    #[test]
    fn test_relaxed_mode_verification() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, b"hello world").unwrap();

        let relaxed_verifier = ModelVerifier::with_relaxed_mode();
        
        // Test with incorrect checksum in relaxed mode (should not fail)
        let incorrect_sha256 = Checksum::sha256("incorrect_hash");
        assert!(relaxed_verifier.verify_file(&test_file, &incorrect_sha256).is_ok());
    }
}