//! Error context and diagnostic information
//! エラーコンテキストと診断情報

use crate::error::RusTorchError;
use std::collections::HashMap;
use std::fmt;

/// Error context for providing additional diagnostic information
/// 追加的な診断情報を提供するエラーコンテキスト
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed
    /// 実行されていた操作
    pub operation: String,
    
    /// File location where error occurred
    /// エラーが発生したファイル位置
    pub location: Option<ErrorLocation>,
    
    /// Additional metadata
    /// 追加のメタデータ
    pub metadata: HashMap<String, String>,
    
    /// Stack trace of operations
    /// 操作のスタックトレース
    pub stack_trace: Vec<String>,
}

/// File location information
/// ファイル位置情報
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// Source file name
    /// ソースファイル名
    pub file: String,
    
    /// Line number
    /// 行番号
    pub line: u32,
    
    /// Column number
    /// 列番号
    pub column: u32,
}

impl ErrorContext {
    /// Create a new error context
    /// 新しいエラーコンテキストを作成
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            location: None,
            metadata: HashMap::new(),
            stack_trace: Vec::new(),
        }
    }
    
    /// Add metadata to the context
    /// コンテキストにメタデータを追加
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Add location information
    /// 位置情報を追加
    pub fn with_location(mut self, file: impl Into<String>, line: u32, column: u32) -> Self {
        self.location = Some(ErrorLocation {
            file: file.into(),
            line,
            column,
        });
        self
    }
    
    /// Add operation to stack trace
    /// スタックトレースに操作を追加
    pub fn push_operation(mut self, operation: impl Into<String>) -> Self {
        self.stack_trace.push(operation.into());
        self
    }
    
    /// Get formatted context information
    /// フォーマットされたコンテキスト情報を取得
    pub fn format_context(&self) -> String {
        let mut context = format!("Operation: {}", self.operation);
        
        if let Some(ref location) = self.location {
            context.push_str(&format!("\nLocation: {}:{}:{}", location.file, location.line, location.column));
        }
        
        if !self.metadata.is_empty() {
            context.push_str("\nMetadata:");
            for (key, value) in &self.metadata {
                context.push_str(&format!("\n  {}: {}", key, value));
            }
        }
        
        if !self.stack_trace.is_empty() {
            context.push_str("\nStack trace:");
            for (i, operation) in self.stack_trace.iter().rev().enumerate() {
                context.push_str(&format!("\n  {}: {}", i, operation));
            }
        }
        
        context
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_context())
    }
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Trait for adding context to errors
/// エラーにコンテキストを追加するトレイト
pub trait WithContext<T> {
    /// Add context to the error
    /// エラーにコンテキストを追加
    fn with_context<F>(self, f: F) -> Result<T, RusTorchError>
    where
        F: FnOnce() -> ErrorContext;
        
    /// Add simple operation context
    /// 簡単な操作コンテキストを追加
    fn with_operation(self, operation: &str) -> Result<T, RusTorchError>;
}

impl<T, E> WithContext<T> for Result<T, E>
where
    E: Into<RusTorchError>,
{
    fn with_context<F>(self, f: F) -> Result<T, RusTorchError>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| {
            let mut error: RusTorchError = e.into();
            let context = f();
            
            // Enhance error with context information
            match &mut error {
                RusTorchError::TensorOp { message, .. } => {
                    *message = format!("{}\n{}", message, context.format_context());
                }
                RusTorchError::Device { message, .. } => {
                    *message = format!("{}\n{}", message, context.format_context());
                }
                RusTorchError::Gpu { message, .. } => {
                    *message = format!("{}\n{}", message, context.format_context());
                }
                _ => {
                    // For other error types, we can't clone, so just enhance the message
                    // 他のエラータイプでは、クローンできないのでメッセージだけ拡張
                    // (this branch handles non-cloneable errors)
                }
            }
            
            error
        })
    }
    
    fn with_operation(self, operation: &str) -> Result<T, RusTorchError> {
        self.with_context(|| ErrorContext::new(operation))
    }
}

/// Macro for creating error context with file location
/// ファイル位置付きのエラーコンテキストを作成するマクロ
#[macro_export]
macro_rules! error_context {
    ($operation:expr) => {
        $crate::error::context::ErrorContext::new($operation)
            .with_location(file!(), line!(), column!())
    };
    ($operation:expr, $($key:expr => $value:expr),+) => {
        {
            let mut context = $crate::error::context::ErrorContext::new($operation)
                .with_location(file!(), line!(), column!());
            $(
                context = context.with_metadata($key, $value);
            )+
            context
        }
    };
}

/// Macro for adding context to results
/// 結果にコンテキストを追加するマクロ
#[macro_export]
macro_rules! with_context {
    ($result:expr, $operation:expr) => {
        $crate::error::context::WithContext::with_context($result, || {
            $crate::error_context!($operation)
        })
    };
    ($result:expr, $operation:expr, $($key:expr => $value:expr),+) => {
        $crate::error::context::WithContext::with_context($result, || {
            $crate::error_context!($operation, $($key => $value),+)
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{TensorError, RusTorchError};
    
    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("matrix multiplication")
            .with_metadata("input_shape", "[2, 3]")
            .with_metadata("weight_shape", "[3, 4]")
            .with_location("tensor.rs", 42, 10);
            
        let formatted = context.format_context();
        assert!(formatted.contains("Operation: matrix multiplication"));
        assert!(formatted.contains("input_shape: [2, 3]"));
        assert!(formatted.contains("Location: tensor.rs:42:10"));
    }
    
    #[test]
    fn test_with_context_trait() {
        let tensor_error = TensorError::EmptyTensor;
        let result: Result<(), _> = Err(tensor_error);
        
        let enhanced = result.with_operation("test operation");
        assert!(enhanced.is_err());
        
        let error_message = enhanced.unwrap_err().to_string();
        assert!(error_message.contains("test operation"));
    }
    
    #[test]
    fn test_error_context_macro() {
        let context = error_context!("tensor add", "shape1" => "[2, 3]", "shape2" => "[2, 3]");
        assert_eq!(context.operation, "tensor add");
        assert!(context.metadata.contains_key("shape1"));
        assert!(context.metadata.contains_key("shape2"));
        assert!(context.location.is_some());
    }
    
    #[test]
    fn test_stack_trace() {
        let context = ErrorContext::new("outer operation")
            .push_operation("middle operation")
            .push_operation("inner operation");
            
        let formatted = context.format_context();
        assert!(formatted.contains("Stack trace:"));
        assert!(formatted.contains("0: inner operation"));
        assert!(formatted.contains("1: middle operation"));
    }
}