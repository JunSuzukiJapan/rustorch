/// Data type enumeration for tensors
/// テンソル用データ型列挙
use std::fmt;

/// Data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer  
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit signed integer
    Int64,
    /// 64-bit unsigned integer
    UInt64,
    /// 16-bit floating point (half precision)
    Float16,
    /// 16-bit brain floating point (bfloat16)
    BFloat16,
    /// 32-bit floating point (single precision)
    Float32,
    /// 64-bit floating point (double precision)
    Float64,
    /// Boolean
    Bool,
    /// Complex 64-bit (32-bit real + 32-bit imaginary)
    Complex64,
    /// Complex 128-bit (64-bit real + 64-bit imaginary)
    Complex128,
}

impl DType {
    /// Get the size in bytes of this data type
    pub fn size(&self) -> usize {
        match self {
            DType::Int8 | DType::UInt8 | DType::Bool => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 | DType::BFloat16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::Float16 | DType::BFloat16 | DType::Float32 | DType::Float64
        )
    }

    /// Check if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::UInt8
                | DType::Int16
                | DType::UInt16
                | DType::Int32
                | DType::UInt32
                | DType::Int64
                | DType::UInt64
        )
    }

    /// Check if this is a signed integer type
    pub fn is_signed_int(&self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
        )
    }

    /// Check if this is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(self, DType::Complex64 | DType::Complex128)
    }

    /// Check if this is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Get the corresponding floating point type for this dtype
    pub fn to_float(&self) -> DType {
        match self {
            DType::Int8 | DType::UInt8 | DType::Bool => DType::Float32,
            DType::Int16 | DType::UInt16 | DType::Float16 | DType::BFloat16 => DType::Float32,
            DType::Int32 | DType::UInt32 | DType::Float32 => DType::Float32,
            DType::Int64 | DType::UInt64 | DType::Float64 => DType::Float64,
            DType::Complex64 => DType::Complex64,
            DType::Complex128 => DType::Complex128,
        }
    }

    /// Get the corresponding integer type for this dtype
    pub fn to_int(&self) -> DType {
        match self {
            DType::Int8 | DType::UInt8 | DType::Bool => DType::Int32,
            DType::Int16 | DType::UInt16 | DType::Float16 | DType::BFloat16 => DType::Int32,
            DType::Int32 | DType::UInt32 | DType::Float32 => DType::Int32,
            DType::Int64 | DType::UInt64 | DType::Float64 => DType::Int64,
            DType::Complex64 | DType::Complex128 => DType::Int64,
        }
    }

    /// Get the minimum value for this data type (if applicable)
    pub fn min_value(&self) -> Option<f64> {
        match self {
            DType::Int8 => Some(i8::MIN as f64),
            DType::UInt8 => Some(0.0),
            DType::Int16 => Some(i16::MIN as f64),
            DType::UInt16 => Some(0.0),
            DType::Int32 => Some(i32::MIN as f64),
            DType::UInt32 => Some(0.0),
            DType::Int64 => Some(i64::MIN as f64),
            DType::UInt64 => Some(0.0),
            DType::Float16 => Some(-65504.0), // Approximate half precision min
            DType::BFloat16 => Some(-3.38e38), // Approximate bfloat16 min
            DType::Float32 => Some(f32::MIN as f64),
            DType::Float64 => Some(f64::MIN),
            DType::Bool => Some(0.0),
            DType::Complex64 | DType::Complex128 => None,
        }
    }

    /// Get the maximum value for this data type (if applicable)
    pub fn max_value(&self) -> Option<f64> {
        match self {
            DType::Int8 => Some(i8::MAX as f64),
            DType::UInt8 => Some(u8::MAX as f64),
            DType::Int16 => Some(i16::MAX as f64),
            DType::UInt16 => Some(u16::MAX as f64),
            DType::Int32 => Some(i32::MAX as f64),
            DType::UInt32 => Some(u32::MAX as f64),
            DType::Int64 => Some(i64::MAX as f64),
            DType::UInt64 => Some(u64::MAX as f64),
            DType::Float16 => Some(65504.0), // Approximate half precision max
            DType::BFloat16 => Some(3.38e38), // Approximate bfloat16 max
            DType::Float32 => Some(f32::MAX as f64),
            DType::Float64 => Some(f64::MAX),
            DType::Bool => Some(1.0),
            DType::Complex64 | DType::Complex128 => None,
        }
    }

    /// Check if two dtypes are compatible for operations
    pub fn is_compatible_with(&self, other: &DType) -> bool {
        // Same types are always compatible
        if self == other {
            return true;
        }

        // Check compatibility groups
        match (self, other) {
            // All numeric types are compatible with each other
            (a, b) if a.is_int() && b.is_int() => true,
            (a, b) if a.is_float() && b.is_float() => true,
            (a, b) if a.is_int() && b.is_float() => true,
            (a, b) if a.is_float() && b.is_int() => true,

            // Bool is compatible with numeric types
            (DType::Bool, b) if b.is_int() || b.is_float() => true,
            (a, DType::Bool) if a.is_int() || a.is_float() => true,

            // Complex types are only compatible with each other
            (a, b) if a.is_complex() && b.is_complex() => true,

            _ => false,
        }
    }

    /// Get the common dtype for two dtypes (promotion)
    pub fn common_dtype(&self, other: &DType) -> DType {
        if self == other {
            return *self;
        }

        // Promotion rules
        match (self, other) {
            // Complex types take priority
            (DType::Complex128, _) | (_, DType::Complex128) => DType::Complex128,
            (DType::Complex64, _) | (_, DType::Complex64) => DType::Complex64,

            // Float64 takes priority over other floats
            (DType::Float64, _) | (_, DType::Float64) => DType::Float64,
            (DType::Float32, _) | (_, DType::Float32) => DType::Float32,
            (DType::BFloat16, _) | (_, DType::BFloat16) => DType::BFloat16,
            (DType::Float16, _) | (_, DType::Float16) => DType::Float16,

            // Integer promotion
            (DType::Int64, _) | (_, DType::Int64) => DType::Int64,
            (DType::UInt64, _) | (_, DType::UInt64) => DType::UInt64,
            (DType::Int32, _) | (_, DType::Int32) => DType::Int32,
            (DType::UInt32, _) | (_, DType::UInt32) => DType::UInt32,
            (DType::Int16, _) | (_, DType::Int16) => DType::Int16,
            (DType::UInt16, _) | (_, DType::UInt16) => DType::UInt16,
            (DType::Int8, _) | (_, DType::Int8) => DType::Int8,
            (DType::UInt8, _) | (_, DType::UInt8) => DType::UInt8,

            // Any remaining combinations default to the first type
            (first, _) => *first,
        }
    }

    /// Parse dtype from string
    pub fn from_str(s: &str) -> Result<DType, String> {
        match s.to_lowercase().as_str() {
            "int8" | "i8" => Ok(DType::Int8),
            "uint8" | "u8" => Ok(DType::UInt8),
            "int16" | "i16" => Ok(DType::Int16),
            "uint16" | "u16" => Ok(DType::UInt16),
            "int32" | "i32" | "int" => Ok(DType::Int32),
            "uint32" | "u32" | "uint" => Ok(DType::UInt32),
            "int64" | "i64" | "long" => Ok(DType::Int64),
            "uint64" | "u64" | "ulong" => Ok(DType::UInt64),
            "float16" | "f16" | "half" => Ok(DType::Float16),
            "bfloat16" | "bf16" => Ok(DType::BFloat16),
            "float32" | "f32" | "float" => Ok(DType::Float32),
            "float64" | "f64" | "double" => Ok(DType::Float64),
            "bool" | "boolean" => Ok(DType::Bool),
            "complex64" | "c64" => Ok(DType::Complex64),
            "complex128" | "c128" => Ok(DType::Complex128),
            _ => Err(format!("Unknown dtype: {}", s)),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DType::Int8 => "int8",
            DType::UInt8 => "uint8",
            DType::Int16 => "int16",
            DType::UInt16 => "uint16",
            DType::Int32 => "int32",
            DType::UInt32 => "uint32",
            DType::Int64 => "int64",
            DType::UInt64 => "uint64",
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Bool => "bool",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        };
        write!(f, "{}", name)
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::Float32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Int8.size(), 1);
        assert_eq!(DType::UInt8.size(), 1);
        assert_eq!(DType::Bool.size(), 1);
        assert_eq!(DType::Int16.size(), 2);
        assert_eq!(DType::Float16.size(), 2);
        assert_eq!(DType::Int32.size(), 4);
        assert_eq!(DType::Float32.size(), 4);
        assert_eq!(DType::Int64.size(), 8);
        assert_eq!(DType::Float64.size(), 8);
        assert_eq!(DType::Complex64.size(), 8);
        assert_eq!(DType::Complex128.size(), 16);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
        assert!(!DType::Int32.is_float());

        assert!(DType::Int32.is_int());
        assert!(DType::UInt32.is_int());
        assert!(!DType::Float32.is_int());

        assert!(DType::Int32.is_signed_int());
        assert!(!DType::UInt32.is_signed_int());

        assert!(DType::UInt32.is_unsigned_int());
        assert!(!DType::Int32.is_unsigned_int());

        assert!(DType::Complex64.is_complex());
        assert!(!DType::Float32.is_complex());

        assert!(DType::Bool.is_bool());
        assert!(!DType::Int32.is_bool());
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(DType::Int32.to_float(), DType::Float32);
        assert_eq!(DType::Int64.to_float(), DType::Float64);
        assert_eq!(DType::Float32.to_float(), DType::Float32);

        assert_eq!(DType::Float32.to_int(), DType::Int32);
        assert_eq!(DType::Float64.to_int(), DType::Int64);
        assert_eq!(DType::Int32.to_int(), DType::Int32);
    }

    #[test]
    fn test_dtype_min_max() {
        assert_eq!(DType::Int8.min_value(), Some(i8::MIN as f64));
        assert_eq!(DType::Int8.max_value(), Some(i8::MAX as f64));
        assert_eq!(DType::UInt8.min_value(), Some(0.0));
        assert_eq!(DType::UInt8.max_value(), Some(u8::MAX as f64));
        assert_eq!(DType::Bool.min_value(), Some(0.0));
        assert_eq!(DType::Bool.max_value(), Some(1.0));
        assert_eq!(DType::Complex64.min_value(), None);
        assert_eq!(DType::Complex64.max_value(), None);
    }

    #[test]
    fn test_dtype_compatibility() {
        assert!(DType::Int32.is_compatible_with(&DType::Int32));
        assert!(DType::Int32.is_compatible_with(&DType::Float32));
        assert!(DType::Float32.is_compatible_with(&DType::Int32));
        assert!(DType::Bool.is_compatible_with(&DType::Int32));
        assert!(DType::Complex64.is_compatible_with(&DType::Complex128));
        assert!(!DType::Complex64.is_compatible_with(&DType::Float32));
    }

    #[test]
    fn test_dtype_promotion() {
        assert_eq!(DType::Int32.common_dtype(&DType::Int32), DType::Int32);
        assert_eq!(DType::Int32.common_dtype(&DType::Float32), DType::Float32);
        assert_eq!(DType::Float32.common_dtype(&DType::Float64), DType::Float64);
        assert_eq!(
            DType::Int32.common_dtype(&DType::Complex64),
            DType::Complex64
        );
        assert_eq!(DType::Bool.common_dtype(&DType::Int32), DType::Int32);
    }

    #[test]
    fn test_dtype_from_string() {
        assert_eq!(DType::from_str("int32").unwrap(), DType::Int32);
        assert_eq!(DType::from_str("float32").unwrap(), DType::Float32);
        assert_eq!(DType::from_str("bool").unwrap(), DType::Bool);
        assert_eq!(DType::from_str("f32").unwrap(), DType::Float32);
        assert_eq!(DType::from_str("i32").unwrap(), DType::Int32);
        assert!(DType::from_str("invalid").is_err());
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::Int32.to_string(), "int32");
        assert_eq!(DType::Float32.to_string(), "float32");
        assert_eq!(DType::Bool.to_string(), "bool");
        assert_eq!(DType::Complex64.to_string(), "complex64");
    }

    #[test]
    fn test_dtype_default() {
        assert_eq!(DType::default(), DType::Float32);
    }
}
