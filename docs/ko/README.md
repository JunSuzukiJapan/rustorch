# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**PyTorch 유사 API, GPU 가속 및 엔터프라이즈급 성능을 갖춘 프로덕션 준비 Rust 딥러닝 라이브러리**

RusTorch는 Rust의 안전성과 성능을 활용하여 포괄적인 텐서 연산, 자동 미분, 신경망 레이어, 트랜스포머 아키텍처, 멀티백엔드 GPU 가속(CUDA/Metal/OpenCL), 고급 SIMD 최적화, 엔터프라이즈급 메모리 관리, 데이터 검증 및 품질 보증, 그리고 포괄적인 디버그 및 로깅 시스템을 제공하는 완전 기능적 딥러닝 라이브러리입니다.

## ✨ 특징

- 🔥 **포괄적 텐서 연산**: 수학 연산, 브로드캐스팅, 인덱싱 및 통계
- 🤖 **트랜스포머 아키텍처**: 멀티헤드 어텐션이 포함된 완전한 트랜스포머 구현
- 🧮 **행렬 분해**: PyTorch 호환성을 가진 SVD, QR, 고유값 분해
- 🧠 **자동 미분**: 그래디언트 계산을 위한 테이프 기반 계산 그래프
- 🚀 **동적 실행 엔진**: JIT 컴파일 및 런타임 최적화
- 🏗️ **신경망 레이어**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout 등
- ⚡ **크로스 플랫폼 최적화**: SIMD(AVX2/SSE/NEON), 플랫폼별 및 하드웨어 인식 최적화
- 🎮 **GPU 통합**: 자동 디바이스 선택이 포함된 CUDA/Metal/OpenCL 지원
- 🌐 **WebAssembly 지원**: 신경망 레이어, 컴퓨터 비전 및 실시간 추론이 포함된 완전한 브라우저 ML
- 🎮 **WebGPU 통합**: Chrome 최적화 GPU 가속 및 크로스 브라우저 호환성을 위한 CPU 폴백
- 📁 **모델 형식 지원**: Safetensors, ONNX 추론, PyTorch state dict 호환성
- ✅ **프로덕션 준비**: 968개 테스트 통과, 통합 오류 처리 시스템
- 📐 **향상된 수학 함수**: 완전한 수학 함수 세트(exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **고급 연산자 오버로드**: 스칼라 연산 및 인플레이스 할당이 포함된 텐서의 완전한 연산자 지원
- 📈 **고급 옵티마이저**: 학습률 스케줄러가 포함된 SGD, Adam, AdamW, RMSprop, AdaGrad
- 🔍 **데이터 검증 및 품질 보증**: 통계 분석, 이상 검출, 일관성 확인, 실시간 모니터링
- 🐛 **포괄적 디버그 및 로깅**: 구조화된 로깅, 성능 프로파일링, 메모리 추적, 자동화된 경고

## 🚀 빠른 시작

**📓 완전한 Jupyter 설정 가이드는 [README_JUPYTER.md](../../README_JUPYTER.md)를 참조하세요**

### Python Jupyter Lab 데모

#### 표준 CPU 데모
한 명령으로 Jupyter Lab과 함께 RusTorch 실행:

```bash
./start_jupyter.sh
```

#### WebGPU 가속 데모
브라우저 기반 GPU 가속을 위한 WebGPU 지원으로 RusTorch 실행:

```bash
./start_jupyter_webgpu.sh
```

두 스크립트 모두:
- 📦 가상 환경을 자동으로 생성
- 🔧 RusTorch Python 바인딩 빌드
- 🚀 데모 노트북과 함께 Jupyter Lab 시작
- 📍 실행 준비된 데모 노트북 열기

**WebGPU 특징:**
- 🌐 브라우저 기반 GPU 가속
- ⚡ 브라우저에서 고성능 행렬 연산
- 🔄 GPU 사용 불가 시 CPU로 자동 폴백
- 🎯 Chrome/Edge 최적화(권장 브라우저)

### 설치

`Cargo.toml`에 다음을 추가하세요:

```toml
[dependencies]
rustorch = "0.5.10"

# 선택적 기능
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # 선형대수 연산(SVD, QR, 고유값)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # 브라우저 ML용 WebAssembly 지원
webgpu = ["rustorch/webgpu"]            # Chrome 최적화 WebGPU 가속

# linalg 기능을 비활성화하려면(OpenBLAS/LAPACK 의존성 회피):
rustorch = { version = "0.5.10", default-features = false }
```

### 기본 사용법

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // 텐서 생성
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // 연산자 오버로드를 사용한 기본 연산
    let c = &a + &b;  // 요소별 덧셈
    let d = &a - &b;  // 요소별 뺄셈
    let e = &a * &b;  // 요소별 곱셈
    let f = &a / &b;  // 요소별 나눗셈
    
    // 스칼라 연산
    let g = &a + 10.0;  // 모든 요소에 스칼라 더하기
    let h = &a * 2.0;   // 스칼라로 곱하기
    
    // 수학 함수
    let exp_result = a.exp();   // 지수 함수
    let ln_result = a.ln();     // 자연 로그
    let sin_result = a.sin();   // 사인 함수
    let sqrt_result = a.sqrt(); // 제곱근
    
    // 행렬 연산
    let matmul_result = a.matmul(&b);  // 행렬 곱셈
    
    // 선형대수 연산(linalg 기능 필요)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // SVD 분해
        let qr_result = a.qr();         // QR 분해
        let eig_result = a.eigh();      // 고유값 분해
    }
    
    // 학습률 스케줄링이 포함된 고급 옵티마이저
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // 5 에포크에 걸쳐 0.1로 워밍업
    
    println!("모양: {:?}", c.shape());
    println!("결과: {:?}", c.as_slice());
}
```

### WebAssembly 사용법

브라우저 기반 ML 애플리케이션용:

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // 신경망 레이어
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // 향상된 수학 함수
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // 통계 분포
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // 훈련용 옵티마이저
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // 학습률, 모멘텀
    
    // 이미지 처리
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // 순전파
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('브라우저 ML 예측:', predictions);
}
```

## 📚 문서

- **[시작하기](../getting-started.md)** - 기본 사용법 및 예제
- **[특징](../features.md)** - 완전한 특징 목록 및 사양
- **[성능](../performance.md)** - 벤치마크 및 최적화 세부사항
- **[Jupyter WASM 가이드](jupyter-guide.md)** - Jupyter Notebook 단계별 설정

### WebAssembly 및 브라우저 ML
- **[WebAssembly 가이드](../WASM_GUIDE.md)** - 완전한 WASM 통합 및 API 참조
- **[WebGPU 통합](../WEBGPU_INTEGRATION.md)** - Chrome 최적화 GPU 가속

### 프로덕션 및 운영
- **[GPU 가속 가이드](../GPU_ACCELERATION_GUIDE.md)** - GPU 설정 및 사용
- **[프로덕션 가이드](../PRODUCTION_GUIDE.md)** - 배포 및 확장

## 📊 성능

**최신 벤치마크 결과:**

| 연산 | 성능 | 세부사항 |
|-----------|-------------|---------|
| **SVD 분해** | ~1ms (8x8 행렬) | ✅ LAPACK 기반 |
| **QR 분해** | ~24μs (8x8 행렬) | ✅ 빠른 분해 |
| **고유값** | ~165μs (8x8 행렬) | ✅ 대칭 행렬 |
| **복소수 FFT** | 10-312μs (8-64 샘플) | ✅ Cooley-Tukey 최적화 |
| **신경망** | 1-7s 훈련 | ✅ Boston housing 데모 |
| **활성화 함수** | <1μs | ✅ ReLU, Sigmoid, Tanh |

## 🧪 테스트

**968개 테스트 통과** - 통합 오류 처리 시스템을 갖춘 프로덕션 준비 품질 보증.

```bash
# 모든 테스트 실행
cargo test --no-default-features

# 선형대수 기능과 함께 테스트 실행
cargo test --features linalg
```

## 🤝 기여

기여를 환영합니다! 특히 도움이 필요한 영역:

- **🎯 특수 함수 정밀도**: 수치 정확도 개선
- **⚡ 성능 최적화**: SIMD 개선, GPU 최적화
- **🧪 테스트**: 더 포괄적인 테스트 케이스
- **📚 문서**: 예제, 튜토리얼, 개선
- **🌐 플랫폼 지원**: WebAssembly, 모바일 플랫폼

## 라이선스

다음 중 하나로 라이선스됩니다:

 * Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) 또는 http://www.apache.org/licenses/LICENSE-2.0)
 * MIT 라이선스 ([LICENSE-MIT](../../LICENSE-MIT) 또는 http://opensource.org/licenses/MIT)

원하는 것을 선택하세요.