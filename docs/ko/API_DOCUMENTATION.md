# RusTorch API 문서

## 📚 완전한 API 참조

이 문서는 RusTorch v0.5.15의 포괄적인 API 문서를 모듈과 기능별로 정리하여 제공합니다. 모든 1060+ 테스트에 걸쳐 일관된 오류 관리를 위해 `RusTorchError`와 `RusTorchResult<T>`를 사용한 통합 오류 처리를 포함합니다. **8단계 완료**로 조건 연산, 인덱싱, 통계 함수를 포함한 고급 텐서 유틸리티가 추가되었습니다. **9단계 완료**로 모델 저장/로드, JIT 컴파일, PyTorch 호환성을 포함한 다중 형식 지원의 포괄적인 직렬화 시스템이 도입되었습니다.

## 🏗️ 핵심 아키텍처

### 모듈 구조

```
rustorch/
├── tensor/              # 핵심 텐서 연산과 데이터 구조
├── nn/                  # 신경망 레이어와 함수
├── autograd/            # 자동 미분 엔진
├── optim/               # 옵티마이저와 학습률 스케줄러
├── special/             # 특수 수학 함수
├── distributions/       # 통계 분포
├── vision/              # 컴퓨터 비전 변환
├── linalg/              # 선형 대수 연산 (BLAS/LAPACK)
├── gpu/                 # GPU 가속 (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # 희소 텐서 연산과 가지치기 (12단계)
├── serialization/       # 모델 직렬화와 JIT 컴파일 (9단계)
└── wasm/                # WebAssembly 바인딩 ([WASM API 문서](WASM_API_DOCUMENTATION.md) 참조)
```

## 📊 텐서 모듈

### 기본 텐서 생성

```rust
use rustorch::tensor::Tensor;

// 기본 생성
let tensor = Tensor::new(vec![2, 3]);               // 형태 기반 생성
let tensor = Tensor::from_vec(data, vec![2, 3]);    // 데이터 벡터로부터 생성
let tensor = Tensor::zeros(vec![10, 10]);           // 영으로 채운 텐서
let tensor = Tensor::ones(vec![5, 5]);              // 일로 채운 텐서
let tensor = Tensor::randn(vec![3, 3]);             // 무작위 정규 분포
let tensor = Tensor::rand(vec![3, 3]);              // 무작위 균등 분포 [0,1)
let tensor = Tensor::eye(5);                        // 단위 행렬
let tensor = Tensor::full(vec![2, 2], 3.14);       // 특정 값으로 채우기
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // 범위 텐서
let tensor = Tensor::linspace(0.0, 1.0, 100);      // 선형 간격
```

### 텐서 연산

```rust
// 산술 연산
let result = a.add(&b);                             // 원소별 덧셈
let result = a.sub(&b);                             // 원소별 뺄셈
let result = a.mul(&b);                             // 원소별 곱셈
let result = a.div(&b);                             // 원소별 나눗셈
let result = a.pow(&b);                             // 원소별 거듭제곱
let result = a.rem(&b);                             // 원소별 나머지

// 행렬 연산
let result = a.matmul(&b);                          // 행렬 곱셈
let result = a.transpose();                         // 행렬 전치
let result = a.dot(&b);                             // 내적

// 수학 함수
let result = tensor.exp();                          // 지수
let result = tensor.ln();                           // 자연 로그
let result = tensor.log10();                        // 상용 로그
let result = tensor.sqrt();                         // 제곱근
let result = tensor.abs();                          // 절댓값
let result = tensor.sin();                          // 사인 함수
let result = tensor.cos();                          // 코사인 함수
let result = tensor.tan();                          // 탄젠트 함수
let result = tensor.asin();                         // 아크사인
let result = tensor.acos();                         // 아크코사인
let result = tensor.atan();                         // 아크탄젠트
let result = tensor.sinh();                         // 쌍곡 사인
let result = tensor.cosh();                         // 쌍곡 코사인
let result = tensor.tanh();                         // 쌍곡 탄젠트
let result = tensor.floor();                        // 바닥 함수
let result = tensor.ceil();                         // 천장 함수
let result = tensor.round();                        // 반올림 함수
let result = tensor.sign();                         // 부호 함수
let result = tensor.max();                          // 최댓값
let result = tensor.min();                          // 최솟값
let result = tensor.sum();                          // 모든 원소 합
let result = tensor.mean();                         // 평균값
let result = tensor.std();                          // 표준편차
let result = tensor.var();                          // 분산

// 형태 조작
let result = tensor.reshape(vec![6, 4]);            // 텐서 재형성
let result = tensor.squeeze();                      // 크기-1 차원 제거
let result = tensor.unsqueeze(1);                   // 인덱스에 차원 추가
let result = tensor.permute(vec![1, 0, 2]);         // 차원 순열
let result = tensor.expand(vec![10, 10, 5]);        // 텐서 차원 확장
```

## 🧠 신경망(nn) 모듈

### 기본 레이어

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// 선형 레이어
let linear = Linear::new(784, 256)?;                // 입력 784, 출력 256
let output = linear.forward(&input)?;

// 합성곱 레이어
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// 배치 정규화
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// 드롭아웃
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### 활성화 함수

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// 기본 활성화 함수
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// 매개변수화된 활성화 함수
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// 사용 예제
let activated = relu.forward(&input)?;
```

## 🚀 GPU 가속 모듈

### 디바이스 관리

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// 사용 가능한 디바이스 확인
let device_count = get_device_count()?;
let device = Device::best_available()?;            // 최적 디바이스 선택

// 디바이스 설정
set_device(&device)?;

// 텐서를 GPU로 이동
let gpu_tensor = tensor.to_device(&device)?;
```

### CUDA 연산

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// CUDA 디바이스 연산
let cuda_device = CudaDevice::new(0)?;              // GPU 0 사용
let stats = memory_stats(0)?;                      // 메모리 통계
println!("사용된 메모리: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 최적화기(Optim) 모듈

### 기본 최적화기

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Adam 최적화기
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// SGD 최적화기
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// 최적화 단계
optimizer.zero_grad()?;
// ... 순전파와 역전파 ...
optimizer.step()?;
```

## 📖 사용 예제

### 선형 회귀

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// 데이터 준비
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// 모델 정의
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// 훈련 루프
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("에포크 {}: 손실 = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ 알려진 제한사항

1. **GPU 메모리 제한**: 큰 텐서(>8GB)의 경우 명시적 메모리 관리 필요
2. **WebAssembly 제한**: 일부 BLAS 연산이 WASM 환경에서 사용 불가
3. **분산 학습**: NCCL 백엔드는 Linux에서만 지원
4. **Metal 제한**: 일부 고급 연산은 CUDA 백엔드에서만 사용 가능

## 🔗 관련 링크

- [메인 README](../README.md)
- [WASM API 문서](WASM_API_DOCUMENTATION.md)
- [Jupyter 가이드](jupyter-guide.md)
- [GitHub 저장소](https://github.com/JunSuzukiJapan/RusTorch)
- [Crates.io 패키지](https://crates.io/crates/rustorch)

---

**최근 업데이트**: v0.5.15 | **라이선스**: MIT | **작성자**: Jun Suzuki