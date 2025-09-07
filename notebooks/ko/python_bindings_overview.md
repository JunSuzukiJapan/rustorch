# RusTorch Python 바인딩 개요

## 개요

RusTorch는 Rust로 구현된 고성능 딥러닝 프레임워크로, Rust의 안전성과 성능 이점을 활용하면서 PyTorch와 유사한 API를 제공합니다. Python 바인딩을 통해 Python에서 직접 RusTorch 기능에 액세스할 수 있습니다.

## 주요 특징

### 🚀 **고성능**
- **Rust 코어**: 메모리 안전성을 보장하면서 C++ 수준의 성능 달성
- **SIMD 지원**: 최적화된 수치 계산을 위한 자동 벡터화
- **병렬 처리**: rayon을 사용한 효율적인 병렬 계산
- **제로 카피**: NumPy와 RusTorch 간의 최소한의 데이터 복사

### 🛡️ **안전성**
- **메모리 안전성**: Rust의 소유권 시스템을 통해 메모리 누수 및 데이터 경쟁 방지
- **타입 안전성**: 컴파일 타임 타입 검사로 런타임 오류 감소
- **오류 처리**: Python 예외로의 자동 변환이 포함된 포괄적인 오류 처리

### 🎯 **사용 편의성**
- **PyTorch 호환 API**: 기존 PyTorch 코드에서 쉬운 마이그레이션
- **Keras 스타일 고수준 API**: model.fit()과 같은 직관적인 인터페이스
- **NumPy 통합**: NumPy 배열과의 양방향 변환

## 아키텍처

RusTorch의 Python 바인딩은 10개의 모듈로 구성됩니다:

### 1. **tensor** - 텐서 연산
```python
import rustorch

# 텐서 생성
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# NumPy 통합
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - 자동 미분
```python
# 그래디언트 계산
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # 그래디언트 얻기
```

### 3. **nn** - 신경망
```python
# 레이어 생성
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# 손실 함수
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - 옵티마이저
```python
# 옵티마이저
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 학습률 스케줄러
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - 데이터 로딩
```python
# 데이터셋 생성
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 데이터 변환
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - 고수준 훈련 API
```python
# Keras 스타일 API
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 훈련 실행
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - 분산 훈련
```python
# 분산 훈련 설정
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# 데이터 병렬
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - 시각화
```python
# 훈련 기록 플롯
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# 텐서 시각화
plotter.plot_tensor_as_image(tensor, title="특성 맵")
```

### 9. **utils** - 유틸리티
```python
# 모델 저장/로드
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# 프로파일링
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## 설치

### 전제 조건
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (GPU 사용시)

### 빌드 및 설치
```bash
# 저장소 클론
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install maturin numpy

# 빌드 및 설치
maturin develop --release

# 또는 PyPI에서 설치 (향후 계획)
# pip install rustorch
```

## 빠른 시작

### 1. 기본 텐서 연산
```python
import rustorch
import numpy as np

# 텐서 생성
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"형태: {x.shape()}")  # 형태: [2, 2]

# 수학 연산
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"결과: {z.to_numpy()}")
```

### 2. 선형 회귀 예제
```python
import rustorch
import numpy as np

# 데이터 생성
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# 텐서로 변환
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# 모델 정의
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# 데이터셋 생성
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# 훈련
history = model.fit(dataloader, epochs=100, verbose=True)

# 결과 표시
print(f"최종 손실: {history.train_loss()[-1]:.4f}")
```

### 3. 신경망 분류
```python
import rustorch

# 데이터 준비
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# 모델 구축
model = rustorch.Model("분류네트워크")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# 모델 컴파일
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 훈련 구성
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# 훈련
history = trainer.train(model, train_loader, val_loader)

# 평가
metrics = model.evaluate(test_loader)
print(f"테스트 정확도: {metrics['accuracy']:.4f}")
```

## 성능 최적화

### SIMD 활용
```python
# 빌드 중 SIMD 최적화 활성화
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # SIMD 최적화 계산
```

### GPU 사용
```python
# CUDA 사용 (향후 계획)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # GPU 계산
```

### 병렬 데이터 로딩
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # 병렬 워커 수
)
```

## 모범 사례

### 1. 메모리 효율성
```python
# 제로 카피 변환 활용
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # 복사 없음

# in-place 연산 사용
tensor.add_(1.0)  # 메모리 효율적
```

### 2. 오류 처리
```python
try:
    result = model(잘못된_입력)
except rustorch.RusTorchError as e:
    print(f"RusTorch 오류: {e}")
except Exception as e:
    print(f"예상치 못한 오류: {e}")
```

### 3. 디버깅 및 프로파일링
```python
# 프로파일러 사용
profiler = rustorch.utils.Profiler()
profiler.start()

# 계산 실행
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## 제한사항

### 현재 제한사항
- **GPU 지원**: CUDA/ROCm 지원 개발 중
- **동적 그래프**: 현재 정적 그래프만 지원
- **분산 훈련**: 기본 기능만 구현됨

### 향후 확장
- GPU 가속 (CUDA, Metal, ROCm)
- 동적 계산 그래프 지원
- 더 많은 신경망 레이어
- 모델 양자화 및 가지치기
- ONNX 내보내기 기능

## 기여

### 개발 참여
```bash
# 개발 환경 설정
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# 테스트 실행
cargo test
python -m pytest tests/

# 코드 품질 검사
cargo clippy
cargo fmt
```

### 커뮤니티
- GitHub Issues: 버그 리포트 및 기능 요청
- Discussions: 질문 및 토론
- Discord: 실시간 지원

## 라이센스

RusTorch는 MIT 라이센스 하에 출시됩니다. 상업적 및 비상업적 목적 모두에 자유롭게 사용 가능합니다.

## 관련 링크

- [GitHub 저장소](https://github.com/JunSuzukiJapan/RusTorch)
- [API 문서](./python_api_reference.md)
- [예제 및 튜토리얼](../examples/)
- [성능 벤치마크](./benchmarks.md)