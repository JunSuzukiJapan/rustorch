# RusTorch 빠른 시작 가이드

## 설치

### 1. 전제 조건
```bash
# Rust 1.70 이상
rustc --version

# Python 3.8 이상
python --version

# 필수 의존성 설치
pip install maturin numpy matplotlib
```

### 2. RusTorch 빌드 및 설치
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 개발 모드로 빌드 및 설치
maturin develop --release
```

## 기본 사용 예제

### 1. 텐서 생성과 기본 연산

```python
import rustorch
import numpy as np

# 텐서 생성
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"텐서 x:\n{x}")
print(f"형태: {x.shape()}")  # [2, 2]

# 영 행렬과 단위 행렬
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"영 행렬:\n{zeros}")
print(f"일 행렬:\n{ones}")
print(f"단위 행렬:\n{identity}")

# 무작위 텐서
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"정규분포 난수:\n{random_normal}")
print(f"균등분포 난수:\n{random_uniform}")

# NumPy 통합
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"NumPy에서 변환:\n{tensor_from_numpy}")

# NumPy로 다시 변환
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"NumPy로 다시 변환:\n{back_to_numpy}")
```

### 2. 산술 연산

```python
# 기본 산술 연산
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 원소별 연산
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (원소별)
div_result = a.div(b)  # a / b (원소별)

print(f"덧셈:\n{add_result}")
print(f"뺄셈:\n{sub_result}")
print(f"곱셈:\n{mul_result}")
print(f"나눗셈:\n{div_result}")

# 스칼라 연산
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"스칼라 덧셈 (+2):\n{scalar_add}")
print(f"스칼라 곱셈 (*3):\n{scalar_mul}")

# 행렬 곱셈
matmul_result = a.matmul(b)
print(f"행렬 곱셈:\n{matmul_result}")

# 수학 함수들
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"제곱근:\n{sqrt_result}")
print(f"지수함수:\n{exp_result}")
print(f"자연로그:\n{log_result}")
```

### 3. 텐서 형태 조작

```python
# 형태 조작 예제
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"원본 형태: {original.shape()}")  # [2, 4]

# 재형성
reshaped = original.reshape([4, 2])
print(f"재형성 [4, 2]:\n{reshaped}")

# 전치
transposed = original.transpose(0, 1)
print(f"전치:\n{transposed}")

# 차원 추가/제거
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"squeeze 전: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"squeeze 후: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"unsqueeze 후: {expanded.shape()}")  # [1, 3]
```

### 4. 통계 연산

```python
# 통계 함수들
data = rustorch.randn([3, 4])
print(f"데이터:\n{data}")

# 기본 통계량
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"평균: {mean_val.item():.4f}")
print(f"합계: {sum_val.item():.4f}")
print(f"표준편차: {std_val.item():.4f}")
print(f"분산: {var_val.item():.4f}")
print(f"최댓값: {max_val.item():.4f}")
print(f"최솟값: {min_val.item():.4f}")

# 특정 차원에 따른 통계량
row_mean = data.mean(dim=1)  # 각 행의 평균
col_sum = data.sum(dim=0)    # 각 열의 합계

print(f"행별 평균: {row_mean}")
print(f"열별 합계: {col_sum}")
```

## 자동 미분 기초

### 1. 기울기 계산

```python
# 자동 미분 예제
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"입력 텐서: {x}")

# Variable 생성
var_x = rustorch.autograd.Variable(x)

# 계산 그래프 구축
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"출력: {y.data().item()}")

# 역전파
y.backward()

# 기울기 획득
grad = var_x.grad()
print(f"기울기: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. 복잡한 계산 그래프

```python
# 더 복잡한 예제
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# 복잡한 함수: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), 그다음 합계

print(f"결과: {z.data().item():.4f}")

# 역전파
z.backward()
grad = var_x.grad()
print(f"기울기: {grad}")
```

## 신경망 기초

### 1. 단순 선형 계층

```python
# 선형 계층 생성
linear_layer = rustorch.nn.Linear(3, 1)  # 3개 입력 -> 1개 출력

# 무작위 입력
input_data = rustorch.randn([2, 3])  # 배치 크기 2, 3개 특성
print(f"입력: {input_data}")

# 순전파
output = linear_layer.forward(input_data)
print(f"출력: {output}")

# 매개변수 확인
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"가중치 형태: {weight.shape()}")
print(f"가중치: {weight}")
if bias is not None:
    print(f"편향: {bias}")
```

### 2. 활성화 함수

```python
# 다양한 활성화 함수들
x = rustorch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# ReLU
relu = rustorch.nn.ReLU()
relu_output = relu.forward(x)
print(f"ReLU: {relu_output}")

# Sigmoid
sigmoid = rustorch.nn.Sigmoid()
sigmoid_output = sigmoid.forward(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh
tanh = rustorch.nn.Tanh()
tanh_output = tanh.forward(x)
print(f"Tanh: {tanh_output}")
```

### 3. 손실 함수

```python
# 손실 함수 사용 예제
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# 평균 제곱 오차
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE 손실: {loss_value.item():.6f}")

# 교차 엔트로피 (분류용)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # 클래스 인덱스

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"교차 엔트로피 손실: {ce_loss_value.item():.6f}")
```

## 데이터 처리

### 1. 데이터셋과 데이터로더

```python
# 데이터셋 생성
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100개 샘플, 4개 특성
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3클래스 분류

# 텐서로 변환
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# 데이터셋 생성
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"데이터셋 크기: {len(dataset)}")

# 데이터로더 생성
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# 데이터로더에서 배치 가져오기
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # 처음 3개 배치만 표시
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"배치 {batch_idx}: 입력 형태 {inputs.shape()}, 타겟 형태 {targets.shape()}")
```

### 2. 데이터 변환

```python
# 데이터 변환 예제
data = rustorch.randn([10, 10])
print(f"원본 데이터 평균: {data.mean().item():.4f}")
print(f"원본 데이터 표준편차: {data.std().item():.4f}")

# 정규화 변환
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"정규화된 데이터 평균: {normalized_data.mean().item():.4f}")
print(f"정규화된 데이터 표준편차: {normalized_data.std().item():.4f}")
```

## 완전한 훈련 예제

### 선형 회귀

```python
# 완전한 선형 회귀 예제
import numpy as np

# 데이터 생성
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# 텐서로 변환
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# 데이터셋과 데이터로더 생성
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# 모델 정의
model = rustorch.nn.Linear(1, 1)  # 1개 입력 -> 1개 출력

# 손실 함수와 옵티마이저
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# 훈련 루프
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    dataloader.reset()
    while True:
        batch = dataloader.next_batch()
        if batch is None:
            break
        
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            
            # 기울기 초기화
            optimizer.zero_grad()
            
            # 순전파
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # 역전파 (단순화됨)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"에포크 {epoch}: 손실 = {avg_loss:.6f}")

print("훈련 완료!")

# 최종 매개변수
final_weight = model.weight()
final_bias = model.bias()
print(f"학습된 가중치: {final_weight.item():.4f} (실제값: 2.0)")
if final_bias is not None:
    print(f"학습된 편향: {final_bias.item():.4f} (실제값: 1.0)")
```

## 문제 해결

### 일반적인 문제와 해결책

1. **설치 문제**
```bash
# maturin을 찾을 수 없는 경우
pip install --upgrade maturin

# Rust가 오래된 경우
rustup update

# Python 환경 문제
python -m pip install --upgrade pip
```

2. **런타임 오류**
```python
# 텐서 형태 확인
print(f"텐서 형태: {tensor.shape()}")
print(f"텐서 데이터 타입: {tensor.dtype()}")

# NumPy 변환 시 데이터 타입 주의
np_array = np.array(data, dtype=np.float32)  # 명시적 float32
```

3. **성능 최적화**
```python
# 릴리스 모드로 빌드
# maturin develop --release

# 배치 크기 조정
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # 더 큰 배치
```

## 다음 단계

1. **고급 예제 시도**: `docs/examples/neural_networks/`의 예제들을 확인하세요
2. **Keras 스타일 API 사용**: 더 쉬운 모델 구축을 위한 `rustorch.training.Model`
3. **시각화 기능**: 훈련 진행 상황 시각화를 위한 `rustorch.visualization`
4. **분산 훈련**: 병렬 처리를 위한 `rustorch.distributed`

자세한 문서:
- [Python API 참조](../en/python_api_reference.md)
- [개요 문서](../en/python_bindings_overview.md)
- [예제 모음](../examples/)