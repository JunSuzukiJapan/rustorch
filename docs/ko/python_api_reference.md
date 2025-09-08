# RusTorch Python API 참조서

머신러닝 및 딥러닝 개발자를 위한 RusTorch Python API의 완전한 참조서입니다.

## 목차

- [Tensor 모듈](#tensor-모듈)
- [자동 미분](#자동-미분)
- [신경망](#신경망)
- [최적화](#최적화)
- [컴퓨터 비전](#컴퓨터-비전)
- [GPU 및 디바이스](#gpu-및-디바이스)
- [유틸리티](#유틸리티)

## Tensor 모듈

### `Tensor`

N차원 텐서 연산을 위한 기본 구조입니다.

#### 생성자

```python
import rustorch_py as torch

# 데이터로부터 텐서 생성
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])

# 영 텐서
zeros = torch.zeros([2, 3], dtype=torch.float32)

# 일 텐서
ones = torch.ones([2, 3], dtype=torch.float32)

# 랜덤 텐서 (정규분포)
randn = torch.randn([2, 3], dtype=torch.float32)

# 랜덤 텐서 (균등분포)
rand = torch.rand([2, 3], dtype=torch.float32)
```

#### 기본 연산

```python
# 산술 연산
result = tensor1.add(tensor2)
result = tensor1.sub(tensor2)
result = tensor1.mul(tensor2)
result = tensor1.div(tensor2)

# 행렬 곱셈
result = tensor1.matmul(tensor2)

# 전치
transposed = tensor.t()

# 모양 변경
reshaped = tensor.reshape([6, 1])
```

#### 축소 연산

```python
# 합계
sum_all = tensor.sum()
sum_dim = tensor.sum(dim=0, keepdim=False)

# 평균
mean_all = tensor.mean()
mean_dim = tensor.mean(dim=0, keepdim=False)

# 최대값과 최소값
max_val, max_indices = tensor.max(dim=0)
min_val, min_indices = tensor.min(dim=0)
```

#### 인덱싱 및 선택

```python
# 인덱스로 선택
slice_result = tensor.slice(dim=0, start=0, end=2, step=1)

# 조건으로 선택
mask = tensor.gt(threshold)
selected = tensor.masked_select(mask)
```

## 자동 미분

### `Variable`

자동 미분을 가능하게 하는 텐서 래퍼입니다.

```python
import rustorch_py.autograd as autograd

# requires_grad=True로 변수 생성
x = autograd.Variable(torch.randn([2, 2]), requires_grad=True)
y = autograd.Variable(torch.randn([2, 2]), requires_grad=True)

# 계산 그래프를 구축하는 연산
z = x.matmul(y)
loss = z.sum()

# 역전파
loss.backward()

# 그래디언트 접근
x_grad = x.grad
print(f"x의 그래디언트: {x_grad}")
```

### 미분 함수

```python
# 그래디언트를 가진 사용자 정의 함수
def custom_function(input_var):
    # 순전파
    output = input_var.pow(2.0)
    
    # 그래디언트는 자동으로 계산됩니다
    return output

# 그래디언트 계산 없이 실행
with torch.no_grad():
    result = model.forward(input_data)
```

## 신경망

### 기본 레이어

#### `Linear`

선형 변환 (완전 연결 레이어)입니다.

```python
import rustorch_py.nn as nn

linear = nn.Linear(784, 256)  # 입력: 784, 출력: 256
input_tensor = torch.randn([32, 784])
output = linear.forward(input_tensor)
```

#### 활성화 함수

```python
# ReLU
relu = nn.ReLU()
output = relu.forward(input_tensor)

# Sigmoid
sigmoid = nn.Sigmoid()
output = sigmoid.forward(input_tensor)

# Tanh
tanh = nn.Tanh()
output = tanh.forward(input_tensor)

# GELU
gelu = nn.GELU()
output = gelu.forward(input_tensor)
```

### 합성곱 레이어

```python
# 2D 합성곱
conv2d = nn.Conv2d(
    in_channels=3,     # 입력 채널
    out_channels=64,   # 출력 채널
    kernel_size=3,     # 커널 크기
    stride=1,          # 스트라이드
    padding=1          # 패딩
)

input_tensor = torch.randn([1, 3, 224, 224])
output = conv2d.forward(input_tensor)
```

### 순차 모델

```python
model = nn.Sequential([
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
])

# 순전파
input_data = torch.randn([32, 784])
output = model.forward(input_data)
```

## 최적화

### 옵티마이저

#### `Adam`

```python
import rustorch_py.optim as optim

optimizer = optim.Adam(
    params=model.parameters(),  # 모델 매개변수
    lr=0.001,                   # 학습률
    betas=(0.9, 0.999),        # 베타 계수
    eps=1e-8                   # 엡실론
)

# 훈련 루프
for batch in data_loader:
    prediction = model.forward(batch.input)
    loss = criterion(prediction, batch.target)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### `SGD`

```python
optimizer = optim.SGD(
    params=model.parameters(),
    lr=0.01,                   # 학습률
    momentum=0.9               # 모멘텀
)
```

### 손실 함수

```python
import rustorch_py.nn.functional as F

# 평균 제곱 오차
mse_loss = F.mse_loss(prediction, target)

# 교차 엔트로피
ce_loss = F.cross_entropy(prediction, target)

# 이진 교차 엔트로피
bce_loss = F.binary_cross_entropy(prediction, target)
```

## 컴퓨터 비전

### 이미지 변환

```python
import rustorch_py.vision.transforms as transforms

# 이미지 크기 조정
resize = transforms.Resize((224, 224))
resized = resize.forward(image)

# 정규화
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # 평균
    std=[0.229, 0.224, 0.225]    # 표준편차
)
normalized = normalize.forward(image)

# 랜덤 변환
random_crop = transforms.RandomCrop(32, padding=4)
cropped = random_crop.forward(image)
```

### 사전 훈련된 모델

```python
import rustorch_py.vision.models as models

# ResNet
resnet18 = models.resnet18(pretrained=True)
output = resnet18.forward(input_tensor)

# VGG
vgg16 = models.vgg16(pretrained=True)
features = vgg16.features(input_tensor)
```

## GPU 및 디바이스

### 디바이스 관리

```python
import rustorch_py as torch

# CPU
cpu = torch.device('cpu')

# CUDA
cuda = torch.device('cuda:0')  # GPU 0
cuda_available = torch.cuda.is_available()

# Metal (macOS)
metal = torch.device('metal:0')

# 텐서를 디바이스로 이동
tensor_gpu = tensor.to(cuda)
```

### 다중 GPU 연산

```python
import rustorch_py.distributed as dist

# 분산 처리 초기화
dist.init_process_group("nccl", rank=0, world_size=2)

# 그래디언트 동기화를 위한 AllReduce
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## 유틸리티

### 직렬화

```python
import rustorch_py.serialize as serialize

# 모델 저장
serialize.save(model, "model.pth")

# 모델 로드
loaded_model = serialize.load("model.pth")
```

### 메트릭

```python
import rustorch_py.metrics as metrics

# 정확도
accuracy = metrics.accuracy(predictions, targets)

# F1 점수
f1 = metrics.f1_score(predictions, targets, average="macro")

# 혼동 행렬
confusion_matrix = metrics.confusion_matrix(predictions, targets)
```

### 데이터 유틸리티

```python
import rustorch_py.data as data

# DataLoader
dataset = data.TensorDataset(inputs, targets)
data_loader = data.DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

for batch in data_loader:
    loss = train_step(batch)
```

## 완전한 예제

### CNN을 이용한 분류

```python
import rustorch_py as torch
import rustorch_py.nn as nn
import rustorch_py.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 모델 생성
model = CNN()

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 훈련 루프
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"에포크 {epoch}: 손실 = {loss.item():.4f}")
```

이 참조서는 RusTorch Python API의 주요 기능을 다룹니다. 더 자세한 예제와 고급 사용 사례는 [Python 바인딩 완전 가이드](python_bindings_overview.md)와 [Jupyter 가이드](jupyter-guide.md)를 참조하세요.