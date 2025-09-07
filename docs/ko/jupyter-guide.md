# RusTorch WASM Jupyter Notebook 가이드

초보자를 위해 설계된 Jupyter Notebook에서 RusTorch WASM을 쉽게 사용하는 단계별 가이드입니다.

## 📚 목차

1. [요구사항](#요구사항)
2. [설치 지침](#설치-지침)
3. [기본 사용법](#기본-사용법)
4. [실용적 예제](#실용적-예제)
5. [문제 해결](#문제-해결)
6. [FAQ](#faq)

## 요구사항

### 최소 요구사항
- **Python 3.8+**
- **Jupyter Notebook** 또는 **Jupyter Lab**
- **Node.js 16+** (WASM 빌드용)
- **Rust** (최신 안정 버전)
- **wasm-pack** (Rust 코드를 WASM으로 변환)

### 권장 환경
- 메모리: 8GB 이상
- 브라우저: Chrome, Firefox, Safari 최신 버전
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## 설치 지침

### 🚀 빠른 시작 (권장)

#### 범용 설치 프로그램 (신규)
**가장 쉬운 방법**: 환경을 자동으로 감지하는 설치 프로그램
```bash
./install_jupyter.sh
```

**기능:**
- 🔍 **자동 감지**: 환경을 자동으로 감지 (OS, CPU, GPU)
- 🦀🐍 **하이브리드 환경**: 기본적으로 Python+Rust 이중 환경 설치
- 📦 **전역 명령어**: 어디서든 작동하는 `rustorch-jupyter` 명령어 생성
- ⚡ **최적화**: 하드웨어에 맞게 최적화 (CUDA, Metal, WebGPU, CPU)

#### 기존 방법
**전통적인 방법**: RusTorch가 포함된 Python만 시작
```bash
./start_jupyter.sh
```

이 스크립트는 자동으로:
- 가상 환경 생성 및 활성화
- 의존성 설치 (numpy, jupyter, matplotlib)
- RusTorch Python 바인딩 빌드
- 데모 노트북과 함께 Jupyter Lab 시작

#### 다음 실행
```bash
rustorch-jupyter          # 전역 명령어 (설치 프로그램 사용 후)
# 또는
./start_jupyter_quick.sh  # 대화형 메뉴
```

### 수동 설치

#### 단계 1: 기본 도구 설치

```bash
# Python 버전 확인
python --version

# Jupyter Lab 설치
pip install jupyterlab

# Node.js 설치 (macOS Homebrew 사용)
brew install node

# Rust 설치
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# wasm-pack 설치
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### 단계 2: RusTorch WASM 빌드

```bash
# 프로젝트 클론
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# WASM 타겟 추가
rustup target add wasm32-unknown-unknown

# wasm-pack으로 빌드
wasm-pack build --target web --out-dir pkg
```

#### 단계 3: Jupyter 시작

```bash
# Jupyter Lab 시작
jupyter lab
```

## 환경 유형

### 🦀🐍 하이브리드 환경 (기본값)
- **적합한 용도**: 풀스택 ML 개발
- **기능**: Python + Rust 커널, RusTorch 브리지, 예제 노트북
- **하드웨어**: 사용 가능한 GPU에 적응 (CUDA/Metal/CPU)

### 🐍 Python 환경
- **적합한 용도**: RusTorch 기능을 원하는 Python 개발자
- **기능**: RusTorch Python 바인딩이 있는 Python 커널
- **하드웨어**: CPU/GPU 최적화

### ⚡ WebGPU 환경
- **적합한 용도**: 브라우저 기반 GPU 가속
- **기능**: WebAssembly + WebGPU, Chrome 최적화
- **하드웨어**: WebGPU 지원이 있는 최신 브라우저

### 🦀 Rust 커널 환경
- **적합한 용도**: 네이티브 Rust 개발
- **기능**: evcxr 커널, RusTorch 라이브러리에 직접 액세스
- **하드웨어**: 네이티브 성능, 모든 기능 사용 가능

## 기본 사용법

### 텐서 생성

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 1D 텐서
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('1D 텐서:', vec.to_array());
    
    // 2D 텐서 (행렬)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // 모양: 2행 3열
    );
    console.log('2D 텐서 모양:', matrix.shape());
});
```

### 기본 연산

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // 덧셈
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // 행렬 곱셈
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### 자동 미분

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 그래디언트 추적이 활성화된 텐서 생성
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // 계산: y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // 역전파
    y.backward();
    
    // 그래디언트 얻기 (dy/dx = 2x + 3 = 7 when x=2)
    console.log('그래디언트:', x.grad().to_array());
});
```

## 실용적 예제

### 선형 회귀

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // 데이터 준비
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // 매개변수 초기화
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // 훈련 루프
    for (let epoch = 0; epoch < 100; epoch++) {
        // 예측: y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // 손실: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // 그래디언트 계산
        loss.backward();
        
        // 매개변수 업데이트
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // 그래디언트 리셋
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`에포크 ${epoch}: 손실 = ${loss.item()}`);
        }
    }
    
    console.log(`최종 w: ${w.item()}, b: ${b.item()}`);
});
```

## 문제 해결

### 🚀 Rust 커널 가속화 (권장)
초기 실행이 느린 경우, 캐싱을 활성화하여 성능을 크게 향상시킬 수 있습니다:

```bash
# 캐시 디렉토리 생성
mkdir -p ~/.config/evcxr

# 500MB 캐시 활성화
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**효과:**
- 첫 번째: 일반적인 컴파일 시간
- 후속 실행: 의존성 재컴파일 없음 (몇 배 빠름)
- `rustorch` 라이브러리도 첫 사용 후 캐시됨

**참고:** 라이브러리 업데이트 후 `:clear_cache`로 캐시 새로고침 권장

### 일반적인 오류

#### "RusTorch is not defined" 오류
**해결책**: 항상 RusTorchReady를 기다리세요
```javascript
window.RusTorchReady.then((rt) => {
    // 여기서 RusTorch 사용
});
```

#### "Failed to load WASM module" 오류
**해결책**:
1. `pkg` 디렉토리가 올바르게 생성되었는지 확인
2. 브라우저 콘솔에서 오류 메시지 확인
3. WASM 파일 경로가 올바른지 확인

#### 메모리 부족 오류
**해결책**:
```javascript
// 메모리를 명시적으로 해제
tensor.free();

// 더 작은 배치 크기 사용
const batchSize = 32;  // 1000 대신 32 사용
```

### 성능 팁

1. **배치 처리 사용**: 루프 대신 배치로 데이터 처리
2. **메모리 관리**: 큰 텐서를 명시적으로 해제
3. **적절한 데이터 타입**: 높은 정밀도가 필요하지 않을 때 f32 사용

## FAQ

### Q: Google Colab에서 사용할 수 있나요?
**A**: 네, WASM 파일을 업로드하고 사용자 정의 JavaScript 로더를 사용하세요.

### Q: Python과 WASM 코드를 혼합할 수 있나요?
**A**: 네, IPython.display.Javascript를 사용하여 Python과 JavaScript 간에 데이터를 전달할 수 있습니다.

### Q: 디버깅은 어떻게 하나요?
**A**: 브라우저 개발자 도구(F12)를 사용하고 Console 탭에서 오류를 확인하세요.

### Q: 어떤 고급 기능을 사용할 수 있나요?
**A**: 현재 기본 텐서 연산, 자동 미분, 간단한 신경망을 지원합니다. CNN 및 RNN 레이어가 계획되어 있습니다.

## 다음 단계

1. 📖 [상세한 RusTorch WASM API](../wasm.md)
2. 🔬 [고급 예제](../examples/)
3. 🚀 [성능 최적화 가이드](../wasm-memory-optimization.md)

## 커뮤니티 및 지원

- GitHub: [RusTorch 저장소](https://github.com/JunSuzukiJapan/rustorch)
- Issues: GitHub에서 버그 신고 및 기능 요청

---

RusTorch WASM으로 즐거운 학습하세요! 🦀🔥📓