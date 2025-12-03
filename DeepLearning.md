# SEAI 1 – 뇌 vs. 딥러닝 정리 자료

## 1️⃣ 서론
인공지능(AI) 연구는 **뇌(생물학적 신경망)**와 딥러닝(인공 신경망) 두 축으로 나뉜다.
두 접근법은 입력 → 가중치·편향 → 활성화 함수 → 출력이라는 기본 흐름은 동일하지만, 구현 방식·학습 방법·해석 가능성에서 차이가 있다.

## 2️⃣ 퍼셉트론(Perceptron) 기본 개념
퍼셉트론은 딥러닝 세포라고 보면 된다.

| 요소 | 의미 |
|------|------|
| **입력 (x_i)** | 신경에 들어오는 값 (예: 이미지 픽셀, 특징) |
| **가중치 (w_i)** | 뉴런 간의 연결강도에 대응하는 것으로 각 입력이 결과에 미치는 영향 정도 |
| **편향 (b)** | 뉴런의 역치에 대응함. 신호 가중합이 일정 크기보다 작으면 무시하게 됨. 전체 출력값을 좌우하는 상수 (threshold) |
| **활성화 함수 (Activation Function)** | 선형 결합 결과를 비선형으로 변환해 다음 층에 전달 |

**수식**: \( y = f\!\left(\sum_i w_i x_i + b\right) \)
-> 가중치와 편향은 학습을 통해 변경되는 파라미터. 수식을 풀어 쓰자면, 입력 각각에 가중치\(w\)를 곱한 후 합산하고, 역치만큼 조정한 뒤, 활성함수 \(f\) 적용한 것.

퍼셉트론은 이진 분류에 사용되며, 출력이 0 또는 1(또는 -1, 1)이다.
학습은 오차 역전파가 아니라 퍼셉트론 학습 규칙(오차가 있으면 가중치와 편향을 조정)으로 수행한다.

## 3️⃣ 활성화 함수(Activation Functions)
| 함수 | 형태 | 특징 | 사용 예시 |
|------|------|------|----------|
| **ReLU (Rectified Linear Unit)** | \(f(x)=\max(0, x)\) | 0 이하를 0으로, 양수는 그대로. 희소성과 기울기 소실 방지에 유리. | 대부분의 CNN, DNN 은 ReLU 사용 |
| **Leaky ReLU** | \(f(x)=\max(\alpha x, x)\) (\(\alpha\approx0.01\)) | ReLU의 “죽은 뉴런” 문제 완화. | 일부 이미지/음성 모델 |
| **Sigmoid** | \(f(x)=\frac{1}{1+e^{-x}}\) | 출력이 0~1 사이, 확률 해석 가능. 기울기 소실이 심함. | 로지스틱 회귀, 출력층(이진) |
| **tanh** | \(f(x)=\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}\) | 출력이 -1~1, 중심이 0이라 학습이 빠름. | 은닉층(소규모) |
| **ELU (Exponential Linear Unit)** | \(f(x)=x\) (x>0), \(f(x)=\alpha(e^{x}-1)\) (x≤0) | 음수 영역에서도 작은 기울기 제공, 평균 출력이 0에 가깝게 유지. | 딥 네트워크에서 성능 향상 |

**핵심 포인트**  
- ReLU는 현재 가장 널리 쓰이며, 희소성(많은 뉴런이 0) 덕분에 연산 효율이 높다.  
- Sigmoid / tanh는 출력 해석이 쉬우나 깊은 네트워크에서는 기울기 소실 문제로 잘 쓰이지 않는다.  
- Leaky ReLU, ELU는 ReLU의 단점을 보완한 변형이다.

## 4️⃣ 가중치와 편향 학습
**전방 전파(Forward Pass)**  
입력 → 가중치·편향 → 활성화 → 다음 층으로 전달.  
**손실 함수(Loss)**  
예: 교차 엔트로피, MSE 등으로 모델 출력과 정답 차이 측정.  
**역전파(Back‑Propagation)**  
손실을 가중치·편향에 대한 미분으로 전파, 경사 하강법(SGD, Adam 등)으로 업데이트.  
**학습률(Learning Rate)**, **모멘텀**, **정규화(L2, Dropout)** 등 하이퍼파라미터 조절이 필요.

## 5️⃣ 뇌와 딥러닝의 차이점 요약
| 구분 | 뇌(생물학적) | 딥러닝(인공) |
|------|--------------|--------------|
| **구조** | 뉴런·시냅스·다양한 화학·전기 신호 | 수학적 연산(가중치·편향·활성화) |
| **학습** | 시냅스 가소성, 장기·단기 기억, 강화학습 등 복합 | 손실 최소화, 역전파 기반 최적화 |
| **해석 가능성** | 부분적(뇌영상, 전기생리) | 가중치 시각화, Grad‑CAM 등으로 부분 해석 가능 |
| **에너지 효율** | 매우 효율적(뇌는 20W) | GPU/TPU 사용 시 전력 소모 큼 |
| **범용성** | 다중 감각·운동·인지 통합 | 특정 태스크에 특화, 전이학습 필요 |

## 6️⃣ 정리 및 학습 포인트
- 퍼셉트론은 인공 신경망의 가장 기본 단위이며, 가중치·편향·활성화 함수가 핵심.  
- 활성화 함수 선택은 모델 깊이·데이터 특성에 따라 달라진다. 현재는 ReLU가 기본, 필요 시 Leaky/ELU 등 변형 사용.  
- 역전파와 경사 하강법이 딥러닝 학습의 핵심 메커니즘이며, 하이퍼파라미터 튜닝이 성능을 좌우한다.  
- 뇌와 딥러닝은 구조·학습·해석에서 차이가 있지만, 입력‑가중치‑비선형‑출력이라는 기본 흐름은 동일하다.

# SEAI 2 – 추상화와 임베딩 (Abstraction & Embedding) 정리

## 1️⃣ 핵심 개념
| 용어 | 정의 | 주요 포인트 |
|------|------|------|
| **추상화 (Abstraction)** | 복잡한 현상을 핵심 특징만 골라내어 간단한 형태로 표현하는 과정. 데이터·문맥·관계 등을 고차원 공간에 매핑한다. | - 인간이 개념을 형성하는 방식과 유사.<br>- 머신러닝에서는 특징 추출·차원 축소가 추상화에 해당. |
| **임베딩 (Embedding)** | 추상화된 개념을 연속적인 실수 벡터(dense vector)로 변환한 것. 동일한 의미·관계를 가진 항목은 벡터 공간에서 가깝게 위치한다. | - 텍스트, 이미지, 그래프 등 다양한 도메인에 적용.<br>- 학습된 임베딩은 전이 학습에 재사용 가능. |

## 학습과 추론
학습 과정 = '어떤 입력이 어떤 개념으로 이어지는 연결을 강화' 하여 추상화
추론 과정 = '이전에 학습해 놓은 특징이 발견되면 연결된 개념을 활성화' 하는 개념화 -> 특징을 보면 개념이 떠오르게 함.
"딥러닝 신경망에서는 이러한 개념화가 여러 층에 걸쳐 반복적으로 일어나고, 입력이 조합되어 개념이 되고, 개념이 조합되어 또 다른 개념이 된다"
- 딥러닝 신경망의 동작은 ‘아래층의 입력이 이러하면 위층의 어떤 개념을 활성화 시킨다’는 동작입니다. 때문에, 나중에 살펴보게 될 트랜스포머 모델의 구성을 분석할 때에 딥러닝으로 학습된 인공신경망이 들어가 있는 블럭은 ‘데이터를 추상화하여 개념에 매칭시키는 기능을 한다’고 이해
- CNN은 경량화 모델에서는 여전히 잘 사용되는 방식입니다만, 트랜스포머 모델이 LLM에서 크게 성공한 이후로 비전 트랜스포머 모델(ViT)이 현재 가장 높은 성능을 내는 것으로 보임

## 인공 신경망은 의미를 공간의 좌표로 다룬다
- 언어모델은 ‘유사한 의미를 갖는 단어들은 주위의 단어들도 유사하다’는 분포의미론을 바탕으로 저수준 개념화를 시작
- 문장에서 단어를 바꿔 넣을 때, 유사한 의미의 단어일수록 많은 문장에서 호환
  - ex) 나는 [강아지]를 키운다. 나의 [강아지]는 귀엽다. 나의 [강아지]는 충성스럽다.
  - 이러한 문장 구조들의 주위 단어 등장 횟수를 세어서 벡터화하면 '사전의 모든 단어'만큼의 열을 갖는 표를 그릴 수 있음.
 
## 2️⃣ 의미 공간 (Semantic Spaces)
의미 공간은 고차원 벡터 공간으로, 각 차원은 특정 의미/속성을 나타낸다.  
같은 의미를 가진 단어·문장은 **거리(코사인 유사도 등)**가 작아진다.  
예시: “king – man + woman ≈ queen” (Word2Vec에서 관찰되는 관계).

## 3️⃣ Word2Vec : 단어를 벡터로 바꾸는 방법이며, 구글에서 만든 차원 축소 방법 (신경망에 단어를 학습시켜서 변환하는 방법)
| 요소 | 설명 |
|------|------|
| **모델** | CBOW(Context → Target)와 Skip‑gram(Target → Context) 두 가지 학습 방식. |
| **학습 목표** | 주변 단어(맥락)로부터 중심 단어를 예측하거나, 반대로 중심 단어로부터 주변 단어를 예측해 단어 벡터를 학습. |
| **핵심 아이디어** | 분포 의미론(Distributional Semantics) – “같은 맥락에 자주 등장하는 단어는 의미가 비슷하다.” |
| **벡터 차원** | 일반적으로 100~300 차원, 최신 모델(BERT, GPT)에서는 768~1024 차원까지 확장. |
| **응용** | 유사도 검색, 군집화, 문서 분류, 전이 학습 등 다양한 NLP 파이프라인의 기초 임베딩으로 활용. |

**Word2Vec 작동 흐름 (요약)**
1. **입력**: 단어를 원‑핫(one‑hot) 벡터로 표현.  
2. **임베딩 레이어**: 원‑핫 → 가중치 행렬(Embedding matrix) 곱 → 밀집 벡터.  
3. **출력 레이어**: 임베딩 벡터 → 소프트맥스를 통해 주변 단어 확률 예측.  
4. **역전파**: 손실을 최소화하도록 **가중치(임베딩 행렬)**를 업데이트.

## 4️⃣ 임베딩의 활용 예시
- **텍스트 유사도** – 문서·문장 간 코사인 유사도로 검색·추천.  
- **클러스터링** – 같은 의미를 가진 단어·문장을 군집화.  
- **전이 학습** – 사전 학습된 임베딩을 특정 도메인(예: 의료, 법률) 데이터에 fine‑tune.  
- **다중 모달** – 이미지 캡션, 비디오 설명 등 텍스트와 비텍스트를 동일 의미 공간에 매핑.

## 5️⃣ 요약 정리
- 추상화는 복잡한 정보를 핵심적인 특징으로 압축하는 과정이며, 임베딩은 그 추상화된 정보를 연속적인 벡터 형태로 구현한다.  
- 의미 공간은 이러한 벡터들이 의미적 관계를 보존하도록 배치된 고차원 공간이다.  
- Word2Vec은 가장 대표적인 단어 임베딩 방법으로, 분포 의미론에 기반해 단어 간 의미적 유사성을 학습한다.  
- 학습된 임베딩은 유사도 검색, 클러스터링, 전이 학습 등 다양한 AI 응용에 핵심적인 역할을 한다.

**학습 포인트**  
- 추상화와 임베딩의 차이와 관계 이해  
- 의미 공간에서 벡터 거리 해석 방법 습득  
- Word2Vec 구조와 학습 원리 파악  
- 임베딩을 실제 프로젝트에 적용하는 방법 탐색

**추가 참고**  
- Mikolov et al., “Efficient Estimation of Word Representations in Vector Space”, 2013.  
- Jurafsky & Martin, *Speech and Language Processing*, 3rd ed., Chapter on Word Embeddings.



# PyTorch 튜토리얼

## 목차
1. [텐서 생성과 기본 연산](#1-텐서-생성과-기본-연산)  
2. [GPU 가속 활용](#2-gpu-가속-활용)  
3. [자동 미분 Autograd](#3-자동-미분-autograd)  
4. [신경망 모델 정의](#4-신경망-모델-정의)  
5. [데이터 로더 활용](#5-데이터-로더-활용)  
6. [손실 함수와 최적화](#6-손실-함수와-최적화)  
7. [모델 학습 루프](#7-모델-학습-루프)  
8. [모델 평가와 예측](#8-모델-평가와-예측)  
9. [학습률 스케줄러](#9-학습률-스케줄러)  
10. [모델 저장과 로드](#10-모델-저장과-로드)  
11. [커스텀 데이터셋 생성](#11-커스텀-데이터셋-생성)  
12. [전이 학습 활용](#12-전이-학습-활용)  

---

### 1. 텐서 생성과 기본 연산
**개요**  
PyTorch의 핵심 자료구조인 텐서를 생성하고 기본적인 수학 연산을 수행하는 방법입니다.

**코드 예제**
```python
import torch

# 텐서 생성
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.ones(2, 2)

# 기본 연산
result = x + y
print(result)  # tensor([[2., 3.],
               #         [4., 5.]])
```

**설명**  
`torch.tensor()` 로 데이터를 텐서로 변환하고, NumPy처럼 직관적인 연산이 가능합니다. GPU 가속을 위한 기본 단위입니다.

---

### 2. GPU 가속 활용
**개요**  
CUDA를 이용해 텐서를 GPU로 이동시켜 연산 속도를 대폭 향상시키는 방법입니다.

**코드 예제**
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

result = torch.mm(x, y)  # GPU에서 행렬 곱셈
print(f"Device: {result.device}")
```

**설명**  
`.to(device)` 로 텐서를 GPU/CPU로 이동시킵니다. 대규모 행렬 연산에서 GPU는 CPU 대비 수십 배 빠릅니다.

---

### 3. 자동 미분 Autograd
**개요**  
역전파를 위한 자동 미분 기능으로, 신경망 학습의 핵심 메커니즘입니다.

**코드 예제**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x ** 2 + x

y.backward()  # dy/dx 계산
print(f"Gradient: {x.grad}")  # 3*4 + 2*2*2 + 1 = 21
```

**설명**  
`requires_grad=True` 로 미분 추적을 활성화하고, `backward()` 로 그래디언트를 자동 계산합니다. 복잡한 수식도 자동으로 처리됩니다.

---

### 4. 신경망 모델 정의
**개요**  
`nn.Module` 을 상속받아 커스텀 신경망을 정의하는 표준 방법입니다.

**코드 예제**
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**설명**  
`__init__` 에서 레이어를 정의하고 `forward` 에서 순전파 로직을 구현합니다. 역전파는 autograd가 자동 처리합니다.

---

### 5. 데이터 로더 활용
**개요**  
대용량 데이터를 배치 단위로 효율적으로 로드하고 셔플하는 방법입니다.

**코드 예제**
```python
from torch.utils.data import DataLoader, TensorDataset
import torch

X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_X, batch_y in loader:
    print(batch_X.shape)  # torch.Size([32, 20])
```

**설명**  
`TensorDataset` 으로 데이터를 묶고 `DataLoader` 로 배치 처리합니다. `shuffle=True` 로 에폭마다 데이터 순서를 섞어 학습 효과를 높입니다.

---

### 6. 손실 함수와 최적화
**개요**  
모델 학습을 위한 손실 함수 계산과 옵티마이저 사용법입니다.

**코드 예제**
```python
import torch.nn as nn
import torch.optim as optim

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

output = model(torch.randn(10, 784))
loss = criterion(output, torch.randint(0, 10, (10,)))

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**설명**  
`CrossEntropyLoss` 로 분류 손실을 계산하고, `Adam` 옵티마이저로 가중치를 업데이트합니다. `zero_grad()` 로 그래디언트 초기화가 필수입니다.

---

### 7. 모델 학습 루프
**개요**  
전체 학습 과정을 포함한 완전한 트레이닝 루프 구현입니다.

**코드 예제**
```python
model.train()
for epoch in range(10):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss {total_loss:.4f}")
```

**설명**  
`model.train()` 으로 학습 모드 활성화 후, 배치별로 순전파 → 손실 계산 → 역전파 → 가중치 업데이트를 반복합니다.

---

### 8. 모델 평가와 예측
**개요**  
학습된 모델로 새로운 데이터에 대한 예측을 수행하는 방법입니다.

**코드 예제**
```python
model.eval()
with torch.no_grad():
    test_input = torch.randn(5, 784)
    predictions = model(test_input)
    predicted_classes = torch.argmax(predictions, dim=1)

print(predicted_classes)  # 예: tensor([7, 2, 0, 4, 9])
```

**설명**  
`model.eval()` 과 `torch.no_grad()` 로 그래디언트 계산을 비활성화해 메모리를 절약합니다. `argmax` 로 가장 높은 확률의 클래스를 선택합니다.

---

### 9. 학습률 스케줄러
**개요**  
학습 진행에 따라 학습률을 동적으로 조정하여 수렴 성능을 개선합니다.

**코드 예제**
```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(20):
    # ... 학습 코드 ...
    scheduler.step()
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
```

**설명**  
`StepLR` 은 `step_size` 에폭마다 학습률을 `gamma` 배로 감소시킵니다. 초반엔 빠르게 학습하고 후반엔 세밀하게 조정합니다.

---

### 10. 모델 저장과 로드
**개요**  
학습된 모델의 가중치를 저장하고 나중에 불러오는 방법입니다.

**코드 예제**
```python
# 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 로드
model = SimpleNet()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

**설명**  
`state_dict()` 로 모델의 파라미터만 저장하는 것이 권장됩니다. 전체 모델을 저장하는 것보다 유연하고 호환성이 좋습니다.

---

### 11. 커스텀 데이터셋 생성
**개요**  
자신만의 데이터를 PyTorch `Dataset` 으로 변환하여 표준 파이프라인에 통합합니다.

**코드 예제**
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

**설명**  
`Dataset` 클래스를 상속받아 `__len__` 과 `__getitem__` 만 구현하면 됩니다. `DataLoader` 와 함께 사용해 표준화된 데이터 처리가 가능합니다.

---

### 12. 전이 학습 활용
**개요**  
사전 학습된 모델을 가져와 마지막 레이어만 교체하여 새로운 태스크에 적용합니다.

**코드 예제**
```python
import torchvision.models as models
import torch.nn as nn

resnet = models.resnet18(pretrained=True)

# 마지막 레이어만 교체
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)

# 사전 학습 레이어는 고정
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc.requires_grad = True
```

**설명**  
ImageNet 으로 학습된 ResNet을 가져와 분류기만 교체합니다. 적은 데이터로도 높은 성능을 얻을 수 있는 강력한 기법입니다.
