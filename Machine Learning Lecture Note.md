# 자료 : "Machine Learning:a Lecture Note" (https://arxiv.org/pdf/2505.03861)
* 저자 : Kyunghyun Cho*

## Chapter 1: An Energy Function
# 에너지 기반 모델(Energy-Based Models, EBM): 머신러닝의 통합적 기초

## 1. 도입: 새로운 교육 방식의 필요성
기존의 머신러닝 교육은 [translate:이진 분류] $\rightarrow$ [translate:회귀] $\rightarrow$ [translate:신경망] $\rightarrow$ [translate:비지도 학습] 순으로 기법들을 나열하는 방식을 주로 사용합니다. 이러한 방식은 학생들이 각 기법 간의 연관성(예: 분류와 군집화의 관계)을 파악하고 통합적인 원리를 이해하는 데 어려움을 줍니다.

이 강의 노트는 **에너지 기반 모델(LeCun et al., 2006)**을 중심으로, 모든 머신러닝 과제를 '에너지 함수를 정의하고 최소화하는 과정'으로 해석하는 통합적 접근 방식을 취합니다.

## 2. 에너지 함수 (The Energy Function)
핵심 개념은 변수들 간의 **'호환성(Compatibility)'**을 측정하는 **에너지 함수(또는 부정적 호환성 점수)**입니다.

### 정의
에너지 함수 $e$는 다음과 같이 정의됩니다:

$$ e : X \times Z \times \Theta \to \mathbb{R} $$

*   **$X$**: 관측된 인스턴스(입력)의 집합
*   **$Z$**: 잠재 인스턴스(Latent variables, 은닉 변수)의 집합
*   **$\Theta$**: 파라미터의 집합
*   **$e(x, z, \theta)$**: 쌍 $(x, z)$의 호환성을 나타내는 실수 값을 반환

### 해석
*   **에너지가 낮음 (Low Energy)**: 호환성이 높음 (선호되는 상태)
*   **에너지가 높음 (High Energy)**: 호환성이 낮음 (선호되지 않는 상태)

### 잠재 변수 ($z$)의 역할
잠재 변수 $z$는 직접 관측되지 않지만 불확실성을 포착하는 중요한 역할을 합니다. 관측값 $x$에 대해 가능한 모든 $z$ 값들의 에너지 분포(평균, 분산)를 통해 불확실성을 파악할 수 있습니다.

## 3. 에너지 최소화로서의 추론 (Inference)
이 관점에서 **추론(Inference)**이란, 주어진 관측값에 대해 에너지 함수를 최소화하는 변수 값을 찾는 과정일 뿐입니다.

### 지도 학습 (Supervised Learning)
입력 $x'$가 주어졌을 때, 에너지를 최소화하는 출력 $y$를 찾습니다.
$$ \hat{y} = \arg \min_{y \in Y} e([x', y], \emptyset, \theta) $$
*   **분류 (Classification)**: $Y$가 이산적인 집합일 때
*   **회귀 (Regression)**: $Y$가 연속적인 변수일 때

### 비지도 학습 (Unsupervised Learning)
관측값 $x$가 주어졌을 때, 에너지를 최소화하는 잠재 변수 $z$를 찾습니다.
$$ \hat{z} = \arg \min_{z \in Z} e(x, z, \theta) $$
*   **군집화 (Clustering)**: $Z$가 이산적인 집합일 때 (어떤 클러스터에 속하는지 할당)
*   **표현 학습 (Representation Learning)**: $Z$가 연속적인 변수일 때

## 4. 학습: 파라미터 $\Theta$ 추정 (Learning)
**학습(Learning)**은 에너지 표면(Energy Surface)의 형상을 올바르게 만드는 파라미터 $\theta$를 찾는 과정입니다.

### 난제
단순히 관측된 데이터($x \sim p_{data}$)의 에너지만 낮추는 것(식 1.6)으로는 부족합니다. **관측된 데이터의 에너지는 낮추고, 관측되지 않은(원치 않는) 데이터의 에너지는 상대적으로 높여야** 합니다.

### 정규화 (Regularization)
이를 위해 적절한 정규화 항 $R(\theta)$을 도입하여 학습을 수행합니다:

$$ \min_{\theta \in \Theta} E_{x \sim p_{data}} [e(x, \emptyset, \theta) - R(\theta)] $$

## 5. 요약: 머신러닝의 3가지 축
모든 머신러닝 문제는 에너지 함수를 중심으로 다음 세 단계로 요약됩니다:

1.  **Parametrization (파라미터화)**: 에너지 함수 $e$를 정의하는 단계
2.  **Learning (학습)**: 데이터로부터 파라미터 $\theta$를 추정하는 단계 (에너지 지형 형성)
3.  **Inference (추론)**: 부분적인 관측값이 주어졌을 때, 에너지를 최소화하는 결측값(출력 또는 잠재변수)을 찾는 단계

## 1. EBM에서의 분류 (Classification Setup)
에너지 기반 모델에서 분류 문제는 관측값 $x$를 입력과 출력 $[x, y]$로 나누는 것입니다. 여기서 $y$는 유한한 카테고리 집합 $Y$에 속하며, 잠재 변수는 없다고 가정합니다($Z = \emptyset$).

### 추론 (Inference)
가장 낮은 에너지를 가진 카테고리를 선택합니다:
$$ \hat{y}(x) = \arg \min_{y \in Y} e([x, y], \emptyset, \theta) $$

### 효율적인 파라미터화 (Parametrization)
에너지 계산의 효율성을 위해(병렬 계산 등), 특징 추출기 $f(x, \theta)$와 원-핫 벡터 $\mathbf{1}(y)$를 사용하여 에너지를 정의합니다:
$$ e([x, y], \emptyset, \theta) = \mathbf{1}(y)^\top f(x, \theta) $$
*   **선형 분류기 (Linear Classifier)**: $f(x, \theta) = Wx + b$인 경우 ($\theta = (W, b)$).

## 2. 학습을 위한 손실 함수 (Loss Functions)
학습은 훈련 데이터셋 $D = \{[x^n, y^n]\}_{n=1}^N$에 대한 평균 손실 함수를 최소화하는 과정입니다.

### Zero-One (0-1) Loss의 한계
이상적인 목표는 **0-1 손실**을 최소화하는 것입니다:
$$ L_{0-1}([x, y], \theta) = \mathbf{1}(y \neq \hat{y}(x)) $$
*   **문제점**: 이 함수는 계단형 상수 함수(piece-wise constant)이므로 미분값이 대부분 0입니다. 따라서 경사 하강법을 사용할 수 없으며 난해한 블랙박스 최적화가 필요합니다.

### 대리 손실 함수 (Proxy Loss Functions)
미분 가능한 대리 손실 함수를 사용하여 이를 해결합니다.

#### 1. 마진 손실 (Margin Loss / Hinge Loss)
정답 $y$의 에너지가 오답 중 가장 에너지가 낮은 $\hat{y}'$보다 적어도 마진 $m$만큼 더 낮도록 강제합니다.
$$ L_{margin}([x, y], \theta) = \max(0, m + e([x, y], \emptyset, \theta) - e([x, \hat{y}'], \emptyset, \theta)) $$
*   **SVM**: 서포트 벡터 머신(Support Vector Machine)의 핵심 원리입니다.

#### 2. 퍼셉트론 손실 (Perceptron Loss)
마진 손실에서 $m=0$인 특수 경우입니다.
$$ L_{perceptron}([x, y], \theta) = \max(0, e([x, y], \emptyset, \theta) - e([x, \hat{y}'], \emptyset, \theta)) $$
*   **특징**: 예측이 틀렸을 때만($y \neq \hat{y}$) 파라미터를 업데이트합니다. 정답이면 손실은 0입니다.

## 3. 확률적 접근: 소프트맥스와 교차 엔트로피
**최대 엔트로피 원리(Principle of Maximum Entropy)**를 사용하여 에너지 값을 확률로 변환할 수 있습니다.

### 소프트맥스 변환 (Softmax)
에너지를 카테고리 확률 $p_\theta(y|x)$로 변환:
$$ p_\theta(y|x) = \frac{\exp(-e([x, y], \emptyset, \theta))}{\sum_{y' \in Y} \exp(-e([x, y'], \emptyset, \theta))} $$

### 교차 엔트로피 손실 (Cross-Entropy / Negative Log-Likelihood)
$$ L_{ce}([x, y], \theta) = -\log p_\theta(y|x) = e([x, y], \emptyset, \theta) + \log \sum_{y' \in Y} \exp(-e([x, y'], \emptyset, \theta)) $$

### 기울기 분석 (Boltzmann Machine Learning)
교차 엔트로피 손실의 기울기(Gradient)는 학습의 동역학을 보여줍니다:
$$ \nabla_\theta L_{ce} = \underbrace{\nabla_\theta e([x, y], \emptyset, \theta)}_{\text{Positive Phase}} - \underbrace{E_{y|x;\theta}[\nabla_\theta e([x, y'], \emptyset, \theta)]}_{\text{Negative Phase}} $$
*   **Positive Phase**: 정답 $y$의 에너지를 높이는 방향 (손실 최소화 과정에서 부호가 반전되어 에너지를 낮춤).
*   **Negative Phase**: 현재 모델의 확률 분포에 따라 가중된 다른 모든 라벨의 에너지를 낮추는 방향 (실제로는 에너지를 높여 정답과 격차를 벌림).

### 퍼셉트론과의 연결
소프트맥스 함수에서 역온도(inverse temperature) $\beta \to \infty$가 되면:
*   Negative Phase 항이 예측된 클래스 $\hat{y}$의 에너지에 지배됩니다.
*   만약 $\hat{y} = y$라면 기울기가 상쇄되어 업데이트가 발생하지 않습니다.
*   이는 퍼셉트론 손실의 동작과 일치하게 됩니다.
