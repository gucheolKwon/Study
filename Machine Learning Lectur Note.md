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
