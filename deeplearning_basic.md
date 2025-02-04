딥러닝 기초 - KAIST 오혜연 교수님


인공 신경망 모델 중 3층 이상의 히든층을 갖고 있을 경우 딥러닝 모델
Loss function을 최소화하기 위해 최적화 알고리즘을 적용
Loss function : 예측값과 실제값간의 오차값
Optimization : 오차값을 최소화하는 모델의 인자를 찾는 것

기본적인 최적화 알고리즘 : Gradient Descent(GD)
: 신경망의 가중치들을 W라고 했을 때, 손실함수 Loss (W)의 값을 최소화하기 위해 기울기 델타Loss(W)를 이용하는 방법

각 가중치들의 기울기를 구하는 방법 : 역전파(Backpropagation)
: target 값과 실제 모델이 예측한 output 값이 얼마나 차이나는지 구한 후 오차값을 다시 뒤로 전파해가며 변수들을 갱신하는 알고리즘

TensorFlow
: 딥러닝 모델 구현을 위해 사용하는 프레임워크 중 하나
유연하고, 효율적, 확장성 있음
Tensor : Multidimensional Arrays = Data
: 다차원 배열로 나타내는 데이터를의미
Flow : 데이터의 흐름
계산이 데이터 플로우 그래프로 수행되며, 그래프를 따라 데이터가 노드를 거쳐 흘러가면서 계산

상수 텐서 (Constant Tensor)
: value - 반환되는 상수값, shape - Tensor의 차원, dtype - 반환되는 Tensor 타입, name - 텐서 이름

import tensorflow as tf
 
#상수형 텐서 생성
tensor_a = tf.constant(value, dtype=None, shape=None, name=None)
 
#모든 원소 값이 0 인 텐서 생성
tensor_b = tf.zeros(shape, dtype=tf.float32,name=None)
 
#모든 원소 값이 1인 텐서 생성
tensor_c = tf.ones(shape, dtype=tf.float32,name=None)
 
# start에서 stop까지 증가하는 num 개수 데이터를 가진 텐서 생성
tensor_d = tf.linspace(start, stop, num,name=None)
시퀀스 텐서 (Sequence Tensor)

import tensorflow as tf
 
# start에서 limit까지 delta씩 증가하는 데이터를 가진 텐서 생성
tensor_e = tf.range(start, limit, delta, name=None)
변수 텐서 (Variable Tensor)

import tensorflow as tf
 
#변수형 텐서 생성
tensor_f = tf.Variable(initial_value=None, dtype=None, name=None)
Epoch 
: 한번의 epoch = 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
Batch 
: 나눠진 데이터 셋 (보통 mini-batch라고 표현)
iteration 
: epoch를 나누어서 실행하는 횟수를 의미

TensorFlow로 딥러닝 모델 구현하기
데이터셋 준비하기 코드 예시
import tensorflow as tf
 
data = np.random.sample((100,2))
labels = np.random.sample((100,1))
 
# numpy array로부터 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
Keras 
: 텐서플로우의 패키지로 제공되는 고수준 API. 딥러닝 모델을 간단하고 빠르게 구현 가능
딥러닝 모델 구축을 위한 Keras 메서드
모델 클래스 객체 생성 - tf.keras.models.Sequential()
모델의 각 Layer 구성 - tf.keras.layers.Dense(units, activation)
units : 레이어 안의 Node의 수
activation : 적용할 activation 함수 설정

모델에 Layer 추가 - [model].add(tf.keras.layers.Dense(units, activation))
모델 학습 방식을 설정하기 위한 함수 - [model].complie(optimizer, loss)
모델을 학습시키기 위한 함수 - [model].fit(x,y)
모델 평가 - [model].evaluate(x,y)
모델로 예측을 수행 - [model].predict(x)
Input Layer의 입력 형태 지정
: 첫번째 = Input Layer는 입력 형태에 대한 정보를 필요로 함. 
input_shape / input_dim 인자 설정하기

import tensorflow as tf
 
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
 
model.compile(loss='mean_squared_error', optimizer='SGD')
model.fit(dataset, epochs=100)
 
# 테스트 데이터 준비하기
dataset_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test))
dataset_test = dataset.batch(32)
 
# 모델 평가 및 예측하기
model.evaluate(dataset_test)
predicted_labels_test = model.predict(data_test)


딥러닝 모델 학습의 문제점

실생활 문제 데이터의 차원이 증가 + 구조가 복잡
학습 속도 문제
최적화 알고리즘 : 전체 데이터가 아닌 부분 데이터만 활용해서 손실 함수를 계산 = SGD(Stochastic Gradient Descent)
→ 한계
: gradient값 계산 시, mini-batch에 따라 gradient 방향의 변화가 큼
: Learning Rate 설정에에 있어서도 문제가 생길 수 있음 

==> 최종적으로 요즘은 Adam(Momentum + RMSProp)이라는 알고리즘을 사용함


기울기 소실 문제
ReLU : 기존에 사용하던 sigmoid 함수 대신 다른 활성화 함수
→ 내부 Hidden Layer 에는 ReLU
Tanh 
→ 외부 Output Layer 에는 Tanh

초기값 설정 문제 - 초기값 설정 방식에 따른 성능 차이가 매우 크게 발생
가중치 초기화(Weight Initialization)
: 활성화 함수의 입력 값이 너무 커지거나 작아지지 않게 만들어주려는 것이 핵심

최근 대부분의 모델에서는 He 초기화

과적합 문제
정규화 (Regularization)
: 기존 손실함수에 규제항을 더해 최적값 찾기 가능

L1 정규화(Lasso Regularization)
: 가중치의 절댓값의 합을 규제 항으로 정의. 작은 가중치들이 거의 0으로 수렴하여 몇개의 중요한 가중치들만 남음

L2 정규화(Ridge Regularization)
: 가중치의 제곱의 합을 규제항으로 정의.  
L1 정규화에 비하여 0으로 수렴하는 가중치가 적음. 큰 값을 가진 가중치를 더욱 제약하는 효과

드롭아웃(Dropout)
: 각 Layer마다 일정 비율의 뉴런을 임의로 drop시켜 나머지 뉴런들만 학습하는 방법
드롭아웃을 적용하면 학습되는 노드와 가중치들이 매번 달라짐



Fashion-MNIST 데이터 분류하기
from __future__ import absolute_import, division, print_function, unicode_literals
 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import elice_utils
elice_utils = elice_utils.EliceUtils()
 
np.random.seed(100)
tf.random.set_seed(100)
 
'''
1. 다층 퍼셉트론 분류 모델을 만들고, 학습 방법을 설정해
   학습시킨 모델을 반환하는 MLP 함수를 구현하세요.
    
   Step01. 다층 퍼셉트론 분류 모델을 생성합니다.
           여러 층의 레이어를 쌓아 모델을 구성해보세요.
            
   Step02. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.
    
   Step03. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
'''
 
def MLP(x_train, y_train):
     
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
     
    # softmax 에서는 loss function = crossentropy를 쓴다
    model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     
    model.fit(x_train, y_train, epochs=10)
     
    return model
 
def main():
     
    x_train = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)
     
    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0
     
    model = MLP(x_train,y_train)
     
    # 학습한 모델을 test 데이터를 활용하여 평가합니다.
    loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
     
    print('\nTEST 정확도 :', test_acc)
     
    # 임의의 3가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    rand_n = np.random.randint(100, size=3)
     
    for i in rand_n:
        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        plt.savefig("test.png")
        elice_utils.send_image("test.png")
         
        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))
 
if __name__ == "__main__":
    main()



PyTroch 기반 Deep Learning 중급에서는
□ 과정 목표
- PyTorch 환경에서 모델 개발 및 최적화를 위한 주요 방법론 이해
- CNN의 개념 및 구조 분석을 통해 모델 설계, 동작 및 구현 방법 학습
- RNN의 개념 및 구조 분석을 통해 모델 설계, 동작 및 구현 방법 학습
- 생성모델 개념 및 GAN 구조 분석을 통해 설계, 동작 및 구현 방법 학습


구 분

내 용

1일차

 - 딥러닝 핵심 개념 정리 

 - 시뮬레이션을 통한 딥러닝 동작 및 용어 이해   

 - PyTorch 모델 설계 방법 및 PyTorch API 정리   

 - Keras / PyTorch 비교  

 - 모델 설계 실습

2일차

 - CNN 모델 개요, 구조 및 특징

 - AlexNet, VGGNet 구조 및 동작

 - CNN 모델 설계 실습 및 Model Optimization 실습

3일차

 - GoogLeNet과 Residual Neural Net(ResNet) 구조 및 구현

 - Weakly Supervised Learning 개념과 Class Activation Map 이해

 - Class Activation Map 실습 

 - Transfer Learning 이해 및 실습

4일차

 - Recurrent Neural Network 개요

 - RNN Cell과 RNN Model 구조 및 동작

 - RNN 실습(Time Series, Char-RNN, MNIST-RNN)

 - Mixture Density Network과 RNN 결합 모델

5일차

 - Generative Model 개요

 - AutoEncoder 구조 및 실습

 - Variational AutoEncoder 구조 및 실습

 - GAN(Generative Adversarial Network) Model 구조 및 동작

 - Conditional GAN, AC-GAN, InfoGAN 구조 및 동작

 - GAN 모델 실습
