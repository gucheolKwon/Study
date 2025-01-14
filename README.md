# Study

Andrew Ng - Deep Learning specialization (https://developers.google.com/machine-learning/glossary?hl=ko#weight)

fastcampus 강의<br>
1 -  파이썬으로 할 수 있는 모든것 with 47개 프로젝트 초격차 패키지 Online

* 알고리즘 : 어떤 문제를 해결하거나 계산 시 수행되는 명확한 명령문의 나열<br>
  ㄴ정해진 초기값과 데이터를 시작으로 유명한 시행 횟수를 거쳐 출력값을 생성

알고리즘의 특징<br>
ㄴ 입력과 출력 : 입력값을 받고, 결과를 출력. 입력 데이터<br>
ㄴ 유한성 : 유한한 단계로 실행됨. 무한루프에 빠져선 안됨<br>
ㄴ 정확성 : 입력값에 대한 정확한 결과. 원하는 목표의 달성<br>
ㄴ 효율성 : 가능한 빠르고, 공간 효율적<br><br>
알고리즘 공부해야하는 이유<br>
ㄴ 문제 해결 능력 / 효율적인 프로그램이 / 문제 예측과 예방 / 프로그래밍 언어 이해 / 면접과 채용<br><br>
알고리즘의 복잡도<br>
* 시간복잡도 : 특정 알고리즘이 실행되는 데에 필요한 시간의 양<br>
* 공간복잡도 : 알고리즘, 프로그램을 수행하는데 필요한 메모리 공간의 총량<br>
ㄴ big-O 표기법으로 표현됨 (최악의 경우로 측정) - n은 입력의 크기<br><br>
복잡도 - O(n)<br>
ㄴ 입력 크기에 비례하여 선형적으로 시간/공간 복잡도가 증가<br>
ㄴ 최악의 경우, 입력된 데이터를 모두 순회해야 함<br>
ㄴ 일반적인 배열에서 데이터 찾기<br>
ㄴ 동일한 배열 복사하기<br><br>
복잡도 - O(n^2)<br>
ㄴ 입력 크기의 제곱에 비례하여 시간/공간 복잡도가 증가<br>
ㄴ 데이터셋이 커질수록 불리해짐<br>
ㄴ 중첩된 반복문을 사용하여 정렬하는 경우<br><br>
복잡도 - O(logn)<br>
ㄴ 입력 크기에 따라 로그 형태로 시간/공간 복잡도가 증가<br>
ㄴ 큰 데이터셋에서 빠른 처리가 필요한 경우 선호됨<br>
ㄴ 이진 트리 탐색 등<br><br>

* 정렬 알고리즘 - 주어진 목록을 정해진 순서대로 정렬 - https://visualgo.net/en/sorting<br>
ㄴ 버블(Bubble) 정렬 / 삽입(Insertion) 정렬 / 퀵(Quick) 정렬 / 합병(Merge) 정렬<br>

1. 버블정렬<br>
: 가장 간단하지만 가장 비효율적<br>
ㄴ 첫 번째 원소부터 시작해서 인접한 두 원소를 비교 - 앞의 원소가 뒤의 원소보다 크면 순서를 변경 - 반복 - 가장 큰 원소가 제일 오른쪽에 위치<br>
```python
def bubble_sort(arr):
  n = len(arr)
  for i in range(n):
    for j in range(0, n-i-1):
      if arr[j] > arr[j+1]:
        arr[j], arr[j+1] = arr[j+1], arr[j]
```
2. 삽입정렬<br>
손에 들고 있는 카드를 정렬하는 것과 유사한 방식<br>
ㄴ 두 번째 원소부터 키(key)로 지정 - 키의 앞 리스트에서 해당 키가 삽입될 위치를 탐색 - 키보다 큰 원소들을 한 칸씩 밀어내고 해당 위치에 키를 삽입 - 반복 <br>
```python
def insertion_sort(arr):
  for i in range(1, len(arr)):
    key = arr[i]
    print(key)
    j = i - 1
    while j >= 0 and key < arr[j]:
      arr[j + 1] = arr[j]
      j -= 1
      print(arr)
    arr[j + 1] = key
```
3. 퀵정렬<br>
: 분할 - 정복 (divide & conquor) 알고리즘<br>
ㄴ 배열을 둘로 나눌 기준(피벗)을 결정 - 피벗보다 작은 값들을 왼쪽 배열에 큰 값을 오른쪽 배열에 저장 - 반복 - 왼쪽 배열 + 피벗 + 오른쪽 배열<br>
```python
def quick_sort(arr):
  if len(arr) <= 1: # 원소가 하나면 즉시 반환
    return arr
  pivot = random.choice(arr) # 무작위로 피벗 선택 
  less = []
  equal = []
  greater = []

  for element in arr:
    if element < pivot:
      less.append(element)
    elif element == pivot:
      equal.append(element)
    else:
      greater.append(element)
  return quick_sort(less) + equal + quick_sort(greater)
```

대표적인 알고리즘<br>
* 검색 알고리즘<br>
1) 선형 탐색 (Linear Search) / 2) 이진 검색 (Binary search) / 3) 해시 테이블 (Hash Table) <br>
* 그 외의 알고리즘 <br>
1) Brute Force (무차별 대입)<br>
ㄴ 문제의 모든 가능한 해를 생성하고, 계속해서 대입하여 해를 찾음<br>
ㄴ 단순하고 직괁거인 방법 / 문제의 범위가 크거나 간으한 해의 종류가 많을 수록 효율이 떨어짐 == 모든 경우의 수를 고려<br>
2) Divide and Conquer (분할-정복)<br>
ㄴ 큰 문제를 작은 하위 문제로 분할하고, 작은 문제를 해결한 후 결합하여 큰 문제를 해결<br>
ㄴ 분할 : 큰 문제를 더 이상 분할할 수 없을 때 까지 재귀적으로 분할 / 정복 : 분할된 작은 문제를 해결 / 결합 : 해결된 하위 문제들을 원래 문제에 대한 결과로 조합<br>
3) Back Tracking (백트래킹)<br>
ㄴ 주어진 문제를 임의의 방법으로 해결하다가 실패하면 이전 상태로 돌아가며 다른 방법으로 해결 <br>
ㄴ 답이 될 수 없는 상황을 정의하고, 답이 될 수 없으면 끝까지 수행하지 않고 즉시 종료<br>
4) Dynamic Programming (동적 계획법)<br>
ㄴ 복잡한 문제를 해결하기 위한 문제해결 방법으로, 큰 문제를 여러 개의 작은 문제로 나누어서 해결<br>
ㄴ 앞서 해결한 해들을 재사용 / 메모이제이션(Memoization) : 동잃한 계산을 반복해야 할 때, 이전에 계산한 값을 저장하여 재사용<br>
6) Recursion (재귀)<br>
ㄴ 함수가 자기 자신을 호출하여 문제를 해결하는 알고리즘<br>
ㄴ 재귀함수는 반드시 종료조건이 있어야 함 / 문제를 더 작은 단위로 나누고, 결과를 결합하여 문제를 해결 / 피보나치 수열 구하기 : F(n) = F(n-1) + F(n-2)<br>

* 알고리즘 트레이닝 사이트<br>
백준 / 프로그래머스 / Codility / LeetCode<br>

예시-------------------------------------------------------------------------------------
```python
import random

def quick_sort(arr):
  if len(arr) <= 1:
    return arr
  pivot = random.choice(arr)
  less = []
  greater = []
  equal = []

  for item in arr:
    if item > pivot:
      greater.append(item)
    elif item < pivot:
      less.append(item)
    else:
      equal.append(item)
  return quick_sort(less) + equal + quick_sort(greater)
```

객체지향 프로그래밍 <br>
- 객체는 속성(필드, 데이터)과 기능(프로시저, 메소드)으로 구성<br>
- 현실 세계의 개념, 사물을 프로그램에서 모델링 한 것 또는 추상적인 개념 / 객체는 변수화할 수 있음 -> 인스턴스<br>

- 클래스 : 객체의 초기상태값과 기능에 대한 구현을 제공 / 객체를 만들기 위한 설계도<br>
- ㄴ self는 자기 자신을 의미하는 예약어<br>
- 멤버 변수 : 객체가 가지는 속성 또는 값 / 멤버 변수로의 접근 : Obj.variable / 인스턴스 내부에서 접근할 때 : self.variable<br>
- ㄴ 멤버 변수의 초기화는 생성자를 통한다. == 생성 시 넘겨받은 파라미터를 활용하여 초기화 ex) self.name = name<br>
- ㄴ 생성자(Constructor) : 클래스를 통해 인스턴스를 생성할 때, 반드시 실행되는 메소드 / __init__() 이라는 이름으로 예약되어 있음<br>

객체지향 4대 주요 개념<br>
1) 상속 (Inheritance)<br>
- 'super()' 함수를 사용하여 부모 클래스의 메소드 사용 가능<br>
- python에서는 다중 상속도 가능
2) 추상화 (Abstraction)<br>
- 복잡한 구조를 모델링을 통해 필수 동작들로 단순화 하는 과정 / 다양한 객체들간의 공통점을 찾고, 이들을 포괄하는 상위 추상 클래스 정의
- 추상 클래스를 위한 표준 라이브러리 : abc / Abstract Base Class / abc 모듈을 통해 추상 클래스 및 추상 메소드를 정의 / ABC 클래스 또는 @abstractmethod로 정의
- ```python
from abc import ABC, abstractmethod

class Shape(ABC):
  @abstractmethod
  def get_size(self):
    pass
class Rectangle(Shape):
  def __init__(self, height, width):
    self.height = height
    self.width = width
  def get_size(self):
    return self.height * self.width

class Circle(Shape):
  def __init__(self, radius):
    self.radius = radius
  def get_size(self):
    return 3.14 * self.radius
```
3) 캡슐화 (Encapsulation)<br>
- 객체 외부에서 접근 가능한 기능(public) / 내부에서만 접근 가능한 기능 (private)<br>
- 접근 제어자 - public (공개) / protected (보호, 객체 및 상속받은 객체에서 접근 가능, '_'를 붙임) / private (비공개, 객체 내에서만 접근 가능, '__'를 붙임)<br>
4) 다형성 (Polymorphism)<br>
- 다양한 클래스들이 동일한 이름의 메소드를 각자의 목적에 맞게 사용<br>
- 메소드 오버라이딩<br>
예시-------------------------------------------------------------------------------------
```python
# class 선언된 객체에다가 .mro()를 입력하면 어떤 메서드 순으로 호출하는지 확인할 수 있음
```
예외처리의 구조<br>
ㄴ try - except - else - finally 구조<br>
ㄴ try : 예외가 일어날 수 있는 코드 / except : 예외가 일어나면 실행될 코드 / else : 예외가 없을 때 실행될 코드 / finally : 예외 발생 여부와 무관하게 마지막에 실행될 코드<br>
```python
try:
  value = int("abc")
except ValueError:
  print("숫자로 변환할 수 없는 문자열입니다.")
except ZeroDivisionError:
  print("0으로 나눌 수 없습니다.")
else:
  print("변환이 완료됐습니다")
finally:
  print("프로그램을 종료합니다.")
```

------- 초격차 패키지: 50개 프로젝트로 완벽하게 끝내는 머신러닝 SIGNATURE<br>
* 머신러닝으로 어떤문제를 푸는가?<br>
1) Forecast<br>
: 주로 Time series data / 변수들간의 인과관계를 분석하여 미래를 예측 / 대표적 알고리즘은 AR(I)MA, DeepAR 등이 상용 수준으로 쓰임<br>
2) Recommendation<br>
: Netflix Prize를 통해 추천 연구 분야가 널리 알려짐. / Collaborative Filtering / Contetnt based Filtering<br>
ㄴ 현실 뎅이터는 희소성 문제가 커서 알고리즘 적용이 어렵다 >> Matrix Factorization & Factorization Machine이 대표적으로 시도되는 방법<br>
3) Anomaly Detection<br>
: Normal을 벗어나는 데이터를 찾는 문제. 단순히 Outlier detection / Out-of-Distribution / One Class Classification 등 다양한 방법으로 접근<br>
4) Image Processing <br>
: Main task - Classification / Localization / Object Detection / Instance Segmentation <br>
ㄴ 연구는 독립적으로 이루어지지만, 현업에서는 sub task와 혼합하여 사용<br>
ㄴ 양품/불량품 자동 판정 모델 이용 / OCR(Optical Character Recognition)을 활용하여 아날로그 -> 디지털 전환에 편리성 제공<br>
5) NLP <br>
: 컴퓨터가 인간의 언어를 처리하는 모든 기술을 의미 <br>
ㄴ 대표 Task : 감성 분석, 대화생성(챗봇), STT(speech to text)<br>
ㄴ 제품 리뷰의 Negative 비율을 관리하여 상품 평판 관리 / CS 업무 중 반복적인 질문 등에 대한 자동 응대 / 다양한 미디어 매체의 데이터에서 부정적 의견을 모니터링해서 회사에 대한 평판관리


virtual metrology 모델을 통해 샘플링 되지 않은 웨이퍼 칩의 계측 값을 예측<br>
ㄴ 센서에서 수집되는 데이터 : 설명변수(x) / 실제 계측 값 : 반응변수(y)<br>

현업 문제해결 유형별 머신러닝 알고리즘<br>
1) eXplainable Method<br>
- Black box : 내부 구조나 작동 원리를 몰라도 입력과 출력을 할 수 있는 장치나 회로 또는 과정<br>
ㄴ eXplainable Artificial Intelligence (xAI)<br>
ㄴ (1) Grad-CAM : Visual Explanations from Deep Networks via Gradient-based Localization<br>
ㄴ (2) Linear Regression / Logistic Regression / Generalizaed Linear Models / Generalized Additive Models / Decision Tree, Naive Bayes Classifier, K-Nearest Neighbors / Local Model-Agnostic Methods (LIME, SHAP)<br>

- Linear Regression : 오차의 선형성, 독립성, 등분산성, 정규성의 가정을 가짐<br>
ㄴ 가정1 : 종속변수와 독립변수들 간의 선형성이 보장되어야 함 / 가정2 : 독립변수들 사이에 선형관계 없음 (Correlation) / 가정3 : 독립변수는 오차항과 상관이 없음 / 가정4 : 오차항들은 서로 독립적이며 서로 연관되어 있지 않음 / 가정5 : 오차항의 평균은 0 / 가정6 : 오차항의 분산은 일정<br>
== "BLUE : Best Linear Unbiased Estimator"<br>
- Generalized Linear Models : 기존의 Linear Regression에서 종속변수의 분포를 정규 분포를 포함한 여러 분포로 확장<br>
ㄴ 기존 종속변수 평균과 독립변수의 선형 관계를 종속변수 평균의 함수와 독립변수의 선형관계로 확장한 모형<br>
- Generalized Additive Models : X 각각에 비선형 함수 f를 적합할 수 있기 때문에 Linear Regression으로 학습하지 못하는 비선형 관계를 학습할 수 있음<br>
- Naive Bayes Classifier <br>

Global vs Local Feature Importance Score <br>
* 기존 분석의 중요인자 추출 기법은 전체 데이터를 대변하는 중요 인자를 추출함<br>
* 데이터 분석의 목적에서 특정 Y(Local, 고효율) 군을 결정 짓는 중요 인자를 추출할 수 있음<br>
* 더 나아가 0.01%(불량 등) 데이터를 결정짓는 중요 인자를 추출<br>
* 전체 Y를 대변하는 중요 인자(global Interpretability)와 특정 데이터에 대한 중요 인자 (Local Interpretability)는 다른 결과를 불러올 수 있음<br>
* 특정 Y(내가 원하는 데이터)에 대한 해석력을 얻기 위해서는 Black Box Model을 열어봐야 함<br>
* Interpretable Machine Learning(IML)을 통해 Black Box Model을 열어봄<br>

Local Model-Agnostic Methods : LIME, SHAP<br>

1-1) LIME <br>
* Local Interpretable Model-agnostic Explanations --> Model agnostic : 어떠한 모델이라도 적용 가능<br>
* 데이터를 바꿔보면서 (noise를 주면서) 정확도를 측정해보는 것.<br>
- Step 1. Modeling<br>
- 평상시 우리가 돌리는 Model을 사용하여 학습을 진행 - Weight based Model (Regression, Logistic Regression...), Tree based Model (Random Forest, Gradient Boosting Tree, Xgboost,...), Deep Learning(CNNs, RNNs, ... ) 모든 모델을 사용해도 됨 <br>
- LIME은 Model에 대한 Scalability를 보장<br>
- Step 2. Dataset and Predictions<br>
- Step 3. Picking Step - 확실한 패턴이 있는 데이터를 추출하기 위하여 학습이 잘된 데이터 (MSE가 낮은)를 추출<br>
- Step 4. Explanations - 도출할 때는 LIME 사용해서 도출<br>
- Step 5. Human makes decision<br>

*  Locally approximation 과정 관련 그림 추가할 것. <br>

1-2) SHAP <br>
* Shapley Additive exPlanations / LIME 개념 + 경제학 개념 (노벨 경제학상을 받은 Shapley Values(게임이론 : 여러 주제가 서로 영향을 미치는 상황에서 서로 어떤 의사결정이나 행동을 하는지에 대한 이론)을 접목시킴<br>
* Core는 효율성이라는 바람직한 특성을 가지고 있으면서 바람직하지 못한 특성도 보유<br>
* (분배에 대한) "유일한" 해를 제공하지 못한다는 단점 -> 다른 해들이 많이 개발돼서 가장 잘 알려진 것이 SV(Shapley Value)<br>

1-2-1) SHAP 실습<br>


** 랜덤 포레스트에서 변수의 중요도가 높다면, <br>
- Random permutation 전-후의 OOB Error 차이가 크게 나타나야 하며, 그 차이의 편차가 적어야함<br>


------------------------------------------------------------------------------------
웹크롤링- fastcampus 강의 - 파이썬으로 할 수 있는 모든것 with 47개 프로젝트 초격차 패키지 Online <br>
* 크롤링 할때 주의사항<br>
  1) robots.txt를 확인해 허용 범위 확인하기<br>
* HTTP와 웹 구동방식<br>
  1) HTTP(Hypertext Transfer Protocol) : 서버와 클라이언트가 인터넷상에서 데이터를 주고받기 위한 프로토콜<br>
     ㄴ HTTP 메시지 = ex) 택배 송장 / 요청 메소드의 종류 : SELECT(GET 정보를 요청), INSERT (POST 정보를 입력), UPDATE  (PUT 정보를 업데이트), DELETE (DELETE 정보를 삭제)<br>
  2) URL (Uniform Resource Locator) : HTTP와는 독립된 체계로 자원의 위치를 알려주기 위한 프로토콜<br>
* HTML과 태그<br>
  1) HTML (Hypertext Markup Language) : 웹페이지를 작성하기 위한 문법들 중 하나. 99% 이상 사용됨<br>
     ㄴ tag 종류들이 어떤게 있는지 확인할 수 있는 사이트 - https://www.w3schools.com/tags/ref_byfunc.asp<br>
* 개발자 도구 : 웹브라우저에서 개발자를 위해 지원하는 편의 기능<br>
* ID : 대체로 한번만 사용 / 주로 HTML 객체를 고유하게 찾을 용도로 사용<br>
* Class : 여러번 사용 / CSS(디자인 서식)을 적용하는 용도로 사용<br>
------------------------------------------------------------------------------------
웹 개발 프로젝트 - Fastapi를 활용해 이미지 저장 및 서빙하기<br>
* Annotation Based API server : Python 3.6+ 부터 지원하며, 그 이유는 Type Annotation<br>
* 내부에서 Starlette과 Pydantic을 사용 / 자동 스웨거를 지원<br>
* 서버를 만들기 위해 사용할 패키지 : SQLAlchemy, PyMySQL, Uvicorn, uvloop, Gunicorn, Pydantic, Starlette<br>
* Pydantic : 파이썬이 제공하는 타입 어노테이션을 이용한 Data 검증하는 시리얼라이저<br>
```python
# dummy code
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index():
  return {"index": "Hello FastAPI"}

@app.get("/math/sum")
def math_sum(number_1: int, number_2: int):
  return {"result": number_1 + number_2}
```
* Uvicorn : 비동기 웹서버를 위한 라이브러리.
* 강의 github : https://github.com/riseryan89/imizi-api.git
* FastAPI DB 핸들링 : MySQL + DBeaver 설치해서 사용.
* SQLAlchemy : 많은 회사들이 사용하고 있는 라이브러리 ex. Yelp!, reddit, DropBox
* 왜 ORM ? -> SQL 없이 OOP 언어로 DB 조작이 가능 / 비즈니스 로직에 집중할 수 있음 / 코드 재사용 및 유지보수 용이 / DBMS에 민감하지 않음.
```python
# 공식 예제 코드
from sqlalchemy import text

sql = text('SELECT * from BOOKS WHERE BOOKS.book_price > 100')
results = engine.execute(sql)

result = engine.execute(sql).fetchall()

for record in result:
  print("\n", record)

# SQLAlchemy 테이블 객체 정의
class Users(Base):
  __tablename__ = "users"
  email = Column(String(64), nullable=False)
  pw = Column(String(256), nullable=False)
  api_keys = relationship("APIKeys", back_populates="users")

class APIKeys(Base):
  __tablename__= "users_api_keys"
  user_id = Column(ForeignKey("users.id"), nullable=False)
  access_key = Column(String(64), nullable=False)
  secret_key = Column(String(64), nullable=False)
  whitelist_ips = Column(String(256), nullable=False)
  users = relationship("Users", back_populates="api_keys")
```
* SQLAlchemy - String / Integer / BigInteger / Boolean / Date / Datetime / Text / Json<br>
* DB - varchar / integer, int / bigint / boolean / date / timestamp, datetime / text / json<br>
* DB 모델링<br>
ㄴ 

* FastAPI 인증 미들웨어 만들기<br>
ㄴ Middleware 하는 일 : API 레벨의 함수가 실행되기 전에 수행 / 데이터 전처리 / 데이터 정합성 검증 / 인증 정보 검증<br>
ㄴ 구현할 수 있는 방법 : Depends 사용 / Middleware 사용<br>

* Github Actions - Github이 제공하는 빌드/테스트/배포 자동화 도구<br>


------------------------------------------------------------------------------------
올인원 패키지 : 머신러닝 서비스 구축을 위한 실전 MLOps + https://mlops-for-all.github.io/docs/introduction/levels<br> 
* 학습 환경과 배포 환경은 같지 않다!<br>
* 학습된 머신러닝 모델이 동작하기 위해서 필요한 3가지 : 파이썬 코드 / 학습된 가중치 / 환경 (패키지, 버전 등)<br>
* 버전 관리 - 테스트 자동화 - 모니터링<br>
* MLOPs의 구성 요소 : 데이터 / 모델 / 서빙<br>
  1-1) 데이터 수집 파이프라인 : Sqoop, Flume, Kafka, Flink, Spark Streaming, Airflow<br>
  1-2) 데이터 저장 : MySQL, Hadoop, Amazon S3, MinIO<br>
  1-3) 데이터 관리 : TFDV, DVC, Feast, Amundsen<br>
  2-1) 모델 개발 : Jupyter Hub, Docker, Kubeflow, Optuna, Ray, katib<br>
  2-2) 모델 버전 관리 : Git, MLFlow, Github Action, Jenkins<br>
  2-3) 모델 학습 스케줄링 관리 : Grafana, Kubernetes<br>
  3-1) 모델 패키징 : Docker, Flask, FastAPI, BentoML, Kubeflow, TFServing, seldon-core<br>
  3-2) 서빙 모니터링 : Prometheus, Grafana, Thanos<br>
  3-3) 파이프라인 매니징 : Kubeflow, argo workflows, Airflow<br>
--> AWS SageMaker / GCP Vertex AI / Azure Machine Learning가 상용 프로그램<br>

* MLOps의 기본 : ML 이론 / SW 구현 능력 / 클라우드 지식 / 협업 능력<br>
* Reproducibililty (실행 환경의 일관성 & 독립성) / Job Scheduling (스케줄 관리, 병렬 작업 관리, 유휴 자원 관리) / Auto-healing & Auto-scaling (장애 대응, 트래픽 대응)<br>
* Docker 실습 <br>
ㄴ Virtualbox 로 VM 만들고 (Ubuntu 20.04.3) 환경에서 시작 <br>
ㄴ Tip) 어떤 오픈소스를 사용하더라도 공식 문서를 참고하는 습관을 들이는 것 추천<br>
* 1) Set up the repository : apt 라는 패키지 매니저 업데이트 (sudo apt-get update) / docker의 prerequisite package 들을 설치 ( sudo apt-get install \ apt-transport-https \ ca-certicficates \ curl \ gnupg \ lsb-release <br>
ㄴ docker의 GPG key 추가 (curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor - o /usr/share/keyrings/docker-archive-keyring.gpg <br>
ㄴ stable 버전의 repository를 바라보도록 설정 ( echo \ "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg]https://download.docker/com/linux/ubuntu \ $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null <br>
* 2) Install Docker Engine (sudo apt-get update) (sudo apt-get install docker-ce docker-ce-cli containerd.io) <br>
* 3) 정상 설치 확인 (sudo docker run hello-world)<br>
* 4) Docker 권한 설정 (sudo usermod -a -G docker $USER) (sudo service docker restart) <br>

* Docker 기본적인 명령<br>
  1) Docker pull : docker image repository 부터 Docker image를 가져오는 커맨드 (docker pull --help) <br>
  2) Docker images : 로컬에 존재하는 docker image 리스트를 출력하는 커맨드 (docker images --help) <br>
  3) Docker ps : 현재 실행중인 도커 컨테이너 리스트를 출력하는 커맨드 (docker ps --help) <br>
  4) Docker run : 도커 컨테이너를 실행시키는 커맨드 (docker run --help) <br>
  ㄴ docker run it --name demo1 ubuntu:18.04 /bin/bash (-it : container를 실행시킴과 동시에 interactive한 terminal로 접속시켜주는 옵션, --name : 컨테이너 id 대신 구분하기 쉽도록 지정해주는 이름, /bin/bash : 컨테이너를 실행시킴과 동시에 실행할 커맨드로, bash 터미널을 사용하는 것 의미 <br>
  5) Docker exec : Docker 컨테이너 내부에서 명령을 내리거나, 내부로 접속하는 커맨드 (docker exec --help) <br>
  6) Docker logs : log를 확인하는 커맨드 (docker logs --help) ex) docker run --name demo3 -d busybox sh -c "while ture; do $(echo date); sleep 1; done" <br>
  7) Docker stop : 실행 중인 도커 컨테이너를 중단시키는 커맨드 (docker stop --help)<br>
  8) Docker rm : 도커 컨테이너를 삭제하는 커맨드 (docker rm --help) <br>
  9) Docker rmi : 도커 이미지를 삭제하는 커맨드 (docker rmi --help) <br>
* Docker Image : 어떤 애플리케이션에 대해서, 단순히 애플리케이션 코드뿐만이 아니라, 그 애플리케이션과 dependent한 모든 것을 함께 패키징한 데이터<br>
* Dockerfile : 사용자가 도커 이미지를 쉽게 만들 수 있도록 제공하는 템플릿<br>
  1) Dockerfile 만들기<br>
     cd $HOME #home 디렉토리로 이동<br>
     mkdir docker-practice #docker-practice 폴더 생성<br>
     cd docker-practice<br>
     touch dockerfile # Dockerfile이라는 빈 파일 생성<br>
  2) 기본 명령어<br>
     FROM : Dockerfile이 base image로 어떠한 이미지를 사용할 것인지를 명시하는 명령어 (FROM <image>[:<tag>] [AS <name>]<br>
     COPY : <src>의 파일 혹은 디렉토리를 <dest> 경로에 복사하는 명령어 (COPY <src> .... <dest>)<br>
     RUN : 명시한 커맨드를 도커 컨테이너에서 실행하는 것을 명시하는 명령어 (RUN <command>) ex) RUN pip install torch<br>
     CMD : 명시한 커맨드를 도커 컨테이너가 시작될 때, 실행하는 것을 명시하는 명령어 (CMD <command>) ex) CMD python main.py<br>
     WORKDIR : 이후 작성될 명령어를 컨테이너 내의 어떤 디렉토리에서 수행할 것인지를 명시하는 명령어 (WORKDIR /path/to/workdir)<br>
     ENV : 컨테이너 내부에서 지속적으로 사용될 environment variable의 값을 설정하는 명령어 (ENV <key> <value>)<br>
     EXPOSE : 컨테이너에서 뚫어줄 포트/프로토콜을 지정할 수 있음. protocol 지정하지 않으면 TCP가 디폴트로 설정 (EXPOSE <port>/<protocol>)<br>
* Docker build from Dockerfile (docker build --help) ex) docker build -t my-image:v1.0.0 . (.은 현재 경로에 있는 dockerfile로부터라는 의미) <br>
  #### grep : my-image가 있는지를 잡아내는 (grep) 하는 명령어 ex. docker images | grep my-image<br>
* Docker Image 저장소 - Docker Registry <br>
  ㄴ docker run -d -p 5000:5000 --name registry registry / docker ps <br>
  ㄴ my-image를 방금 생성한 registry를 바라보도록 tag - > docker tag my-image:v1.0.0 localhost:5000/my-image:v1.0.0 / docker images | grep my-image <br>
  ㄴ my-image를 registry에 push (업로드) > docker push localhost:5000/my-image:v1.0.0<br>
  ㄴ 정상적으로 push됐는지 확인 > curl -X GET http://localhost:5000/v2/_catalog<br>
* Docker Hub<br>

* Kubernetes (public cloud - Amazon EKS, Google Kubernetes Engine, Azure Kubernetes Service) <br>
ㄴ 나만의 작은 쿠버네티스 : https://github.com/kubernetes/minikube / https://github.com/ubuntu/microk8s / https://github.com/k3s-io/k3s <br>
ㄴ 쿠버네티스의 컨셉 : 선언형 인터페이스와 Desired State / Master Node & Worker Node / <br>

* Kubernetes 실습 <br>
ㄴ yaml : 데이터 직렬화에 쓰이는 포맷/양식 중 하나 / 데이터 직렬화 = 서비스간에 Data를 전송할 때 쓰이는 포맷으로 변환하는 작업. 다른 데이터 직렬화 포맷은 XML, JSON <br>
   -> kubernetes manifests 명세 / docker compose 명세 / ansible playbook 명세 / github action workflow 명세 등에 쓰임 <br>
   -> - 를 사용해서 list를 명시할 수 있음 /  [] 를 사용해도 됨<br>
ㄴ minikube를 사용하여 쿠버네티스 실습 <br>
   -> 설치 : https://minikube.sigs.k8s.io/docs/start/ 또는 https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-linux/ <br> 
   -> (커맨드는 cmd 기반 CPU) curl -LO https://storage.googleapis.com/minikube/releases/v1.22.0/minikube-linux-amd64 <br>
   -> sudo install minikube-linux-amd64 /usr/local/bin/minikube <br> / 정상 다운로드 확인 minikube --help <br>
   * Basic Commands : start / status / stop / delete<br>
   minikube start --driver=docker / 정상적으로 생성됐는지 minikube 상태 확인 : minikube status / 
ㄴ kubectl : kubernetes cluster (server)에 요청을 간편하게 보내기 위해서 널리 사용되는 client 툴<br>
  -> 설치 : curl -LO https://dl.k8s.io/release/v1.22.1/bin/linux/amd64/kubectl <br>
  -> kubectl 바이너리를 사용할 수 있도록 권한과 위치 변경 : sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl <br>
  -> 정상적으로 설치됐는지 확인 : kubectl --help / 버전 확인 : kubectl version <br>
  -> kubectl을 사용해서 minikube 내부의 default pod들이 정상적으로 생성됐는지 확인 : kubectl get pod -n kube-system<br>
ㄴ pod : 쿠버네티스에서 생성하고 관리할 수 있는 배포 가능한 가장 작은 컴퓨팅 단위 <br>
  -> Pod 단위로 스케줄링, 로드밸런싱, 스케일링 등의 관리 작업을 수행 == "쿠버네티스에 어떤 애플리케이션을 배포하고 싶다면 최소 Pod으로 구성해야 한다는 의미"<br>
  -> Pod = Container를 감싼 개념 / 하나의 Pod = 한 개의 container 또는 여러 개의 container. Pod 내부의 여러 Container는 자원을 공유함 <br>
  -> vi pod.yaml 해서 파일을 만든 다음에 kubectl apply -f pod.yaml 을 입력하면 yaml-file-path에 해당하는 kubernetes resource를 생성 또는 변경할 수 있음 <br>
  ** kubernetes resource의 desired state를 기록해놓기 위해 항상 YAML 파일을 저장하고, 버전 관리하는 것을 권장. <br>
  -> 생성한 Pod의 상태를 확인 : kubectl get pod <br>
  -> Pod 조회 : kubectl get pod을 통해 얻은 것은 Current State를 출력. <br>
     namespace : kubernetes에서 리소스를 격리하는 가상의(논리적인) 단위 / kubectl config view --minify | grep namespace: 로 current namespace가 어떤 namespace로 설정되었는지 확인할 수 있음 <br>

  -> Pod 로그 : kubectl logs <pod-name> / kubectl logs <pod-name> -f : 로그를 계속 보여줌 / kubectl logs <pod-name> -c <container-name> : 여러개의 container가 있는 경우 다음과 같이 실행 <br>
  -> Pod 내부 접속 : kubectl exec -it <pod-name> -- <명령어> / kubectl exec -it <pod-name> -c <container-name> -- <명령어> <br>
  -> Pod 삭제 : kubectl delete pod <pod-name> / kubectl delete -f <YAML-파일경로> <br>
  
ㄴ Depolyment : Pod와 Replicaset에 대한 관리를 제공하는 단위 <br>
  -> 관리 = "Self-healing, Scaling, rollout (무중단 업데이트) 와 같은 기능" 포함 <br>
  -> Deployment 는 Pod을 감싼 개념 = Pod을 Deployment로 배포함으로써 여러 개로 복제된 Pod, 여러 버전의 Pod을 안전하게 관리할 수 있음.<br>
  -> Deployment 조회 : kubectl get deployment / kubectl get deployment,pod (deployment,pod 동시에 조회) <br>
  -> Deployment Auto-healing : kubectl delete pod <pod-name> / kubectl get pod --> 기존 pod이 삭제되고, 동일한 pod이 새로 하나 생성된 것을 확인할 수 있음. <br>
  -> Deployment Scaling : kubectl scale deployment/nginx-deployment --replicas=5 (replica 갯수 늘리기) / kubectl get deployment / kubectl get pod <br>
  -> Deployment 삭제 : kubectl delete deployment <deployment-name> / kubectl get deployment / kubectl get pod <br>

ㄴ Service : 쿠버네티스에 배포한 애플리케이션을 외부에서 접근하기 쉽게 추상화한 리소스 <br>
  -> Pod은 IP를 할당받고 생성되지만, 언제든지 죽었다가 다시 살아날 수 있으며, 그 과정에서 IP는 항상 재할당받기에 고정된 IP로 원하는 Pod에 접근할 수는 없음<br>
  -> 클러스터 외부 혹은 Pod에 접근할 때는 Pod의 IP가 아닌 Service 를 통해서 접근하는 방식을 거침 / Service는 고정된 IP를 가지며, Service는 하나 혹은 여러 개의 Pod과 매칭 / 클라이언트가 Service 주소로 접근하면 실제로는 Service에 매칭된 Pod에 접속할 수 있게 됨 <br>


MEW Model : Denoising Autoencoder (Supervised-Learning)

모델 개발 Tracking : MLflow 
Tensorboard (TensorFlow의 시각화 툴킷) 실행하여 머신러닝 실험 이해하고 디버깅 및 로그 관리
Flask : web application 개발 도구, REST API 통한 on demand inference 기능을 구현하기 위함. 
mewtwo 서버에서 Docker로 flask를 실행하는 환경으로 운영됨
13 pt → 64 pt 계측 결과를 추론해내는 시스템

MLflow
mlflow server -h 0.0.0.0 -p 5000
```python
import mlflow
mlflow.set_tracking_uri("http://{server}:{port}")
mlflow.create_experiment('RF_TEST')
mlflow.set_experiment('RF_TEST')
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
 
mlflow.autolog()
 
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
 
rf = RandomForestRegressor(n_estimators=20, max_depth=6, max_features=3)
rf.fit(X_train, y_train)
 
predictions = rf.predict(X_test)
```

스케쥴러(Workflow Management) : Apache Airflow w. Celery Executor
데이터 전처리, 학습, 예측, 배포, ETL 등에 강점을 보이는 Data Workflow Management Tool
13 pt from DataLake 취득 방식 : Polling Job → Crawling Job → Run Predictor 
시스템에 사용하는 Repository 방식 : Realational Database (ex. MySQL, PostgreSQL)
User Service 형태 : WebService : Mendix

CD (배포 자동화) 설정 - Jenkins
pipeline {
    agent {
        label "project_name"
    }
    stages{
        stage("SCM Checkout"){
            steps{
                script{
                    echo '********************** START CHECKOUT **********************'
         
                    echo '** Clean Project **'
                    sh 'if ! [ -f $WORKSPACE ]; then rm -rf $WORKSPACE; fi'
                     
                    echo '** Checkout **'
                    sh 'git config --global http.sslVerify false'
                    git branch: 'master', credentialsId: 'id', url: 'git repo URL'
                     
                    echo '********************** FINSISH CHECKOUT **********************'
                }
            }
        }
        stage("Build"){
            steps{
                script{
                    echo '********************** START BUILD **********************'
                     
                    sh 'cp /appdata/config/db_config.json ./project_1/config/'
                    sh 'cp /appdata/config/db_config.json ./project_2/config/'
                    //sh 'cp /appdata/config/db_config.json ./project_3/config/'
                     
                    echo '********************** FINSISH BUILD **********************'
                }
            }
        }
        stage("deploy"){
            steps{
                script{
                    echo '********************** START DEPLOY **********************'
                     
                    // copy to target directory
                    sh 'rm -r /appdata/production/project_1/'
                    sh 'mkdir /appdata/production/project_1/'
                    sh 'cp -r $WORKSPACE/* /appdata/production/project_1/'
                    sh 'chmod -R ugo+rwx /appdata/production/project_1/'
                     
                    echo '********************** DEPLOY COMPLETE **********************'
                }
            }
        }
        stage("operate"){
            steps{
                script{
                    echo '********************** START OPERATE **********************'
                     
                    //sh 'python ~~~'
                     
                    echo '********************** OPERATE COMPLETE & END PIPELINE **********************'
                }
            }
        }
    }
}

서버 모니터링 - Grafana
mkdir prometheus  # 아무데에나 만들면 됨
cd prometheus
curl -O node_exporter-1.6.1.linux-amd64.tar.gz 설치 주소
tar -xvf node_exporter-1.6.1.linux-amd64.tar.gz
cd node_exporter-1.6.1.linux-amd64
nohup ./node_exporter > /dev/null 2>&1 &
ps -ef | grep -v grep | grep node_exporter
 
## 참고 ##
## gitlab 설치되어있는 서버에서 prometheus 같이 설치 시 충돌 발생 ##
## gitlab에서 제공하고 있는 모니터링 기능 종료 명령어 ##
sudo gitlab-ctl stop node-exporter
 
## 이후 다시 다음 명령어 실행 ##
cd ../prometheus/node_exporter-1.6.1.linux-amd64
nohup ./node_exporter > /dev/null 2>&1 &

Airflow 
: batch-oriented workflow를 개발, 예약, 모니터링하기 위한 오픈소스 플랫폼
- 구성요소
  1) schedular : worokflow를 처리하는 역할. executor로 task 보냄
  2) Executor : task를 처리하는 역할. 주로 worker에게 task execution을 push
  3) web server : DAG, task 관리 도와주는 user interface
  4) DAG (Directed Acyyclic Graph) : Python으로 작성된 workflow. task 모음이면서 task 관계 정의
  5) Database : DAG과 task의 metadata가 저장되는 DB
  6) Worker : 실제로 task를 실행하는 주체
 
- 실제 동작
  1) DAG(workflow)에 따라서 schedular가 task를 scheduling
  2) schedule에 맞게 worker가 task 처리
  3) task의 진행 상황은 database에 저장
  4) airflow UI에서 확인 가능
- DAG 설정 python code 예시
  1) import module
  2) define DAG : DAG id나 schedule 정보 등 입력
  3) define task 
  3-1) Operator : task 실행하는 방버 정의. 다양한 operator가 있음. ex) BashOperator : bash command 실행. PythonOperator : python 구문 실행
  4) task dependancy 정의
