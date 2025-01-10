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
ㄴ 구현할 수 있는 방법 : Depends 사용 / Middleware 사용



* Github Actions - Github이 제공하는 빌드/테스트/배포 자동화 도구<br>

