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

