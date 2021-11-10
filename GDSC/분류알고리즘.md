### 저희는 앞장에서 선형회귀 알고리즘을 사용해 보았습니다. ( 다시 생각해보기 )

저희는 앞에서 이미 많은 분류 알고리즘을 알아보았습니다.

- KNN 알고리즘 ( 최근접 이웃 알고리즘 ) , 트리 알고리즘

여기서는

- 로지스틱 회귀

이름은 회귀라고 되어 있지만 결국 분류 모델이다.

이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 학습합니다.

우리는 앞 장에서 여러 개의 특성 값을 준비하여 다중 회귀를 함으로 좋은 선형 방정식을 찾는 것을 해보았습니다.

여기서는 그렇게 나온 선형 방정식을 '**확률**'로 표현해 보는 작업을 해보겠습니다.

확률이라는 것은 결국 0~1 사이의 값으로 표현되는 것을 의미합니다. 근데 저희가 배운 선형 방정식에서는 z는 어떤 값이든 나올 수 있죠?

### 시그모이드 함수 = 로지스틱 회귀 —> 하나의 선형 방정식

 0-1 사이의 값을 표현하기 위해 저희는 '시그모이드 함수'를 이용하여 학습된 선형 방정식에서 나온 값을 시그모이드 함수를 통과시켜 0-1 사이의 값으로 표현해보겠습니다.

```python
#시그모이드(로지스틱함수)
#어떤 실수가 들어와도 0~1 사이의 범위를 벗어날 수 없는 함수
#즉, 확률로 해석할 수 있다
import matplotlib.pyplot as plt
z = np.arange(-5,5,0.1)
phi = 1 / ( 1 + np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```

## 로지스틱 회귀로 이진 분류를 수행해 보겠습니다.

```python
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head() # 7개의 생선에 대한 데이터들이다
```

판다스를 사용해 CSV 파일을 데이터프레임으로 변환
fish.head() 를 통해서 나온 결과 다양한 특성 값이 준비되어 있다.

```python
#여기서 species 열을 타깃으로 만들고 나머지는 입력데이터로 만들겠다
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])

fish_target = fish['Species'].to_numpy()
print(fish['Species'].unique()) #7개의 생선 종류 확인
```

```python
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = train_test_split(fish_input,fish_target,random_state=42)
```

train_test_split으로 훈련 데이터와 타깃 데이터를 나누어주고

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

앞 장에서 배운 정규화를 시행했다. 기억이 나지 않는 다면 앞 장을 다시 봐보세요

특성의 스케일을 정규화하는 작업. 

ex) 특성 : 무게, 키

두개의 스케일은 무게 : (50-80) 키: (140-190) 값의 범위 차이가 심해서 예측할 때 정확한 예측이 힘들어 진다.

```python
#도미와 빙어 행만 골라내는 것
bream_smelt_indexes = (train_target == 'Bream') | (train_target =='Smelt')
print(bream_semlt_indexes) #확인해보세요
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

로지스틱 회귀를 해보겠습니다.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
```

predict_proba는 예측 확률을 나타내는 메소드이다.

```python
print('훈련된 로지스틱 회귀알고리즘으로 5번째까지 예측')
print(lr.predict(train_bream_smelt[:5]))

print('예측확률 나타내기')
print(lr.predict_proba(train_bream_smelt[:5])
```

coef_는 가중치 intercept_는 바이어스 값이다.

```python
print('가중치랑 바이어스 값 확인하기')
print(lr.coef_,lr.intercept_)
```

따라서 이 로지스틱 회귀 모델이 학습한 방정식은

z = - 0.404 x (weight) - 0.576 x ( length ) - 0.662 x ( diagonal ) - 1.013 x ( height ) - 0.732 x ( width ) -2.16

## 다중 분류를 수행해 보겠습니다. = 소프트맥스

ex ) 생선이 7개가 있으면 7개에 대한 선형방정식이 생성됩니다.

만약에 길이 : 30 높이 : 20 너비 : 10 대각선길이 : 20 —> 무슨 생선일까요?

7개의 선형방정식을 통과하면서

그 전에 이진 분류를 하는 데에는 저희가 시그모이드 함수를 사용했습니다.

하지만 다중 분류에서는 **소프트맥스 함수**를 사용합니다. 왜 소프트맥스 함수를 사용하냐면 이진 분류와 다르게 우리는 여러개의 예측값을 가지고 확률을 계산하기 때문입니다. 

### 소프트맥스 함수
여기서 z1 ~ z7은 우리가 사용하는 생선 7개를 선형방정식에 통과한 예측값을 말한다.

생선 7개에 대한 예측확률 

식을 보면 우리가 흔히 사용하는 확률를 구하는 것과 같다.

왜? e^x 를 사용하여 확률을 구하나? 소프트 맥스함수는 시그모이드 함수로 부터 유도되었기 때문이다.

자세한 공식 유도 방식 → [https://gooopy.tistory.com/53](https://gooopy.tistory.com/53)

로지스틱 회귀도 선형 방정식을 사용하여 최선의 선형을 찾는 반복적인 작업이기 때문에 과도하게 학습하여 오버피팅이 일어날 수 있기 때문에, 규제를 해야한다.

로지스틱 회귀는 기본적으로 릿지 규제를 사용한다. —> 릿지규제가 기억이 나지 않는다면 앞장에...

기본적으로 계수의 제곱을 규제하는 것

```python
lr = LogisticRegression(C=20,max_iter =1000) #여기서 C값이 높아질 수록 규제 완화
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
print(lr.predict("테스트 세트의 처음 5개의 대한 예측 : "test_scaled[:5]))
```

이번에는 테스트 세트의 처음 5개 샘플에 대한 예측 확률 출력

```python
print('확률계산')
proba = lr.predict_proba(test_scaled[:5])
print(lr.classes_) #클래스 정보 확인
print(np.round(proba,decimals=3)) #클래스별 확률값 보기
```

선형 방정식의 모습도 확인해 봅시다

coef_(기울기)와 intercept_(절편)의 크기

```python
print(lr.coef_.shape,lr.intercept_.shape) # 가중치랑 바이어스 형태확인
```

### 소프트맥스 함수를 사용해서 구해보자!

```python
#이 데이터는 5개의 특성을 사용하므로 coef_ 배열의 열은 5개
#행이 7개라는 것은 z를 7개나 계산햇다는 의미
#다중 분류는 클래스마다 z값을 하나씩 계산한다.
#당연히 가장높은 z값을 선택
#이진 분류에서는 시그모이드 함수를 사용해 z를 0과 1사이의 값으로 변환
#다중 분류는 이와 달리 소프트맥스 함수를 이용해 7개의 확률로 변환한다.
#즉, 이진분류면 시그모이드 , 세개 이상이면 소프트 맥스

print('decision_function으로 확률로 바꾸어보았다')
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2)) # 여기를 한번 봐주세요!

from scipy.special import softmax
proba = softmax(decision,axis=1)
print('사이파이에서도 소프트맥스가 있다')
print(np.round(proba,decimals=3))
```
