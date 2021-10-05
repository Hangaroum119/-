# k-shape : Efficient and Accurate Clustering of Time Series

John Paparrizos, Luis Gravano 2015년도 작성



### 목차

---

- 이론적 배경
- 새로운 군집 중심 계산 방법 제시(K-shape)

   (1) 거리 측정(SBD)   
   (2) centroid 계산

- K-Shape 성능 평가
   
   
   
### 이론적 배경

---

(1) 클러스터링   
:탐색 능력 뿐만 아니라 다른 기법의 사전 처리 단계로 가장 널리 사용되는 데이터 마이닝   
:기본 데이터에서 패턴이나 상관관계를 식별하고 요약 가능   

(2)  시퀀스   
: 데이터를 순서대로 하나씩 나열하여 나타낸 데이터 구조   
:시퀀스의 각 요소에는 특정위치의 데이터를 가리키는 인덱스가 지정됨   

(3)  시계열 시퀀스   
: 일정한 시간별로 측정한 연속된 실수값의 데이터   

- 데이터 시퀀스가 타이밍(오디오,음성 등)에 대한 명시적 정보를 포함하고 있는 경우   
- 값의 순서를 유추할 수 있는 경우(스트림 및 필기)   

(4) 시계열 클러스터링의 장점   
: 비용이 많이 드는 human supervision이나 data annotation(meta data)에 의존 X   
→ 데이터 라벨링에 의존 X   

(5) 시계열 클러스터링 방법

- 기본 거리 측정값을 시계열에 더 적합한 값으로 대체
- 기존 클러스터링 알고리즘을 직접 사용할 수 있도록 시계열을 “평탄한”데이터로 변환

(6) 두 시계열 시퀀스 비교 시, distortions 처리 중요
: 진폭과 위상에 대한 불변성을 제공하는 거리 측정치가 좋은 성능을 보임
: 진폭과 위상, 도메인 차이로 인해 새로운 클러스터링 알고리즘보다 새로운 거리 측정 생성에 
  더 많은 관심이 주어짐

(7)  스케일(크기) 및 이동 불변 거리 측정을 사용한 형태(패턴) 기반 클러스터링의 단점

- 계산 비용이 많이 드는 방법이나 거리 측정값에 의존하기 때문에 데이터의 크기를 확장 불가능
- 특정 영역에 대한 접근법이 개발되었거나 제한된 데이터 세트에서만 효과가 나타남

(8) 클러스터링의 응집도(경도)
: 군집은 동질성의 개념으로 나눠지는데 그 기준은 군집 내 유사성과 다른 군집과의 거리(유사도)임
: 군집 내 데이터끼리는 가깝게, 다른 군집끼리는 멀게

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled.png)

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled%201.png)

- *추가 설명*
    
    *※ 이 목적 함수는 전역 최솟값(global minimum)을 찾는 것(=NP난해문제)
    이에 k-maens 알고리즘에서 사용하는 휴리스틱 기법(hill climbing)을 사용해
    지역 최솟값(local minimum)을 찾는다
    
    ※ NP난해문제
    비결정론적 튜링머신으로 다항시간 내에 풀 수 있는 문제
    (시간복잔도가 𝑂(𝑛^𝑘)로 표현될 수 있음)
    
    ※ 휴리스틱 기법(hill climbing)
    필요한 정보를 느슨하게 적용시켜 접근을 시도하는 전략
    ( global을 찾기 어렵기에 local로 차근차근 접근하는 방법)*
    

- (실제 사용 예시)   `※k-shape의 군집 중심 구하는 방법과 동일하다`

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled%202.png)

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled%203.png)

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled%204.png)

(9) Steiner’s sequence(슈타이너 수열)
:군집을 대표하는 중심점 계산 , 거리 측정 방법에 따라 달라짐

![Untitled](k-shape%20Efficient%20and%20Accurate%20Clustering%20of%20Time%20%207388b9211abd433c89b5a0a34f445569/Untitled%205.png)

### Time-series Invariance

: 시퀀스에 왜곡이 생기는 경우, 유사도 계산 시 적합한 값이 나오지 않을 시 있음.
이에 time series 에 대한 distance measure를 구할 때 필요한 요구 조건 존재

---

- Scaling and translation invariances
: 시퀀스에 scaling이나 translation(a𝑥 +𝑏)을 적용했을 경우

- Shift invariance
: 두 시퀀스에 위상이 다르거나(global aligment) 
  시퀀스에 정렬되지 않은 부분이 존재하는 경우(local aligment)
    
    *(예시)* 
    
    - *위상 차이 (global aligment) : 심장박동의 측정 시작이 다른 경우*
    - *정렬되지 않은 부분이 존재(local aligment)*
    
           *: 사람마다 손글씨의 크기나 단어 사이 공백에 따라 정렬 필요*
    
      *`※ 위상
      반복되는 파형의 한 주기에서 첫 시작점의 각도 혹은
      어느 한 순간의 위치를 말함`*
    

- Uniform scaling invariance
: 두 시퀀스의 길이가 다른 경우  (*예시) 측정 시간이 다른 심장박동*

- Occlusion invariance
: 시퀀스의 일부인 sub-sequence가 missing 되었을 경우

      *(예시) 오타가 있거나 문자가 누락된 경우*

- Complexity invariance
: 두 시퀀스의 복잡도가 다를 경우
*(예시) 실내와 실외의 노이즈 차이 존재*
