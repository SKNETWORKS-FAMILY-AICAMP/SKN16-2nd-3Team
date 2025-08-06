# SKN16-2nd-3Team
## 😑 퇴사하지말아조

## 프로젝트 개요
### 주제 선정 배경


<br>

### 분석 목적
본 프로젝트의 목적은 사내 인사 데이터를 바탕으로 직원의 이탈 여부를 예측하는 모델을 개발하여 이탈 가능성이 높은 직원을 사전에 파악하고, 이를 바탕으로 조직의 인적 리스크 관리의 유용성을 높이는 것이다.

<br>

### 팀 소개
|<img width="367" height="264" alt="image" src="https://github.com/user-attachments/assets/b0132a3d-51b0-4aec-aa44-a63973af1844" />|ㅎㅎㅎ|ㅎㅎ|ㅎㅎㅎ|
|:---:|:---:|:---:|:---:|
|차하경|임종민|김나은|이용채|

<br>

### 기술 스택
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>

<br>

## 데이터셋 개요

**기업 HR 데이터셋**을 사용하였으며, 해당 데이터셋에는 근무 환경에서 직원 성과, 생산성 및 인구 통계와 관련된 주요 측면을 파악할 수 있는 데이터가 포함되어 있다.

데이터 출처: [Kaggle, Employee Performance and Productivity Data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data)

<br>

### 데이터프레임 구조

**총 행 수**: 100,000

**총 열 수**: 20

**데이터 타입**: bool(1), float64(1), int64(13), object(5)

**메모리 사용량**: 14.6+ MB


| 변수명                     | Not null 개수 | 데이터 타입 | 변수 설명 |
|:--------------------------:|:---------:|:-----------:|:-----:|
| Employee_ID              | 100,000    | int64       | 각 직원의 고유 식별자 |
| Department               | 100,000    | object      | 근무 부서(예: 영업, 인사, IT)|
| Gender                   | 100,000    | object      | 성별(남성, 여성, 기타)|
| Age                      | 100,000    | int64       | 연령(22~60세)|
| Job_Title                | 100,000    | object      | 맡은 역할(예: 관리자, 분석가, 개발자)|
| Hire_Date                | 100,000    | object      | 고용된 날짜|
| Years_At_Company         | 100,000    | int64       | 회사에서 근무한 연수|
| Education_Level          | 100,000    | object      | 최종 학력(고등학교, 학사, 석사, 박사)|
| Performance_Score        | 100,000    | int64       | 성과 평가(1~5점 척도)|
| Monthly_Salary           | 100,000    | int64       | 직책과 성과 점수에 따른 월급(USD)|
| Work_Hours_Per_Week      | 100,000    | int64       | 주당 근무 시간 수|
| Projects_Handled         | 100,000    | int64       | 총 수행 프로젝트 수|
| Overtime_Hours           | 100,000    | int64       | 지난 1년 동안 총 초과근무 시간|
| Sick_Days                | 100,000    | int64       | 사용 병가일 수|
| Remote_Work_Frequency    | 100,000    | int64       | 원격으로 근무한 시간의 백분율(0%, 25%, 50%, 75%, 100%)|
| Team_Size                | 100,000    | int64       | 소속 팀의 인원 수|
| Training_Hours           | 100,000    | int64       | 교육에 소요된 시간|
| Promotions               | 100,000    | int64       | 근무 기간 동안 승진 횟수|
| Employee_Satisfaction_Score | 100,000 | float64     | 직원 만족도 평가(1.0~5.0 척도)|
| Resigned                 | 100,000    | bool        | 퇴직 여부(퇴직: 1, 재직: 0)|

<br>

### 데이터 전처리 과정
**결측치 및 이상치 처리**

모든 컬럼에서 결측치는 발견되지 않았기 때문의 별도의 결측치 처리 과정은 거치지 않았다. Gender 컬럼에 other로 기록 되어 있는 데이터가 Male, Female에 비해 매우 드물기 때문에 이상치로 판단하여 제거하였다.

<br>

**범주형 데이터 인코딩**

<br>

**클래스 가중치 부여**

- 범주형 변수의 클래스별 퇴사율
  
|![교육수준별 퇴사율](image/퇴사율%20변수%20분포/교육수준별%20퇴사율.png)|![근무부서별 퇴사율](image/퇴사율%20변수%20분포/근무부서별%20퇴사율.png)|
|:---:|:---:|
|교육수준별 퇴사율|근무부서별 퇴사율|
|![원격근무 빈도별 퇴사율](image/퇴사율%20변수%20분포/원격근무%20빈도별%20퇴사율.png)|![직무별 퇴사율](image/퇴사율%20변수%20분포/직무별%20퇴사율.png)|
|원격근무 빈도별 퇴사율|직무별 퇴사율|

<br>

- 가중치 조절에 따른 수치형 변수의 분포 차이
  
|![가중치 조절 전 만족도](image/가중치%20조절%20전.png)|![가중치 조절 후 만족도]image/가중치%20조절%20후.png|
|:---:|:---:|
|가중치 조절 전 지원 만족도 Boxplot. 퇴직자와 재직자의 만족도 분포 차이가 거의 없음.|가중치 부여하여 인위적으로 분포의 차이를 만듦.|

<br>

**샘플링**

|![퇴직자 재직자 클래스 불균형](image/퇴직자_재직자_비율.png)|
|:---:|
|퇴직자 재직자 비율. 클래스 불균형을 볼 수 있음.|

Target인 Resigned 변수는 퇴직자(1)와 재직자(0)의 비율이 약 9:1로 클래스가 매우 불균형한 데이터이다. 이러한 경우 많은 모델이 데이터의 대부분을 차지하는 다수 클래스에 치우친 예측을 하기 때문에 정확도는 높아보일지 몰라도 의미 있는 예측을 하기 어렵다. 하지만 이탈 예측 모델 같은 경우 소수 클래스인 퇴직을 분류하는 것이 더 중요하기 때문에 불균형을 해소할 필요가 있다.  

클래스의 불균형을 해소하기 위해 다음과 같은 샘플링 기법을 적용하여 균형을 맞추었다.
- **Over Sampling**: 소수 클래스의 데이터를 다수 클래스의 데이터 수에 맞게 증가시키는 방식으로, 여기선 Resampling을 사용. 퇴사자(Resigned: 1)를 무작위로 복제하여 10,000명으로 증가시킴.  
- **Under Sampling**: 다수 클래스의 데이터를 소수 클래스 데이터 수에 맞게 줄이는 방식으로, 여기선 Random Sampling을 사용. 재직자(Resigned: 0) 10,000명 무작위 추출.


<br>

## 분석결과
### 머신러닝 모델

**사용 모델 설명**

|모델명|모델 설명|
|:---|:---|
|Logistic Regression|ㅇㄹㅇㄹ|
|Decision Tree|ㅇㄹㅇ|
|RandomForest|df|

<br>

**성능 평가 및 비교**
| 모델         | Accuracy | F1-score |Recall(퇴직자)|   AUC   |
|:------------:|:--------:|:--------------:|:-------:|-------:|
| Logistic Regression          |   0.90   |0.00|     0.85       | 0.67  |
| Decision Tree      |   0.97   |     0.9       |0.94| 0.98  |
| RandomForest      |   0.94   |     0.93       |0.43| 0.97  |

- **Logistirc Regression**: 재직자(0)은 잘 예측하지만, 퇴직자(1)은 완전히 무시하는 편향이 발생함. 퇴직자와 재직자의 구분 규칙을 단순 선형 결정 경계로는 파악할 수 없음을 알 수 있음. 따라서 실제 이탈 예측 모델로 활용하기에는 적합하지 않음.
- **Decision Tree**: 퇴직자와 재직자 모두를 매우 정확하게 예측함. 하지만 과적합에 대한 검토가 필요함. 
- **RandomForest**: 재직자 예측은 매우 정확하지만, 퇴직자 예측은 Recall 값이 낮아 놓치는 퇴사자가 많을 수 있음. 

<br>   
  
### 딥러닝 모델

**모델링 개요**

본 프로젝트에서 사용한 HR 데이터셋은 target 클래스 내 설명 변수들의 분포가 균일하게 나타나기 때문에 특정 변수에 기반한 퇴사 패턴이 명확하게 드러나지 않았다. 이는 데이터셋이 실제 HR 데이터를 기반으로 하면서도, 편향이나 민감한 상관관계를 제거하여 예측 모델이나 알고리즘의 성능을 테스트하기 위해 인위적으로 제작된 데이터셋이기 때문으로 판단된다. 하지만 딥러닝 모델은 숨겨진 변수 간의 패턴이나 규칙을 발견하여 더 좋은 예측 성능을 낼 가능성이 있기 때문에 딥러닝 기반 모델도 설계하였다. 


<br>

**사용 모델 설명**

|모델명|모델 설명|
|:---|:---|
|Multi-Layer Perceptron|으앙|
|DeepMLP|흐앙|
|CNN 1D|설명|
|LSTM|설명|
|GRU|설명|
|Transformer|설명|
|AutoEncoder|설명|

<br>

**성능 비교 및 모델 선정**

| 모델(Early Stopping + Dropout 적용)          | Accuracy | Macro F1-score |   AUC   | Best Threshold |
|:---------------------------------------------|:---------|:---------------|:--------|:---------------|
| MLP                    |   0.66   |     0.64       | 0.73    |      0.54      |
| DeepMLP                                      |   0.66   |     0.65       | 0.74    |      0.48      |
| Transformer    |   0.65   |     0.65       | 0.74    |      0.50      |
| Autoencoder Classifier                       |   0.67   |     0.66       | 0.75    |      0.46      |
| MLP + DeepMLP|   0.66   |     0.65       |0.75     |       -        |
|Transformer + MLP                             |   0.65   |     0.64       |  0.72   |      0.49      |


- **MLP**: 기본적인 딥러닝 구조의 모델이지만 비선형적 관계를 비교적 효과적으로 학습함. 하지만 재직자(0)에 대한 Recall 값이 낮아 균형적 예측에 대한 성능은 다소 떨어진다고 볼 수 있음.
- **DeepMLP**: Dropout 및 BatchNorm을 활용하여 regularization 효과를 넣었으나, 성능은 MLP보다 낮았다. 복잡한 구조가 오히려 일반화에 방해가 되었을 가능성이 있다. 하지만 하이퍼파라미터 튜닝을 통해 향상될 가능성이 있다고 판단하였다.
- **Transformer**: 소수 클래스의 데이터(퇴직자)를 탐지하기 적합한 구조의 모델로, Recall 0.83의 높은 값을 나타냄. 하지만 재직자 예측에 대해서는 퇴직자 예측보다는 낮은 성능으로 False Positive에 대한 우려가 존재함.
- **AutoEncoder Classifier**:입력 데이터를 잠재 공간으로 압축한 후, 해당 벡터를 기반으로 분류를 진행하는 모델. 성능은 DeepMLP와 유사했으며, AUC는 0.5728로 중간 수준.
- **MLP + DeepMLP**: Dropout과 Early Stopping을 적용하여 과적합을 방지하였음. 퇴직자 Recall이 전체 모델 중 가장 높게 나타남. 재직자 예측에 대해서는 퇴직자에 비해서 약하지만, False Negative를 방지하고 퇴직자를 탐지하는 것이 더 중요하기 때문에 유용한 모델이라고 볼 수 있음.


<br>

|ROC-Curve|모델 설명|Best Parameters|
|:---|:---|:---|
|![MLP](image/MLP_ROC.png)|MLP| ```hidden1: 200``` ```hidden2: 74``` ```lr: 0.008040853765733618``` ```batch_size: 128```|
|![DeepMLP](image/DeepMLP_ROC.png)|DeepMLP|```hidden1: 174``` ```hidden2: 89``` ```dropout: 0.4907642593510716``` ```lr: 0.0020379497782479967``` ```batch_size: 128```|
|![AutoEncoder](image/AutoEncoder_ROC.png)|AutoEncoder|```encoding_dim: 96``` ```dropout_rate: 0.27937315793717066``` ```lr: 0.00939236014289022``` ```batch_size: 128```|
|![Transformer](image/Transformer_ROC.png)|Transformer|```d_model: 32``` ```nhead: 2``` ```num_layers: 1``` ```dropout: 0.24372847923584276``` ```lr: 0.002500389542492971``` ```batch_size: 128```|
|![Ensemble](image/Ensemble_MLPDEEPMLP_ROC.png)|MLP + DeepMLP 앙상블 모델|```hidden_dim1: 66``` ```hidden_dim2: 53``` ```dropout: 0.4091386964275948``` ```lr: 0.008331419334917479``` ```batch_size: 128```|
|![Trans + MLP](image/TRANSFORMERMLP_ENSEMBLE_ROC.png)| Transformer + MLP 앙상블 모델|```d_model: 128``` ```nhead: 8``` ```num_layers: 3``` ```dropout: 0.2281870695395805``` ```lr: 0.00015211612225622003``` ```batch_size: 128```|

