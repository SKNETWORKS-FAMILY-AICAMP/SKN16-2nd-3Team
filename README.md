# SKN16-2nd-3Team
## 프로젝트 주제
### 주제 선정 배경
### 분석 목적
본 프로젝트의 목적은 사내 인사 데이터를 바탕으로 직원의 이탈 여부를 예측하는 모델을 개발하여 이탈 가능성이 높은 직원을 사전에 파악하고, 이를 바탕으로 조직의 인적 리스크 관리의 유용성을 높이는 것입니다.

## 팀 소개

## 데이터셋 개요

기업의 근무 환경에서 직원 성과, 생산성 및 인구 통계와 관련된 주요 측면을 파악할 수 있는 데이터가 포함되어 있습니다.

데이터 출처: [Kaggle, Employee Performance and Productivity Data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data)



### 데이터 설명
**데이터프레임 구조**

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


### 데이터 전처리 과정
**결측치 및 이상치 처리**

모든 컬럼에서 결측치는 발견되지 않았기 때문의 별도의 결측치 처리 과정은 거치지 않았습니다. Gender 컬럼에 other로 기록 되어 있는 데이터가 Male, Female에 비해 매우 드물기 때문에 이상치로 판단하여 제거하였습니다.

**범주형 데이터 인코딩**

**샘플링**

## EDA

## 분석결과
### 머신러닝 모델
1. 사용 모델 설명
2. 성능 비교 및 모델 선정
3. 파라미터 튜닝

### 딥러닝 모델
1. 모델링 개요
2. 최적화 과정
3. 최종 성능
