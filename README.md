# Gold & Bitcoin Price forecast

## 프로젝트 개요
본 프로젝트는 금(Gold) 시세와 비트코인(Bitcoin) 시세 간의 상관관계를 분석하고, 다양한 머신러닝 및 딥러닝 모델을 활용하여 비트코인 가격을 예측하는 것을 목표로 합니다.
jupyter Notebook으로 제작되었습니다.

## 기술 스택
### 사용한 언어 및 라이브러리
Python : NumPy, Pandas, Matplotlib, Seaborn
  scikit-learn : LinearRegression, DecisionTree, RandomForest, XGBoost, GradientBoosting 
  TensorFlow/Keras : Dense Network, LSTM
  기타: MinMaxScaler, EarlyStopping, ModelCheckpoint

### 머신러닝 & 딥러닝 모델링
회귀 기반: **선형 회귀, 다항 회귀**
트리 기반: **Decision Tree, Random Forest, XGBoost, Gradient Boosting**
딥러닝: **ANN(MLP), LSTM**

### EDA & 시각화
상관관계 분석: **Heatmap, Pairplot**
시계열 데이터 분석: **Line Plot**
예측값 vs. 실제값 비교: **Scatter Plot, kdeplot**


## 데이터 분석 및 EDA

### 데이터셋 개요
금 시세 데이터: `gold_usd_price.csv`
(2000년~2024년)
비트코인 시세 데이터: `btc_2012_2024.csv` 
(2012년~2024년)

데이터 병합: 날짜(`date`)를 기준으로 병합
학습 데이터(train): 2012년 ~ 2022년
테스트 데이터(test): 2023년 ~ 2024년

### 상관관계 분석
- 히트맵(Heatmap) 결과에 따르면 금(Gold) 시세와 비트코인(BTC) 시세 간 0.73의 양의 상관관계가 있다는 것으로 상관관계가 거의 없다는 것이 확인됨.
- 분석 결과, 2016년 이후 비트코인의 변동성이 급격히 증가하는 패턴을 보임.

## 모델링
다양한 모델을 활용하여 비트코인 가격을 예측하고 성능을 비교하였습니다.
금 시세와 비트코인 시세의 상관관계가 거의 없다는것을 확인(0.6497510785139097)한 후 BTC 지표(고가(high_btc), 저가(low_btc) , 거래량(volume_btc))를 분석에 포함하였습니다.

1. 선형 회귀 (Linear Regression)
활용 데이터 : 금 시세 + BTC 지표 

예측 성능 (R²): 0.99

2. 다항 회귀 (Polynomial Regression)]
활용 데이터 : 금 시세만
예측 성능 (R²): 0.60 

3. 의사결정트리 (Decision Tree)
활용 데이터:금 시세 + BTC 지표
예측 성능 (R²): 0.98

4. 랜덤 포레스트 (Random Forest)
활용 데이터:금 시세 + BTC 지표 
예측 성능 (R²): 0.99

5. XGBoost
활용 데이터:금 시세 + BTC 지표
예측 성능 (R²): 0.99

6. Gradient Boosting
활용 데이터:금 시세 + BTC 지표
예측 성능 (R²): 0.99

7. 인공신경망 (ANN - MLP)
활용 데이터:금 시세 + BTC 지표
예측 성능 (R²): 0.98

8. LSTM
활용 데이터:금 시세 + BTC 지표
예측 성능 (R²): 0.99

## 모델링 요약
**선형 회귀, 랜덤 포레스트, XGBoost, LSTM** 모델이 가장 높은 성능(R² ≈ 0.99) 기록
**다항 회귀 모델(Polynomial Regression)의 경우 과소적합(Underfitting) 문제 발생 (R² = 0.60)**
**LSTM 모델이 금융 데이터의 시계열적 특성을 반영하여 우수한 성능을 보임**
**랜덤 포레스트 및 XGBoost 모델이 비교적 안정적인 예측을 수행**

## 성능 평가 및 분석

### 1. 평가 지표
**MSE (Mean Squared Error)**
**MAE (Mean Absolute Error)**
**RMSE (Root Mean Squared Error)**
**R² (결정계수, 모델 설명력 평가)**

### 2. 시각화 분석
**금 가격(Gold price)과 비트코인 가격(BTC price)의 시계열 변동 분석**
**모델별 예측값(Predicted)과 실제값(Actual)을 비교하는 그래프를 통해 성능 평가**
**XGBoost, 랜덤 포레스트, LSTM 모델이 가장 실제값과 유사한 경향을 보임**
**R² 성능 비교 차트에서 다항 회귀(Polynomial Regression)의 성능이 가장 낮게 나타남**




## 결과
**LSTM 모델이 금융 시계열 데이터에서 효과적임을 확인**
**XGBoost와 랜덤 포레스트가 높은 성능을 제공하면서도 계산량이 적음**
**단순 선형 회귀도 강력한 성능을 보였으나, 미래 예측에는 한계가 존재**








