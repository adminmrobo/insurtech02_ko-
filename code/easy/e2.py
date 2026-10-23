# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 2편 초보용 코드 (e2.py)
# 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0
# 규칙: def · for · lambda · 들여쓰기 없음. 맨 위 [다이얼] 몇 줄만 바꿔 다시 실행하세요.

# [다이얼] 판정 임계값 (기본 0.5)
THRESHOLD = 0.5

# 데이터: 이 저장소에 복사해 둔 원본 (원 출처 fintech-data/Revolution 의 travel insurance.csv 와 같은 파일)
DATA_URL = 'https://raw.githubusercontent.com/adminmrobo/ai-biz-insurtech-02/main/data/travel_insurance.csv'
# 저장소 주소가 열리지 않으면 아래 줄의 # 를 지우고 원 출처에서 읽으세요
# DATA_URL = 'https://raw.githubusercontent.com/fintech-data/Revolution/main/data/travel%20insurance.csv'

import pandas as pd
import numpy as np
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ① 데이터 읽기 + 정리 규칙 (인슈어테크_01 2편과 같음 · 행 삭제 없음 · 증강 없음)
raw = pd.read_csv(DATA_URL)
raw = raw.rename(columns={'Agency Type': 'Agency_Type', 'Distribution Channel': 'Distribution_Channel', 'Product Name': 'Product_Name', 'Net Sales': 'Net_Sales', 'Commision (in value)': 'Commision_in_value'})
df = raw.copy()
df['Gender'] = df['Gender'].fillna('Unknown')
df['Duration'] = df['Duration'].mask(df['Duration'] <= 0, df.loc[df['Duration'] > 0, 'Duration'].median()).clip(upper=731)
df['Age'] = df['Age'].mask(df['Age'] > 100, df.loc[df['Age'] <= 100, 'Age'].median())
df['Net_Sales'] = df['Net_Sales'].clip(df['Net_Sales'].quantile(0.001), df['Net_Sales'].quantile(0.999))
df['Commision_in_value'] = df['Commision_in_value'].clip(upper=df['Commision_in_value'].quantile(0.999))
top10 = df['Destination'].value_counts().index[:10]
df['Destination'] = df['Destination'].where(df['Destination'].isin(top10), 'OTHER')
y = (df['Claim'] == 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns='Claim'), drop_first=True, dtype=int)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ② 5-Fold 검증 확률 (각 고객은 자기를 학습에 쓰지 않은 모델이 판정)
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
proba = cross_val_predict(logistic, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
pred = (proba >= THRESHOLD).astype(int)

# ③ 청구자만 정확도 = P(예측 청구 | 실제 청구) = 재현율(TPR)
tpr = (pred[y == 1] == 1).mean()
# ④ 미청구자만 정확도 = P(예측 없음 | 실제 없음) = 특이도(TNR)
tnr = (pred[y == 0] == 0).mean()
# ⑤ 전체 정확도 = π × TPR + (1 − π) × TNR
pi = y.mean()
print('TPR (claim only):', round(tpr, 4), '| TNR (no-claim only):', round(tnr, 4))
print('overall accuracy:', round((pred == y).mean(), 4), '| pi*TPR + (1-pi)*TNR:', round(pi * tpr + (1 - pi) * tnr, 4), '| pi:', round(pi, 5))
print('balanced accuracy (TPR+TNR)/2:', round((tpr + tnr) / 2, 4))
