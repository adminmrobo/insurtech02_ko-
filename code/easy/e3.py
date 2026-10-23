# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 3편 초보용 코드 (e3.py)
# 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0
# 규칙: def · for · lambda · 들여쓰기 없음. 맨 위 [다이얼] 몇 줄만 바꿔 다시 실행하세요.

# [다이얼] 배포 환경에서 예상하는 청구 비율, 심사할 상위 비율
DEPLOY_CLAIM_RATE = 0.03
TOP_SHARE = 0.20

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
from sklearn.metrics import average_precision_score

# ② 5-Fold 검증 확률
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
proba = cross_val_predict(logistic, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
pred = (proba >= 0.5).astype(int)
tp = int(((pred == 1) & (y == 1)).sum())
n = int(y.sum())
tpr = tp / n
tnr = ((pred == 0) & (y == 0)).sum() / (y == 0).sum()

# ③ 문제: 청구자 수가 적어 흔들린다 → Wilson 95% 신뢰구간 (데이터를 늘리지 않는 계산식)
z = 1.96
center = (tpr + z * z / (2 * n)) / (1 + z * z / n)
half = z * np.sqrt(tpr * (1 - tpr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
print('TPR', round(tpr, 4), '95% CI [', round(center - half, 4), ',', round(center + half, 4), ']')

# ④ 문제: 임계값 하나에 묶인다 → 임계값과 무관한 PR-AUC, 상위 k% 재현율
print('PR-AUC:', round(average_precision_score(y, proba), 4), '| random baseline:', round(y.mean(), 4))
top_k = int(TOP_SHARE * len(y))
print('recall @ top', TOP_SHARE, ':', round(y.to_numpy()[np.argsort(-proba, kind='stable')][:top_k].sum() / n, 4))

# ⑤ 문제: 배포 환경의 청구율이 다르면 정밀도가 바뀐다 → PPV = TPR·π / (TPR·π + (1−TNR)(1−π))
pi = DEPLOY_CLAIM_RATE
print('precision at claim rate', pi, ':', round(tpr * pi / (tpr * pi + (1 - tnr) * (1 - pi)), 4))
