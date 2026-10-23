# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 1편 초보용 코드 (e1.py)
# 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0
# 규칙: def · for · lambda · 들여쓰기 없음. 맨 위 [다이얼] 몇 줄만 바꿔 다시 실행하세요.

# [다이얼] 비교할 나무 깊이
TREE_DEPTH = 6

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# ② 5-Fold: 청구 비율을 fold 마다 같게 (층화), 섞기 고정
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# ③ 모델 2개: 표준화는 Pipeline 안에 넣어 fold 학습 데이터로만 fit (누수 방지), 불균형은 class_weight
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, class_weight='balanced', random_state=42)
r_lr = cross_validate(logistic, X, y, cv=cv5, scoring=scoring)
r_dt = cross_validate(tree, X, y, cv=cv5, scoring=scoring)

# ④ fold 별 점수 5개 + 평균 ± 표준편차 (표준편차는 ddof=1)
table = pd.DataFrame({'Logistic accuracy': r_lr['test_accuracy'], 'Logistic recall': r_lr['test_recall'], 'Logistic roc_auc': r_lr['test_roc_auc'], 'Tree accuracy': r_dt['test_accuracy'], 'Tree recall': r_dt['test_recall'], 'Tree roc_auc': r_dt['test_roc_auc']}, index=['fold1', 'fold2', 'fold3', 'fold4', 'fold5'])
table.loc['mean'] = table.iloc[:5].mean()
table.loc['std'] = table.iloc[:5].std(ddof=1)
print(table.round(4))

# ⑤ 기준선: 모두 '청구 없음'이라고 답해도 정확도는 이만큼 나온다
print('Baseline accuracy (all No):', round((y == 0).mean(), 4), '| Baseline recall: 0.0')
