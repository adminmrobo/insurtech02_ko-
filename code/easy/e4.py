# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 4편 초보용 코드 (e4.py)
# 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0
# 규칙: def · for · lambda · 들여쓰기 없음. 맨 위 [다이얼] 몇 줄만 바꿔 다시 실행하세요.

# [다이얼] 성별 빈칸 채우기 규칙을 배포 앱에서 빼먹으면? (True = 규칙 적용)
FILL_GENDER = True

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

import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# ② 최종 모델: 학습 80% 로 학습, 임계값은 학습 데이터 교차검증으로 (목표 재현율 0.80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_train)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
oof = cross_val_predict(lr, scaler.transform(X_train), y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
pr_p, pr_r, pr_t = precision_recall_curve(y_train, oof)
threshold = float(pr_t[np.where(pr_r[:-1] >= 0.80)[0][-1]])
lr.fit(scaler.transform(X_train), y_train)

# ③ HTML 로 함께 내보낼 것: 열 순서 · 표준화 값 · 계수 · 절편 · 임계값 (+ 정리 규칙은 ① 그대로)
export = {'columns': X.columns.tolist(), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'threshold': threshold}
json.dump(export, open('logistic_model.json', 'w'))

# ④ 배포 앱처럼: 원본 입력(raw)에 규칙을 적용하고, 열 순서를 export 에 맞춰 계산
dep = raw.loc[X_test.index].drop(columns='Claim').copy()
dep['Gender'] = dep['Gender'].fillna('Unknown') if FILL_GENDER else dep['Gender']
dep['Destination'] = dep['Destination'].where(dep['Destination'].isin(top10), 'OTHER')
dep['Duration'] = df.loc[X_test.index, 'Duration']
dep['Age'] = df.loc[X_test.index, 'Age']
dep['Net_Sales'] = df.loc[X_test.index, 'Net_Sales']
dep['Commision_in_value'] = df.loc[X_test.index, 'Commision_in_value']
X_dep = pd.get_dummies(dep, dtype=int).reindex(columns=export['columns'], fill_value=0)
z = ((X_dep.to_numpy() - np.array(export['mean'])) / np.array(export['scale'])) @ np.array(export['coef']) + export['intercept']
p_dep = 1 / (1 + np.exp(-z))

# ⑤ sklearn 판정과 비교: 규칙을 다 지키면 100%, 하나라도 빼먹으면 달라진다
p_sk = lr.predict_proba(scaler.transform(X_test))[:, 1]
print('threshold:', round(threshold, 4), '| same decisions:', round(((p_dep >= threshold) == (p_sk >= threshold)).mean() * 100, 2), '%', '| changed:', int(((p_dep >= threshold) != (p_sk >= threshold)).sum()))
