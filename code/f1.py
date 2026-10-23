"""
AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 1편 정식 코드 (f1.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

과제 문항 1: 데이터 증강 없이 5-Fold 교차검증 기준 성능 산출
 ① 전체 63,326건을 StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 로 나눈다
 ② 모델 5개를 fold 마다 다시 학습한다 (표준화는 Pipeline 안 = fold 학습 데이터로만 fit)
    - 불균형: 증강 없이 class_weight='balanced' (KNN 은 임계값 1/15)
 ③ fold 별 Accuracy · Precision · Recall · F1 · ROC-AUC 와 평균 ± 표준편차, 기준선을 출력한다
 ④ 모든 fold 의 검증 예측(OOF)을 oof_predictions.csv 로 저장한다 (2·3편에서 사용)
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import common

PATH = os.environ.get('TI_DATA')   # 비우면 저장소 data/travel_insurance.csv 부터 읽는다
raw = common.load_raw(PATH)
df, info = common.clean(raw)
X, y = common.encode(df)
X, y = X.to_numpy(dtype=float), y.to_numpy()

MODELS = {
    'KNN': lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    'SVC (RBF)': lambda: make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', random_state=42)),
    'Linear SVM': lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=5000, class_weight='balanced', random_state=42)),
    'Logistic': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    'Decision Tree': lambda: DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, oof = [], {m: np.zeros(len(y)) for m in MODELS}
oof_score = {m: np.zeros(len(y)) for m in MODELS}
for k, (tr, va) in enumerate(skf.split(X, y), 1):
    # 기준선: 모두 '청구 없음'
    rows.append({'model': 'Baseline (all No)', 'fold': k, 'accuracy': (y[va] == 0).mean(), 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.5})
    for name, make in MODELS.items():
        m = make()
        Xtr, ytr = X[tr], y[tr]
        if name == 'SVC (RBF)':   # 계산량 때문에 fold 학습 데이터에서 15,000건 층화 추출 (새 데이터 생성 아님)
            Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=15000, stratify=ytr, random_state=42)
        m.fit(Xtr, ytr)
        if hasattr(m, 'predict_proba') and name != 'SVC (RBF)':
            s = m.predict_proba(X[va])[:, 1]
        else:
            s = m.decision_function(X[va])
        p = (s >= 1 / 15).astype(int) if name == 'KNN' else m.predict(X[va])
        oof[name][va], oof_score[name][va] = p, s
        rows.append({'model': name, 'fold': k, 'accuracy': accuracy_score(y[va], p), 'precision': precision_score(y[va], p),
                     'recall': recall_score(y[va], p), 'f1': f1_score(y[va], p), 'roc_auc': roc_auc_score(y[va], s)})
res = pd.DataFrame(rows)
print('claims per fold:', [int(y[va].sum()) for _, va in skf.split(X, y)])
for metric in ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']:
    t = res.pivot(index='model', columns='fold', values=metric)
    t['mean'], t['std'] = t.mean(axis=1), t.std(axis=1, ddof=1)
    print(f'\n[{metric}] fold 1~5, mean, std')
    print(t.round(4).to_string())
out = pd.DataFrame({'y': y, **{f'pred_{m}': oof[m].astype(int) for m in MODELS}, **{f'score_{m}': oof_score[m] for m in MODELS}})
out.to_csv('oof_predictions.csv', index=False)
print('\nsaved oof_predictions.csv', out.shape)
