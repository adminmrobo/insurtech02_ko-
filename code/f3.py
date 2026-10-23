"""
AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 3편 정식 코드 (f3.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

과제 문항 3: 문항 2의 결과를 그대로 '성능'으로 보고할 때의 문제점과 개선 방안을 숫자로 확인
 ① 문제 1 · 한쪽만 본다      → 개선: TNR·정밀도·균형 정확도를 함께 (모두 청구 모델과 비교)
 ② 문제 2 · 임계값 하나의 값  → 개선: 임계값과 무관한 PR-AUC, 상위 k% 재현율
 ③ 문제 3 · 청구자가 적어 흔들림 → 개선: fold 표준편차 + Wilson 95% 신뢰구간 (데이터를 늘리지 않는 분석식)
 ④ 문제 4 · 배포 환경의 청구율이 다르면 정밀도가 바뀐다 → 개선: 청구율별 정밀도(PPV) 표
 ⑤ 문제 5 · 임계값·전처리를 검증 데이터로 고르면 부풀려진다 → 개선: fold 안에서만 임계값 선택(중첩 교차검증)
 (먼저 f1.py 를 실행해 oof_predictions.csv 를 만든다)
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, precision_recall_curve
import common

oof = pd.read_csv('oof_predictions.csv')
y = oof['y'].to_numpy()
models = [c[5:] for c in oof.columns if c.startswith('pred_')]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ① ② ③ 모델별 요약
rows = []
for m in models:
    p, s = oof[f'pred_{m}'].to_numpy(), oof[f'score_{m}'].to_numpy()
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    lo, hi = wilson(tp, int(y.sum()))
    order = np.argsort(-s, kind='stable')
    cum = np.cumsum(y[order])
    rows.append({'model': m, 'TPR': tp / y.sum(), 'TPR_95CI_low': lo, 'TPR_95CI_high': hi, 'TNR': ((p == 0) & (y == 0)).sum() / (y == 0).sum(),
                 'precision': tp / max(tp + fp, 1), 'PR_AUC': average_precision_score(y, s),
                 'recall@10%': cum[int(0.1 * len(y)) - 1] / y.sum(), 'recall@20%': cum[int(0.2 * len(y)) - 1] / y.sum(), 'recall@30%': cum[int(0.3 * len(y)) - 1] / y.sum()})
print('PR_AUC baseline (random) =', round(y.mean(), 4))
print(pd.DataFrame(rows).round(4).to_string(index=False))

# fold 하나(청구 185명)의 Wilson 신뢰구간 폭
lo, hi = wilson(133, 185)
print(f'\nexample fold: 133/185 = {133 / 185:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]')

# ④ 청구율(π)에 따른 정밀도: PPV = TPR·π / (TPR·π + (1−TNR)(1−π))
lr = [r for r in rows if r['model'] == 'Logistic'][0]
for pi in [0.005, 0.0146, 0.03, 0.05, 0.10]:
    ppv = lr['TPR'] * pi / (lr['TPR'] * pi + (1 - lr['TNR']) * (1 - pi))
    print(f'Logistic PPV at claim rate {pi:.2%}: {ppv:.4f}')

# ⑤ 중첩 교차검증: 바깥 fold 의 학습 데이터 안에서만(안쪽 5-Fold) 임계값을 고른다 → 바깥 검증 fold 에 한 번 적용
raw = common.load_raw(os.environ.get('TI_DATA'))
df, info = common.clean(raw)
X, yy = common.encode(df)
X, yy = X.to_numpy(dtype=float), yy.to_numpy()
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nest = []
for k, (tr, va) in enumerate(outer.split(X, yy), 1):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_in = cross_val_predict(pipe, X[tr], yy[tr], cv=inner, method='predict_proba')[:, 1]
    pr, rc, th = precision_recall_curve(yy[tr], s_in)
    t = th[np.where(rc[:-1] >= 0.80)[0][-1]]
    s_out = pipe.fit(X[tr], yy[tr]).predict_proba(X[va])[:, 1]
    p = (s_out >= t).astype(int)
    nest.append({'fold': k, 'threshold': t, 'TPR': (p[yy[va] == 1] == 1).mean(), 'TNR': (p[yy[va] == 0] == 0).mean(),
                 'precision': p[yy[va] == 1].sum() / max(p.sum(), 1), 'flagged': p.mean()})
nest = pd.DataFrame(nest)
print('\nnested CV (threshold chosen inside each training fold, target recall 0.80):')
print(nest.round(4).to_string(index=False))
print('mean', nest.mean(numeric_only=True).round(4).to_dict(), 'std', nest.std(numeric_only=True).round(4).to_dict())
