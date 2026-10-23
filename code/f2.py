"""
AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 2편 정식 코드 (f2.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

과제 문항 2: 실제 청구자(Claim=Yes)만 / 미청구자(Claim=No)만 대상으로 한 정확도
 ① 1편(f1.py)이 저장한 5-Fold 검증 예측 oof_predictions.csv 를 읽는다 (먼저 f1.py 실행)
 ② 청구자만 정확도  = P(예측=청구 | 실제=청구)  = 재현율 = 민감도(TPR)
    미청구자만 정확도 = P(예측=없음 | 실제=없음) = 특이도(TNR)
 ③ 전체 정확도 = π × TPR + (1 − π) × TNR  (π = 청구 비율) 가 성립하는지 숫자로 확인
 ④ fold 별 값과 평균 ± 표준편차, 극단 모델(모두 청구 / 모두 없음)과 비교
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

oof = pd.read_csv('oof_predictions.csv')
y = oof['y'].to_numpy()
pi = y.mean()
models = [c[5:] for c in oof.columns if c.startswith('pred_')]
preds = {m: oof[f'pred_{m}'].to_numpy() for m in models}
preds['All Yes (모두 청구)'] = np.ones_like(y)
preds['All No (모두 없음)'] = np.zeros_like(y)

rows = []
for m, p in preds.items():
    tpr = (p[y == 1] == 1).mean()
    tnr = (p[y == 0] == 0).mean()
    acc = (p == y).mean()
    rows.append({'model': m, 'acc_claim_only(TPR)': tpr, 'acc_noclaim_only(TNR)': tnr, 'overall_acc': acc,
                 'pi*TPR+(1-pi)*TNR': pi * tpr + (1 - pi) * tnr, 'balanced_acc': (tpr + tnr) / 2,
                 'caught': int(p[y == 1].sum()), 'missed': int((p[y == 1] == 0).sum()), 'false_alarm': int(p[y == 0].sum())})
print(f'pi (claim rate) = {pi:.5f}  claims = {int(y.sum())}  no-claims = {int((y == 0).sum())}')
print(pd.DataFrame(rows).round(4).to_string(index=False))

# fold 별 (1편과 같은 나누기)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fr = []
for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    for m in models:
        p, t = preds[m][va], y[va]
        fr.append({'model': m, 'fold': k, 'TPR': (p[t == 1] == 1).mean(), 'TNR': (p[t == 0] == 0).mean()})
fr = pd.DataFrame(fr)
for col in ['TPR', 'TNR']:
    t = fr.pivot(index='model', columns='fold', values=col)
    t['mean'], t['std'] = t.mean(axis=1), t.std(axis=1, ddof=1)
    print(f'\n[{col}] fold 1~5, mean, std')
    print(t.round(4).to_string())
