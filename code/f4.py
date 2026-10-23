"""
AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 4편 정식 코드 (f4.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

과제 문항 4: 외부 서버 없는 단일 HTML 로 배포할 때, 문항 2의 모델을 '그대로' 쓸 수 있나
 ① 최종 모델(로지스틱 + 교차검증으로 고른 임계값)을 학습 데이터(80%)로 학습한다
 ② 브라우저가 계산하려면 함께 내보내야 할 것을 JSON 하나로 묶는다
    (전처리 규칙 · 범주 사전 · 열 순서 · 표준화 평균/표준편차 · 계수 · 절편 · 임계값)
 ③ JSON 만 보고 계산하는 함수로 시험 데이터를 다시 판정 → sklearn 과 같은지 확인
 ④ 전처리 규칙을 빼먹고 '원본 입력을 그대로' 넣으면 판정이 몇 건 바뀌는지 센다 (학습-배포 분포 차이)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve
import common

raw = common.load_raw(os.environ.get('TI_DATA'))
df, info = common.clean(raw)
X, y = common.encode(df)
tr, te = common.split_index(y)

# ① 최종 모델
scaler = StandardScaler().fit(X.iloc[tr])
Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
oof = cross_val_predict(lr, Xtr, y.iloc[tr], cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba')[:, 1]
p_, r_, t_ = precision_recall_curve(y.iloc[tr], oof)
thr = float(t_[np.where(r_[:-1] >= 0.80)[0][-1]])
lr.fit(Xtr, y.iloc[tr])
p_sk = lr.predict_proba(Xte)[:, 1]

# ② 함께 내보낼 것
cats = {c: sorted(df[c].unique().tolist()) for c in common.CAT_COLS}
export = {
    'columns': X.columns.tolist(), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
    'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]), 'threshold': thr,
    'clean': {'duration_median': float(info['duration_median']), 'duration_max': 731, 'age_max': 100, 'age_median': float(info['age_median']),
              'net_low': float(info['net_low']), 'net_high': float(info['net_high']), 'comm_high': float(info['comm_high']),
              'top10_destinations': info['top10_destinations'], 'gender_missing': 'Unknown'},
    'categories': cats,
}
json.dump(export, open('logistic_model.json', 'w'))
print(f'saved logistic_model.json  ({os.path.getsize("logistic_model.json"):,} bytes) · columns {len(export["columns"])} · threshold {thr:.4f}')


# ③ JSON 만 보고 계산 (브라우저 JS 와 같은 순서)
def prepare(row, e=export, do_clean=True, map_other=True, fill_gender=True):
    r = dict(row)
    c = e['clean']
    if fill_gender and (pd.isna(r['Gender']) or r['Gender'] in ('None', '')):
        r['Gender'] = c['gender_missing']
    if do_clean:
        if r['Duration'] <= 0: r['Duration'] = c['duration_median']
        r['Duration'] = min(r['Duration'], c['duration_max'])
        if r['Age'] > c['age_max']: r['Age'] = c['age_median']
        r['Net_Sales'] = min(max(r['Net_Sales'], c['net_low']), c['net_high'])
        r['Commision_in_value'] = min(r['Commision_in_value'], c['comm_high'])
    if map_other and r['Destination'] not in c['top10_destinations']:
        r['Destination'] = 'OTHER'
    x = np.zeros(len(e['columns']))
    for i, col in enumerate(e['columns']):
        if col in common.NUM_COLS:
            x[i] = r[col]
        else:
            base = max((b for b in common.CAT_COLS if col.startswith(b + '_')), key=len)
            x[i] = 1.0 if str(r[base]) == col[len(base) + 1:] else 0.0
    return x


def proba(x, e=export):
    z = np.dot((x - np.array(e['mean'])) / np.array(e['scale']), np.array(e['coef'])) + e['intercept']
    return 1 / (1 + np.exp(-z))


raw_te = raw.iloc[te].reset_index(drop=True)
p_json = np.array([proba(prepare(r)) for r in raw_te.to_dict('records')])
print(f'JSON vs sklearn: max |diff| = {np.abs(p_json - p_sk).max():.2e} · same decisions = {((p_json >= thr) == (p_sk >= thr)).mean():.4%}')

# ④ 빼먹으면 생기는 일 (시험 데이터 12,666명, 원본 입력 그대로)
base_dec = p_sk >= thr
for name, kw in [('skip numeric cleaning', dict(do_clean=False)), ('skip OTHER mapping', dict(map_other=False)),
                 ('skip Gender fill', dict(fill_gender=False)), ('skip all three', dict(do_clean=False, map_other=False, fill_gender=False))]:
    pj = np.array([proba(prepare(r, **kw)) for r in raw_te.to_dict('records')])
    ch = (pj >= thr) != base_dec
    print(f'{name:>22}: decisions changed {int(ch.sum()):>5} ({ch.mean():.2%}) · max prob change {np.abs(pj - p_sk).max():.3f}')
print('test rows needing each rule:', {'Duration<=0 or >731': int(((raw_te.Duration <= 0) | (raw_te.Duration > 731)).sum()),
      'Age>100': int((raw_te.Age > 100).sum()), 'Destination not top10': int((~raw_te.Destination.isin(info['top10_destinations'])).sum()),
      'Gender missing': int(raw_te.Gender.isna().sum())})
