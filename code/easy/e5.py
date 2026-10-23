# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 5편 초보용 코드 (e5.py)
# 글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0
# 규칙: def · for · lambda · 들여쓰기 없음. 맨 위 [다이얼] 몇 줄만 바꿔 다시 실행하세요.

# [다이얼] 사례의 기저율(양성 비율) · 민감도 · 특이도
PREVALENCE = 0.00172
SENSITIVITY = 0.80
SPECIFICITY = 0.95

import pandas as pd
import numpy as np
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ② 베이즈 정리로 정밀도(PPV) 계산: 걸러 낸 것 중 진짜 양성 비율
ppv = SENSITIVITY * PREVALENCE / (SENSITIVITY * PREVALENCE + (1 - SPECIFICITY) * (1 - PREVALENCE))
print('PPV (precision):', round(ppv, 4))

# ③ 10,000건 기준으로 그려 보기
n = 10000
pos = n * PREVALENCE
print('positives:', round(pos, 1), '| caught:', round(pos * SENSITIVITY, 1), '| false alarms:', round((n - pos) * (1 - SPECIFICITY), 1))

# ④ 이 과제 데이터 (5-Fold 로지스틱: TPR 0.7184, TNR 0.7992, 청구율 1.46%) 와 비교
tpr = 0.7184
tnr = 0.7992
pi = 927 / 63326
print('travel insurance PPV:', round(tpr * pi / (tpr * pi + (1 - tnr) * (1 - pi)), 4))
