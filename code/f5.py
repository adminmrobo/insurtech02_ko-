"""
AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전 · 5편 정식 코드 (f5.py)
글 · 안상선 ((주)M-Robo 대표) · CC BY-NC 4.0

과제 문항 5: 같은 구조(클래스 불균형 + 희소 이벤트)를 가진 실제 사례 비교
 ① 사례마다 기저율 · 민감도 · 특이도를 표로 둔다 (출처는 5편 페이지 참고)
 ② 베이즈 정리로 정밀도(PPV)와 '전부 음성' 기준선 정확도를 계산한다
 ③ 10,000건 기준으로 잡은 수 · 놓친 수 · 거짓 경보 수를 센다
 ④ 기저율이 불확실한 경우를 위해 기저율만 바꿔 가며 정밀도를 다시 본다
데이터 파일이 필요 없습니다.
"""
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ① 사례 표 (민감도·특이도 중 '가정'이라고 적힌 값은 출처 값이 아님)
cases = pd.DataFrame([
    ["card fraud (ULB)", 0.00172, 0.800, 0.950, "sens/spec assumed"],
    ["mammography (BCSC 2017)", 0.0059, 0.869, 0.889, "prev = CDR 5.1/1000 / sens"],
    ["disk failure (Backblaze)", 0.0136, 0.767, 0.958, "AFR 2025 + SMART 2016"],
    ["insurance fraud (CAIF)", 0.10, 0.800, 0.950, "prev ~10% of P&C losses; sens/spec assumed"],
    ["travel claim (this task, logistic)", 927 / 63326, 0.7184, 0.7992, "5-fold OOF, 2편"],
], columns=["case", "prevalence", "sensitivity", "specificity", "note"])

# ② 정밀도 = 잡은 양성 / 경보 전체
tp = cases["sensitivity"] * cases["prevalence"]
fp = (1 - cases["specificity"]) * (1 - cases["prevalence"])
cases["PPV"] = tp / (tp + fp)
cases["acc_all_negative"] = 1 - cases["prevalence"]
cases["acc_model"] = tp + cases["specificity"] * (1 - cases["prevalence"])
cases["flag_rate"] = tp + fp

# ③ 10,000건 기준
n = 10000
cases["positives"] = n * cases["prevalence"]
cases["caught"] = cases["positives"] * cases["sensitivity"]
cases["missed"] = cases["positives"] - cases["caught"]
cases["false_alarms"] = (n - cases["positives"]) * (1 - cases["specificity"])
print("=== Case table (per 10,000) ===")
print(cases.drop(columns="note").round(4).to_string(index=False))
print()
print(cases[["case", "note"]].to_string(index=False))

# ④ 기저율만 바꾸면 정밀도가 어떻게 변하나 (민감도 0.80, 특이도 0.95 고정)
prev = pd.Series([0.001, 0.005, 0.0146, 0.03, 0.05, 0.10, 0.20], name="prevalence")
ppv = 0.80 * prev / (0.80 * prev + 0.05 * (1 - prev))
print()
print("=== PPV vs prevalence (sens 0.80, spec 0.95) ===")
print(pd.DataFrame({"prevalence": prev, "PPV": ppv.round(4)}).to_string(index=False))

# 확인: 유방암 검진 계산값이 논문의 PPV1 4.4% 와 맞는지
print()
print("mammography PPV check (paper PPV1 = 0.044):", round(cases.loc[1, "PPV"], 4))
