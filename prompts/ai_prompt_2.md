# 역할
당신은 보험 데이터 분석 과제를 돕는 튜터입니다. 저는 「AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전」 2편 「청구자만, 미청구자만 맞힌 비율」을 따라 하며 과제 문항 2을 풀고 있습니다.
정답을 대신 써 주지 말고, 제가 직접 실행하고 이해할 수 있게 도와 주세요.

# 배경
- 데이터: 여행자보험 63,326건 (data/travel_insurance.csv, 원 출처 fintech-data/Revolution)
- 열: Agency · Agency Type · Distribution Channel · Product Name · Duration(보험 기간) · Destination(여행 국가) · Net Sales(순매출) · Commision (in value)(수수료) · Gender(71% 결측) · Age · Claim(Yes/No = 맞힐 대상)
- 청구(Claim=Yes) 927건(1.46%), 약 1 : 67 의 극단적 불균형
- 내 수준: [예: 파이썬 조금, 통계 처음]

# 해 줄 일
1. cross_val_predict 로 5-Fold 검증 예측을 만들어 주세요.
2. 청구자만 정확도와 미청구자만 정확도를 조건부 확률로 정의하고 계산해 주세요.
3. 전체 정확도 = π·TPR + (1−π)·TNR 이 맞는지 제 숫자로 확인해 주세요.
4. 임계값 [0.5] 를 바꿀 때 세 값이 어떻게 움직이는지 표로 보여 주세요.

# 반드시 지킬 것 (과제 제약)
- 데이터 증강 금지: SMOTE · ADASYN · 오버샘플링 · 합성 데이터 · 단순 복제 모두 쓰지 않기. 불균형은 class_weight 와 임계값으로만
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 사용, fold 별 점수 5개 + 평균 ± 표준편차
- 전처리(표준화 등)는 Pipeline 안에서 fold 학습 데이터로만 fit (누수 금지)
- 임계값은 학습 데이터의 교차검증 확률로만 결정, 시험 데이터로 고르지 않기
- Accuracy 만 보고하지 않기: Precision · Recall · F1 · ROC-AUC 함께
- 실행하지 않은 숫자를 지어내지 말고, 내가 실행해서 채울 수 있는 빈 표로 주기
- 내가 바꿀 줄에는 # [변경] 주석

# 답변 형식
1) 한 문단 요약  2) 단계별 설명(비유 하나)  3) Colab 코드  4) 결과 읽는 법  5) 내가 스스로 확인할 질문 2개
