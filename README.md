**[한국어](#ko)** · English · 中文 · 日本語 · O'zbekcha · Русский (번역 예정)

---
<a id="ko"></a>
# AI(인공지능) x 비즈니스 #1 인슈어테크_02_분류모델_실전

기계학습 과제 「여행자 보험금 청구(Claim) 예측」의 다섯 문항을 한 편에 한 문항씩 따라 푸는 인터랙티브 튜토리얼입니다. 청구 1.46%의 극단적 불균형 데이터에서, 데이터 증강 없이 5-Fold 교차검증으로 성능을 재고, 청구자만의 정확도를 정의하고, 그 숫자를 성능이라 부를 때의 함정을 고치고, 모델을 서버 없는 HTML 하나로 내보내고, 다른 산업의 같은 문제와 비교합니다. 앞 시리즈: [인슈어테크_01 분류모델](https://adminmrobo.github.io/ai-biz-insurtech-01/)

- 글: 글 · 안상선 ((주)M-Robo 대표)
- 라이선스: CC BY-NC 4.0 (글·코드·그림). 데이터는 원 출처의 조건을 따릅니다.
- 공개 주소: https://adminmrobo.github.io/ai-biz-insurtech-02/ (저장소 이름은 작가가 정함)

## 언어

| 언어 | 폴더 | 상태 |
|---|---|---|
| 한국어 | `/` | 원본 |
| English · 中文 · 日本語 · O'zbekcha · Русский | `en/` `zh/` `ja/` `uz/` `ru/` | 번역 예정 |

## 과제 문항과 편

| 편 | 과제 | 파일 | 제목 | 유형 |
|---|---|---|---|---|
| 1 | 문항 1 | `1_kfold_cv.html` | 증강 없이 5-Fold로 성능 재기 | 해설형 · 확인 문제 4 |
| 2 | 문항 2 | `2_conditional_accuracy.html` | 청구자만, 미청구자만 맞힌 비율 | 해설형 · 확인 문제 3 |
| 3 | 문항 3 | `3_problems_fixes.html` | 그 숫자를 성능이라 부르면 생기는 일 | 방법형 · 확인 문제 4 |
| 4 | 문항 4 | `4_html_deploy.html` | 그 모델, HTML에 그대로 넣어도 될까 | 정리형 · 확인 문제 3 · 가이드라인 |
| 5 | 문항 5 | `5_similar_cases.html` | 희소 이벤트는 어디에나 있다 | 정리형 · 확인 문제 3 · 가이드라인 |

처음 화면 `index.html` · 교육자용 정답 `answers.html` (#a1 … #a5)

## 과제 공통 제약 (모든 편이 지킴)

- 데이터 증강 금지 (SMOTE · ADASYN · 오버샘플링 · 합성 · 복제). 불균형은 `class_weight='balanced'` 와 임계값으로만
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, fold 별 점수 5개 + 평균 ± 표준편차
- 전처리는 Pipeline 안에서 학습 fold 로만 fit, 임계값은 학습 데이터 교차검증 확률로만
- Accuracy 단독 보고 금지

## 폴더

```
├── index.html                처음 화면 (과제-편 대응 · 식구 · 데이터 · 연구 지도 · 보는 법)
├── 1_kfold_cv.html … 5_similar_cases.html
├── answers.html              교육자용 정답
├── 404.html
├── code/                     정식 코드: common.py, f1.py ~ f5.py (f2·f3 는 f1 이 만드는 oof_predictions.csv 사용)
│   └── easy/                 초보용 코드: e1.py ~ e5.py (def·for·lambda·들여쓰기 없음)
├── data/                     travel_insurance.csv (원본 사본), split.csv, logistic_model.json
├── prompts/                  AI 튜터 프롬프트 ai_prompt_1.md ~ ai_prompt_5.md
├── README.md · LICENSE.md · .nojekyll
```

## 데이터

- 원 출처: https://raw.githubusercontent.com/fintech-data/Revolution/main/data/travel%20insurance.csv
- 이 저장소의 `data/travel_insurance.csv` 는 위 파일과 **내용이 같은 사본**입니다(MD5 97b63f816e280e59165695868b734afd, 파일 이름의 공백만 밑줄로). 63,326행 × 11열, 청구 927건(1.46%).
- 코드(`common.load_raw`)는 저장소 안 파일 → 저장소 주소 → 원 출처 주소 순서로 읽습니다.
- 가상 데이터 생성·데이터 증강을 하지 않습니다. 정리 규칙(행 삭제 없음)은 인슈어테크_01 2편과 같습니다.

## 주의

교육용 예시이며 보험 가입·보험료·보상에 대한 조언이 아닙니다. 실제 보험 심사와 보험료 산정은 감독 규정과 계리 기준을 따릅니다. 특정 집단에 불리한 특성(예: 성별·나이)의 사용은 금지되거나 제한될 수 있습니다.
