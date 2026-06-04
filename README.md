# 🏭 설비 예지보전 시스템 (Predictive Maintenance)
> Smart Factory AI PM 포트폴리오 프로젝트  
> **설비 고장 예측 모델 + FastAPI 배포 + MLOps 모니터링 파이프라인**

---

## 📌 프로젝트 개요

스마트팩토리 현장에서 **설비가 고장나기 전에 미리 감지**하는 AI 서비스를 기획·개발·배포까지 구현한 프로젝트입니다.  
단순 모델 학습에서 끝나지 않고, **실제 서비스 운영 환경(API + 모니터링 로그)**까지 구축하여 MLOps 관점의 AI 솔루션 전 주기를 다룹니다.

| 항목 | 내용 |
|---|---|
| 도메인 | 스마트팩토리 / 설비 관리 |
| 과제 유형 | 회귀(RUL 예측) + 분류(고장 임박 탐지) |
| 핵심 기술 | Python, Scikit-learn, FastAPI, Pandas |
| 배포 방식 | REST API (FastAPI + Uvicorn) |
| 모니터링 | 예측 로그 CSV 저장 + 알림 등급 자동 분류 |

---

## 🎯 왜 이 프로젝트인가

> **"AI를 붙이는 것보다, 붙인 AI가 계속 작동하도록 만드는 것이 더 어렵다."**

공장에서 AI 검사기나 예측 모델을 도입해도, 현장 환경이 바뀌면 모델 성능이 서서히 저하됩니다.  
이 프로젝트는 **예측 → API 서빙 → 운영 로그 → 알림 체계**의 전 과정을 설계함으로써,  
단순 PoC(개념 검증)를 넘어 **실제 운영 가능한 AI 솔루션**의 구조를 보여줍니다.

---

## 🏗️ 시스템 아키텍처

```
[센서 데이터 입력]
       ↓
[Feature Engineering]  ← 롤링 평균/표준편차 (최근 5사이클 추세 반영)
       ↓
[AI 모델 예측]
  ├── RandomForest Regressor  → RUL(잔여 수명) 예측
  └── RandomForest Classifier → 고장 임박 확률 (0~1)
       ↓
[FastAPI 서버]  ← /predict, /health, /metrics, /logs
       ↓
[알림 등급 판정]
  ├── ✅ NORMAL   : RUL > 30, 고장확률 < 50%
  ├── ⚠️  WARNING  : RUL ≤ 30 또는 고장확률 ≥ 50%
  └── 🚨 CRITICAL : RUL ≤ 15 또는 고장확률 ≥ 85%
       ↓
[예측 로그 저장]  ← monitoring/prediction_log.csv
```

---

## 📊 모델 성능

| 지표 | 값 | 의미 |
|---|---|---|
| RUL 예측 MAE | **8.63 사이클** | 평균 8.63사이클 오차 이내 예측 |
| RUL R² | **0.9614** | 잔여 수명 변동의 96% 설명 |
| 고장 분류 AUC | **0.9987** | 고장 임박 탐지 정밀도 |
| CRITICAL F1 | **0.95** | 고위험 설비 탐지율 |

---

## 🛠️ 프로젝트 구조

```
predictive_maintenance/
├── data/
│   ├── generate_data.py     # CMAPSS 스타일 합성 데이터 생성
│   └── sensor_data.csv      # 150개 엔진, 34,250행
├── model/
│   ├── train.py             # 모델 학습 + 피처 엔지니어링
│   ├── rul_model.pkl        # RUL 회귀 모델
│   ├── clf_model.pkl        # 고장 분류 모델
│   ├── scaler.pkl           # 표준화 스케일러
│   └── metrics.json         # 모델 성능 기록
├── api/
│   ├── main.py              # FastAPI 서버 (4개 엔드포인트)
│   └── test_api.py          # 시나리오별 API 테스트
└── monitoring/
    ├── visualize.py         # 포트폴리오 시각화 생성
    └── prediction_log.csv   # 실시간 예측 로그
```

---

## 🚀 실행 방법

```bash
# 1. 데이터 생성
python3 data/generate_data.py

# 2. 모델 학습
python3 model/train.py

# 3. API 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. API 테스트
python3 api/test_api.py
```

### API 사용 예시

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": 1, "cycle": 240,
    "sensor_1": 518.68, "sensor_2": 650.5, "sensor_3": 1610.0,
    "sensor_4": 1423.0, "sensor_5": 14.63, "sensor_6": 21.61,
    "sensor_7": 549.2,  "sensor_8": 2378.0,"sensor_9": 9000.0,
    "sensor_10": 1.348, "sensor_11": 49.7
  }'
```

**응답 예시:**
```json
{
  "engine_id": 1,
  "cycle": 240,
  "predicted_rul": 2.9,
  "failure_probability": 0.9,
  "alert_level": "CRITICAL",
  "message": "즉시 점검 필요 — 잔여 수명 약 3사이클"
}
```

---

## 💡 MLOps 관점에서 본 설계 의도

| 관점 | 구현 내용 |
|---|---|
| **재현성** | 모든 파라미터와 성능을 metrics.json으로 버전 관리 |
| **모니터링** | 모든 예측 결과를 로그로 저장 → 성능 저하 감지 기반 |
| **운영 알림** | 3단계 알림 등급으로 현장 담당자 대응 우선순위 제공 |
| **확장 가능성** | MLflow 연동, 재학습 트리거 추가로 완전한 MLOps로 확장 가능 |

---

## 🔄 향후 발전 방향

- [ ] MLflow 실험 추적 연동 (모델 버전 관리 자동화)
- [ ] 드리프트 감지 로직 추가 (성능 저하 자동 알림)
- [ ] Docker 컨테이너화
- [ ] 실제 CMAPSS 데이터셋 적용 검증

---

**기술 스택:** Python 3.11.9 · Scikit-learn · FastAPI · Pandas · NumPy · Matplotlib
