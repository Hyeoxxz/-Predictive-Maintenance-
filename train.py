import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 12)]

def add_rolling_features(df):
    """최근 5사이클 평균/표준편차 추가 — 추세를 반영하기 위해"""
    df = df.sort_values(['engine_id', 'cycle']).copy()
    for col in SENSOR_COLS:
        df[f'{col}_mean5'] = (
            df.groupby('engine_id')[col]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df[f'{col}_std5'] = (
            df.groupby('engine_id')[col]
            .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        )
    return df

def main():
    print("=== 모델 학습 시작 ===")

    # 1. 데이터 로드
    df = pd.read_csv("sensor_data.csv")
    print(f"데이터 로드: {len(df):,}행")

    # 2. 피처 엔지니어링
    df = add_rolling_features(df)
    feature_cols = SENSOR_COLS + [f'{c}_mean5' for c in SENSOR_COLS] + [f'{c}_std5' for c in SENSOR_COLS] + ['cycle']

    X = df[feature_cols]
    y_rul = df['RUL']
    y_clf = df['failure_within_30']

    # 3. 학습/테스트 분할
    X_train, X_test, y_rul_train, y_rul_test, y_clf_train, y_clf_test = train_test_split(
        X, y_rul, y_clf, test_size=0.2, random_state=42
    )

    # 4. 스케일링
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print(f"학습 {len(X_train):,}행 / 테스트 {len(X_test):,}행")

    # 5. RUL 예측 모델 (회귀)
    print("\nRUL 예측 모델 학습 중...")
    rul_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rul_model.fit(X_train_s, y_rul_train)
    rul_pred = rul_model.predict(X_test_s)
    print(f"MAE : {mean_absolute_error(y_rul_test, rul_pred):.2f} 사이클")
    print(f"R²  : {r2_score(y_rul_test, rul_pred):.4f}")

    # 6. 고장 임박 분류 모델
    print("\n고장 임박 분류 모델 학습 중...")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_model.fit(X_train_s, y_clf_train)
    clf_prob = clf_model.predict_proba(X_test_s)[:, 1]
    print(f"AUC : {roc_auc_score(y_clf_test, clf_prob):.4f}")

    # 7. 저장
    joblib.dump(rul_model,    "rul_model.pkl")
    joblib.dump(clf_model,    "clf_model.pkl")
    joblib.dump(scaler,       "scaler.pkl")
    joblib.dump(feature_cols, "feature_cols.pkl")
    print("\n모델 저장 완료!")

if __name__ == "__main__":
    main()