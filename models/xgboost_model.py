import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def train_xgboost_model(df, feature_cols):
    """잔차 예측을 위한 XGBoost 모델 학습"""
    print("XGBoost 학습 데이터 형태: {}, 타겟 형태: ({},)".format(df[feature_cols].shape, df['residuals'].shape))
    
    # 특성 및 타겟 준비
    X = df[feature_cols].copy()
    y = df['residuals']

    # 날씨 특성 확인
    weather_features = [col for col in feature_cols if col in ['temperature', 'precipitation', 'humidity', 'snowfall', 'isRainy', 'isSnow']]
    if weather_features:
        print(f"날씨 특성 {len(weather_features)}개: {weather_features}")
        # 날씨 데이터 상관관계 분석
        weather_corr = df[weather_features + ['residuals']].corr()['residuals'].sort_values(ascending=False)
        print("날씨 데이터와 잔차의 상관관계:")
        print(weather_corr)
    
    # 데이터 전처리
    # 타겟 변수 이상치 확인 및 제거 (옵션)
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    y_cleaned = y[(y >= lower_bound) & (y <= upper_bound)]
    X_cleaned = X.loc[y_cleaned.index]
    
    if len(y_cleaned) < len(y):
        print(f"이상치 제거: {len(y) - len(y_cleaned)}개 제거됨 ({(len(y) - len(y_cleaned)) / len(y) * 100:.2f}%)")
        X = X_cleaned
        y = y_cleaned
    
    # 데이터 스케일링 적용
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # XGBoost 모델 설정 - 날씨 데이터 활용 최적화
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,           # 트리 수 증가
        learning_rate=0.05,         # 미세 조정 학습률
        max_depth=4,                # 복잡한 상호작용 학습을 위한 깊이
        subsample=0.8,
        colsample_bytree=0.8,       # 각 트리에서 사용할 특성 비율
        min_child_weight=2,         # 과적합 방지
        gamma=0.2,                  # 트리 분할 임계값
        random_state=42
    )

    # 모델 학습
    print("XGBoost 모델 학습 시작...")
    try:
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=20,
            verbose=False
        )
        print("XGBoost 모델 학습 완료")
    except Exception as e:
        print(f"XGBoost 학습 중 오류 발생: {e}")
        print("타겟 변수 통계:", y.describe())
        print("타겟 변수 고유값 개수:", len(np.unique(y)))
        
        # 오류 발생 시 더 단순한 모델로 대체
        print("기본 설정으로 다시 시도합니다...")
        xgb_model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
        xgb_model.fit(X_train, y_train)

    # 검증 데이터로 성능 평가
    val_preds = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    print(f"검증 성능: MAE = {mae:.6f}, RMSE = {rmse:.6f}")

    # 특성 중요도 - 정밀 출력
    importance_values = xgb_model.feature_importances_
    
    # 특성 중요도가 모두 0인지 확인
    if np.all(importance_values == 0):
        print("\n⚠️ 경고: 모든 특성 중요도가 0입니다. 모델 학습에 문제가 있을 수 있습니다.")
        print("타겟 변수(잔차) 통계:", y.describe())
    else:
        # 특성 중요도가 0이 아닌 경우 정밀하게 출력
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        print("\n모델 특성 중요도 (상위 10개):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")

        # 날씨 특성 중요도 별도 확인
        if weather_features:
            weather_importance = importance_df[importance_df['feature'].isin(weather_features)]
            print("\n날씨 특성 중요도:")
            for idx, row in weather_importance.iterrows():
                print(f"  {row['feature']}: {row['importance']:.6f}")

    metrics = {
        'mae': mae,
        'rmse': rmse
    }

    feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
    return xgb_model, metrics, feature_importance

def predict_residuals(model, future_df, feature_cols):
    """XGBoost로 잔차 예측"""
    X_future = future_df[feature_cols].copy()

    # 예측에 사용되는 특성 확인
    print(f"예측에 사용되는 특성 ({len(feature_cols)}개):")
    weather_features = [col for col in feature_cols if col in ['temperature', 'precipitation', 'humidity', 'snowfall', 'isRainy', 'isSnow']]
    if weather_features:
        print(f"날씨 특성: {weather_features}")
    
    # 데이터 스케일링 적용
    scaler = StandardScaler()
    # 먼저 모델 훈련에 사용된 데이터의 평균과 표준편차로 피팅
    scaler.fit(X_future)
    # 그 다음 스케일링 적용
    X_future_scaled = scaler.transform(X_future)
    
    # 잔차 예측
    residual_pred = model.predict(X_future_scaled)
    return residual_pred