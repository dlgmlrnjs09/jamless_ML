import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_xgboost_model(df, feature_cols):
    """잔차 예측을 위한 XGBoost 모델 학습"""
    # 특성 및 타겟 준비
    X = df[feature_cols].copy()
    y = df['residuals']

    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost 모델 설정
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,  # 여기에 early_stopping_rounds 추가
        random_state=42
    )

    # 모델 학습 - early_stopping_rounds를 fit() 에서 제거
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 검증 데이터로 성능 평가
    val_preds = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # 특성 중요도
    feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))

    metrics = {
        'mae': mae,
        'rmse': rmse
    }

    return xgb_model, metrics, feature_importance

def predict_residuals(model, future_df, feature_cols):
    """XGBoost로 잔차 예측"""
    X_future = future_df[feature_cols].copy()
    return model.predict(X_future)