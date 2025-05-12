from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import traceback

from models.prophet_model import train_prophet_model, predict_with_prophet
from models.xgboost_model import train_xgboost_model, predict_residuals
from utils.data_utils import prepare_data, density_to_crowd_level

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Crowd Forecast API")

# 예외를 콘솔에 출력하는 미들웨어 추가
@app.middleware("http")
async def log_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("❗ 요청 처리 중 예외 발생:")
        traceback.print_exc()  # 콘솔에 전체 에러 스택 출력
        raise e

class ForecastRequest(BaseModel):
    historicalData: List[Dict[str, Any]]
    futureDates: List[Dict[str, Any]]

class ForecastResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Crowd Forecast API is running"}

@app.post("/api/forecast/prophet-xgboost", response_model=ForecastResponse)
async def forecast_with_prophet_xgboost(request: ForecastRequest):
    try:
        print("📥 forecast_with_prophet_xgboost 호출됨")

        # 1. 데이터 준비
        historical_df, future_df, feature_cols = prepare_data(
            request.historicalData, request.futureDates
        )
        print("✅ 데이터 준비 완료")

        # 2. Prophet 모델 학습 및 예측
        prophet_model = train_prophet_model(historical_df)
        prophet_historical_pred, prophet_future_pred = predict_with_prophet(
            prophet_model, historical_df[['ds']], future_df[['ds']]
        )
        print("✅ Prophet 예측 완료")

        # 3. 잔차 계산
        historical_df['prophet_pred'] = prophet_historical_pred['yhat'].values
        historical_df['residuals'] = historical_df['y'] - historical_df['prophet_pred']

        # 4. XGBoost 모델 학습 및 예측
        xgb_model, metrics, feature_importance = train_xgboost_model(
            historical_df, feature_cols
        )
        print("✅ XGBoost 학습 및 예측 완료")

        future_df['prophet_pred'] = prophet_future_pred['yhat'].values
        future_df['residual_pred'] = predict_residuals(xgb_model, future_df, feature_cols)

        # 5. 최종 예측 계산
        future_df['final_pred'] = future_df['prophet_pred'] + future_df['residual_pred']
        future_df['final_pred'] = future_df['final_pred'].clip(lower=0)

        # 6. 혼잡도 레벨 계산
        future_df['crowd_level'] = future_df['final_pred'].apply(density_to_crowd_level)

        # 7. 신뢰 구간 계산
        future_df['lower_bound'] = prophet_future_pred['yhat_lower'] + future_df['residual_pred']
        future_df['upper_bound'] = prophet_future_pred['yhat_upper'] + future_df['residual_pred']
        future_df['lower_bound'] = future_df['lower_bound'].clip(lower=0)

        # 8. 응답 준비
        predictions = []
        for _, row in future_df.iterrows():
            prediction = {
                'date': str(row['ds']).split(' ')[0],  # '2025-05-13 00:00:00'에서 날짜 부분만 추출
                'predictedDensity': float(row['final_pred']),
                'crowdLevel': int(row['crowd_level']),
                'lowerBound': float(row['lower_bound']),
                'upperBound': float(row['upper_bound'])
            }

            for col in ['isWeekend', 'isHoliday', 'isSchoolVacation', 'temperature']:
                if col in row:
                    # NumPy 데이터 타입을 Python 기본 타입으로 변환
                    val = row[col]
                    if isinstance(val, (np.integer, np.floating, np.bool_)):
                        val = val.item()  # NumPy 스칼라 값을 Python 스칼라로 변환
                    prediction[col] = val

            predictions.append(prediction)

        # 9. 지표 및 특성 중요도
        # NumPy 데이터 타입 변환
        response_metrics = {
            'mae': float(metrics['mae']),  # 명시적으로 float로 변환
            'rmse': float(metrics['rmse']), # 명시적으로 float로 변환
        }

        # feature_importance 처리: NumPy 데이터 타입을 Python 기본 타입으로 변환
        converted_feature_importance = {}
        for k, v in feature_importance.items():
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                converted_feature_importance[k] = v.item()
            else:
                converted_feature_importance[k] = v

        response_metrics['featureImportance'] = converted_feature_importance

        print("✅ 예측 응답 반환 완료")
        return ForecastResponse(predictions=predictions, metrics=response_metrics)

    except Exception as e:
        print("❌ forecast_with_prophet_xgboost 내부 예외:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 추가 엔드포인트: 롯데월드 예측
@app.get("/api/forecast/lotte-world")
async def get_lotte_world_forecast(days: int = 30):
    try:
        return {"message": f"롯데월드 {days}일 예측 API는 아직 구현되지 않았습니다."}
    except Exception as e:
        print("❌ 롯데월드 예측 API 오류:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")