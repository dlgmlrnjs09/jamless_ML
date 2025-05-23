from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import traceback
import json
from datetime import date, datetime

from models.prophet_model import train_prophet_model, predict_with_prophet
from models.xgboost_model import train_xgboost_model, predict_residuals
from utils.data_utils import prepare_data, density_to_crowd_level

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
        
        # 원본 API 파라미터 로그 출력
        print("📊 원본 입력 데이터:")
        
        # historicalData 로그 출력
        print(f"📈 과거 데이터 샘플 (총 {len(request.historicalData)}개):")
        if len(request.historicalData) > 0:
            # 첫 번째와 마지막 레코드 출력
            print(f"첫 번째 레코드: {json.dumps(request.historicalData[0], ensure_ascii=False, default=str)}")
            print(f"마지막 레코드: {json.dumps(request.historicalData[-1], ensure_ascii=False, default=str)}")
            
            # 모든 키 목록 (어떤 특성이 포함되어 있는지 확인)
            all_keys = set()
            for item in request.historicalData:
                all_keys.update(item.keys())
            print(f"과거 데이터 포함 특성: {sorted(list(all_keys))}")
        
        # futureDates 로그 출력
        print(f"🔮 예측 데이터 샘플 (총 {len(request.futureDates)}개):")
        if len(request.futureDates) > 0:
            # 첫 번째와 마지막 레코드 출력
            print(f"첫 번째 레코드: {json.dumps(request.futureDates[0], ensure_ascii=False, default=str)}")
            print(f"마지막 레코드: {json.dumps(request.futureDates[-1], ensure_ascii=False, default=str)}")
            
            # 모든 키 목록 (어떤 특성이 포함되어 있는지 확인)
            all_keys = set()
            for item in request.futureDates:
                all_keys.update(item.keys())
            print(f"예측 데이터 포함 특성: {sorted(list(all_keys))}")

        # 1. 방학 데이터 제외 처리
        modified_historical_data = []
        for item in request.historicalData:
            new_item = {k: v for k, v in item.items() if k != 'isSchoolVacation' and k != 'vacationType'}
            modified_historical_data.append(new_item)
            
        modified_future_dates = []
        for item in request.futureDates:
            new_item = {k: v for k, v in item.items() if k != 'isSchoolVacation' and k != 'vacationType'}
            modified_future_dates.append(new_item)

        # 데이터 준비
        historical_df, future_df, feature_cols = prepare_data(
            modified_historical_data, modified_future_dates
        )
        
        # 변환된 DataFrame 로그 출력
        print("📊 변환된 DataFrame 정보:")
        print("-- 과거 데이터 컬럼:")
        print(historical_df.columns.tolist())
        print("-- 예측 데이터 컬럼:")
        print(future_df.columns.tolist())
        
        # 과거 데이터의 처음 3개 행과 마지막 3개 행 출력
        print("-- 과거 데이터 샘플:")
        if len(historical_df) > 6:
            print("처음 3개 행:")
            print(historical_df.head(3).to_string())
            print("마지막 3개 행:")
            print(historical_df.tail(3).to_string())
        else:
            print(historical_df.to_string())
        
        # 예측 데이터의 처음 3개 행과 마지막 3개 행 출력
        print("-- 예측 데이터 샘플:")
        if len(future_df) > 6:
            print("처음 3개 행:")
            print(future_df.head(3).to_string())
            print("마지막 3개 행:")
            print(future_df.tail(3).to_string())
        else:
            print(future_df.to_string())
            
        print("✅ 데이터 준비 완료")

        # 2. Prophet 모델 학습 및 예측
        prophet_model = train_prophet_model(historical_df)
        prophet_historical_pred, prophet_future_pred = predict_with_prophet(
            prophet_model, historical_df, future_df
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
        
        # 최종 예측 결과 로그 출력
        print("📊 최종 예측 결과:")
        result_cols = ['ds', 'prophet_pred', 'residual_pred', 'final_pred', 'crowd_level']
        print(future_df[result_cols].head(10).to_string())

        # 8. 응답 준비
        predictions = []
        prediction_comments = {}  # 예측 결과에 대한 설명 저장

        # Prophet 및 XGBoost 모델의 주요 요인 파악
        prophet_components = prophet_future_pred[['ds', 'trend', 'yearly', 'weekly', 'multiplicative_terms']]

        # XGBoost 모델의 특성 중요도
        feature_importance_dict = dict(zip(feature_cols, xgb_model.feature_importances_))
        top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        for _, row in future_df.iterrows():
            # 날짜에서 시간 부분 제거
            date_only = str(row['ds']).split(' ')[0] if ' ' in str(row['ds']) else str(row['ds'])
            date_obj = pd.to_datetime(date_only).date()

            # 해당 날짜의 Prophet 컴포넌트 찾기
            prophet_comp = prophet_components[prophet_components['ds'].dt.date == date_obj]

            # 예측 결과 생성
            prediction = {
                'date': date_only,
                'predictedDensity': float(row['final_pred']),
                'crowdLevel': int(row['crowd_level']),
                'lowerBound': float(row['lower_bound']),
                'upperBound': float(row['upper_bound'])
            }

            # 추가 특성 포함
            for col in ['isWeekend', 'isHoliday', 'temperature']:
                if col in row:
                    val = row[col]
                    if isinstance(val, (np.integer, np.floating, np.bool_)):
                        val = val.item()
                    prediction[col] = val

            # 예측 설명 생성
            comment = generate_prediction_comment(
                date_obj,
                row,
                prophet_comp,
                historical_df,
                top_features,
                feature_cols
            )

            prediction_comments[date_only] = comment
            predictions.append(prediction)

        # 9. 지표 및 특성 중요도
        # NumPy 데이터 타입 변환
        response_metrics = {
            'mae': float(metrics['mae']),  # 명시적으로 float로 변환
            'rmse': float(metrics['rmse']), # 명시적으로 float로 변환
        }

        # feature_importance 처리: NumPy 데이터 타입을 Python 기본 타입으로 변환
        converted_feature_importance = {}
        for k, v in feature_importance_dict.items():
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                converted_feature_importance[k] = v.item()
            else:
                converted_feature_importance[k] = v

        response_metrics['featureImportance'] = converted_feature_importance
        response_metrics['predictionComments'] = prediction_comments

        print("✅ 예측 응답 반환 완료")
        return ForecastResponse(predictions=predictions, metrics=response_metrics)

    except Exception as e:
        print("❌ forecast_with_prophet_xgboost 내부 예외:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def generate_prediction_comment(date, row, prophet_comp, historical_df, top_features, feature_cols):
    """예측 결과에 대한 설명 생성"""
    comment = f"{date.strftime('%Y년 %m월 %d일')}의 혼잡도 예측 설명:\n"

    # 1. 요일 및 주말 여부
    weekdays = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}
    day_of_week = date.weekday()
    day_name = weekdays[day_of_week]
    is_weekend = int(row.get('isWeekend', 0)) == 1
    day_type = "주말" if is_weekend else "평일"
    comment += f"- 해당 일자는 {day_name}({day_type})입니다.\n"

    # 2. 특별 일자 (공휴일)
    is_holiday = int(row.get('isHoliday', 0)) == 1

    if is_holiday:
        holiday_name = row.get('holidayName', '공휴일')
        comment += f"- 이 날은 {holiday_name}입니다.\n"

    # 3. 날씨 정보
    weather_comment = []

    # 기온
    if 'temperature' in row and row['temperature'] is not None:
        temp = float(row['temperature'])
        weather_comment.append(f"기온은 {temp:.1f}°C")

    # 습도
    if 'humidity' in row and row['humidity'] is not None:
        humidity = float(row['humidity'])
        weather_comment.append(f"습도는 {humidity:.1f}%")

    # 강수량/적설량 정보
    precip_info = []
    if 'isRainy' in row and int(row.get('isRainy', 0)) == 1:
        precip_val = f"{float(row.get('precipitation', 0)):.1f}mm" if 'precipitation' in row else "있음"
        precip_info.append(f"강수량 {precip_val}")

    if 'isSnow' in row and int(row.get('isSnow', 0)) == 1:
        snow_val = f"{float(row.get('snowfall', 0)):.1f}cm" if 'snowfall' in row else "있음"
        precip_info.append(f"적설량 {snow_val}")

    if precip_info:
        weather_comment.append("강수 예보(" + ", ".join(precip_info) + ")")

    # 날씨 정보 합치기
    if weather_comment:
        comment += f"- 날씨 조건: {', '.join(weather_comment)}입니다. 이는 방문객 수에 영향을 줄 수 있습니다.\n"

    # 4. 과거 유사 일자 찾기
    similar_dates = find_similar_dates(date, historical_df)
    if similar_dates:
        similar_date, similarity, crowd_level = similar_dates[0]
        crowd_level_desc = get_crowd_level_description(crowd_level)
        comment += f"- 과거 유사 일자인 {similar_date.strftime('%Y년 %m월 %d일')}의 혼잡도는 {crowd_level}({crowd_level_desc})이었습니다.\n"

    # 5. 주요 영향 요인
    comment += "- 예측에 가장 큰 영향을 준 요인:\n"

    # Prophet 트렌드 및 계절성 효과
    if not prophet_comp.empty:
        trend_effect = prophet_comp['trend'].values[0]
        weekly_effect = prophet_comp['weekly'].values[0] if 'weekly' in prophet_comp else 0
        yearly_effect = prophet_comp['yearly'].values[0] if 'yearly' in prophet_comp else 0

        comment += f"  * 장기 트렌드: {'증가' if trend_effect > 0 else '감소'} 추세\n"
        comment += f"  * 주간 패턴: 해당 요일은 평균 대비 {'+' if weekly_effect > 0 else ''}{weekly_effect:.2f} 영향\n"
        comment += f"  * 연간 패턴: 해당 월/일은 평균 대비 {'+' if yearly_effect > 0 else ''}{yearly_effect:.2f} 영향\n"

    # XGBoost 특성 중요도
    for feature, importance in top_features:
        if feature in row:
            feature_value = row[feature]
            if isinstance(feature_value, (np.integer, np.floating, np.bool_)):
                feature_value = feature_value.item()

            # 특성 이름 가독성 향상
            feature_name = {
                'isWeekend': '주말 여부',
                'isHoliday': '공휴일 여부',
                'temperature': '기온',
                'humidity': '습도',
                'precipitation': '강수량',
                'snowfall': '적설량',
                'isRainy': '강수 여부',
                'isSnow': '눈 여부',
                'dayOfWeek': '요일',
                'month': '월'
            }.get(feature, feature)

            # 특성 값 포맷팅
            if feature in ['temperature', 'precipitation', 'humidity', 'snowfall']:
                feature_display = f"{float(feature_value):.1f}"
                # 단위 추가
                if feature == 'temperature': feature_display += "°C"
                elif feature == 'precipitation': feature_display += "mm"
                elif feature == 'humidity': feature_display += "%"
                elif feature == 'snowfall': feature_display += "cm"
            elif feature in ['isRainy', 'isSnow', 'isWeekend', 'isHoliday']:
                feature_display = "예" if int(feature_value) == 1 else "아니오"
            else:
                feature_display = str(feature_value)

            comment += f"  * {feature_name}: {feature_display} (중요도: {importance:.4f})\n"

    # 6. 최종 예측 결과
    density = float(row['final_pred'])
    crowd_level = int(row['crowd_level'])
    crowd_level_desc = get_crowd_level_description(crowd_level)

    comment += f"\n최종 예측: 단위 면적당 방문자 수 {density:.6f}명/㎡로 혼잡도 {crowd_level}단계({crowd_level_desc})로 예상됩니다."

    return comment


def find_similar_dates(target_date, historical_df):
    """과거 데이터에서 유사한 날짜 찾기"""
    similar_dates = []

    if 'ds' not in historical_df.columns or historical_df.empty:
        return similar_dates

    for _, row in historical_df.iterrows():
        hist_date = pd.to_datetime(row['ds']).date()

        # 같은 월/일인 경우
        if hist_date.month == target_date.month and hist_date.day == target_date.day:
            # 같은 요일인지 확인
            if hist_date.weekday() == target_date.weekday():
                similarity = 1.0  # 매우 유사 (같은 월/일, 같은 요일)
            else:
                similarity = 0.8  # 다소 유사 (같은 월/일, 다른 요일)

            if 'crowd_level' in row:
                crowd_level = row['crowd_level']
            elif 'y' in row:
                # 혼잡도 계산
                density = float(row['y'])
                crowd_level = density_to_crowd_level(density)
            else:
                crowd_level = None

            similar_dates.append((hist_date, similarity, crowd_level))

    # 유사도 순으로 정렬
    similar_dates.sort(key=lambda x: x[1], reverse=True)
    return similar_dates


def get_crowd_level_description(crowd_level):
    """혼잡도 레벨에 대한 설명 반환"""
    descriptions = {
        1: "여유",
        2: "보통",
        3: "혼잡",
        4: "매우 혼잡"
    }
    return descriptions.get(crowd_level, "알 수 없음")


# 기존의 추가 엔드포인트: 롯데월드 예측
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