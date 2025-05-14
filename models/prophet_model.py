from prophet import Prophet
import pandas as pd
import numpy as np

def train_prophet_model(df):
    """Prophet 모델 학습"""
    # 기본 데이터 준비
    prophet_df = df[['ds', 'y']].copy()

    # 날씨 관련 추가 변수 확인
    weather_regressors = [col for col in df.columns if col in ['temperature', 'humidity', 'precipitation', 'snowfall']]
    has_weather_data = len(weather_regressors) > 0

    if has_weather_data:
        print(f"Prophet 모델에 날씨 회귀변수 {len(weather_regressors)}개 추가: {weather_regressors}")
        # 날씨 데이터를 prophet_df에 추가
        for regressor in weather_regressors:
            prophet_df[regressor] = df[regressor]

    # 휴일 준비
    holidays = None
    if 'isHoliday' in df.columns and 'holidayName' in df.columns:
        holiday_df = df[df['isHoliday'] == 1][['ds', 'holidayName']].copy()
        if not holiday_df.empty:
            holiday_df.columns = ['ds', 'holiday']
            holidays = holiday_df

    # Prophet 모델 설정
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        seasonality_mode='multiplicative'  # 관광지에 적합한 모드
    )

    # 월별 계절성 추가
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 날씨 관련 추가 변수 처리
    for regressor in weather_regressors:
        model.add_regressor(regressor, mode='multiplicative')

    # 모델 학습
    print("Prophet 모델 학습 시작...")
    model.fit(prophet_df)
    print("Prophet 모델 학습 완료")

    return model

def predict_with_prophet(model, historical_df, future_df):
    """Prophet으로 예측 수행"""
    # 필요한 회귀변수 확인
    required_regressors = []
    try:
        if hasattr(model, 'extra_regressors'):
            required_regressors = list(model.extra_regressors.keys())
    except Exception as e:
        print(f"회귀변수 목록 확인 중 오류 (계속 진행): {e}")

    print(f"필요한 회귀변수: {required_regressors}")

    # 과거/미래 데이터 준비
    historical_pred_df = pd.DataFrame({'ds': historical_df['ds']})
    future_pred_df = pd.DataFrame({'ds': future_df['ds']})
    
    # 회귀변수 데이터 추가
    for regressor in required_regressors:
        # 과거 데이터에 회귀변수 추가
        if regressor in historical_df.columns:
            historical_pred_df[regressor] = historical_df[regressor].values
            print(f"과거 데이터에 회귀변수 '{regressor}' 이미 존재함")
        else:
            historical_pred_df[regressor] = 0
            print(f"과거 데이터에 회귀변수 '{regressor}' 추가 (0으로 채움)")
            
        # 미래 데이터에 회귀변수 추가
        if regressor in future_df.columns:
            future_pred_df[regressor] = future_df[regressor].values
            print(f"미래 데이터에 회귀변수 '{regressor}' 이미 존재함")
        else:
            future_pred_df[regressor] = 0
            print(f"미래 데이터에 회귀변수 '{regressor}' 추가 (0으로 채움)")

    # 전처리 로그
    print(f"Prophet 예측 - 과거 데이터: {historical_pred_df.shape}, 미래 데이터: {future_pred_df.shape}")
    print(f"과거 데이터 컬럼: {list(historical_pred_df.columns)}")
    print(f"미래 데이터 컬럼: {list(future_pred_df.columns)}")

    # 과거 데이터 예측
    prophet_historical = model.predict(historical_pred_df)

    # 미래 데이터 예측
    prophet_future = model.predict(future_pred_df)

    return prophet_historical, prophet_future