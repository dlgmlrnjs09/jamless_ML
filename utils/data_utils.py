import pandas as pd
import numpy as np

def prepare_data(historical_data, future_dates):
    """데이터 전처리 및 특성 준비"""
    # DataFrame 변환
    historical_df = pd.DataFrame(historical_data)
    future_df = pd.DataFrame(future_dates)

    # 날짜 형식 변환
    historical_df['ds'] = pd.to_datetime(historical_df['ds'])
    future_df['ds'] = pd.to_datetime(future_df['ds'])

    # 날씨 관련 데이터 처리
    weather_features = ['temperature', 'precipitation', 'humidity', 'snowfall', 'isRainy', 'isSnow']

    # 날씨 데이터 수신 확인 및 로깅
    received_weather_features = [f for f in weather_features if f in historical_df.columns]
    print(f"수신된 날씨 특성: {received_weather_features}")

    # 수치형 날씨 데이터 처리 (temperature, precipitation, humidity, snowfall)
    for feature in ['temperature', 'precipitation', 'humidity', 'snowfall']:
        # 과거 데이터 처리
        if feature in historical_df.columns:
            # 숫자로 변환하고 NaN은 0으로 대체
            historical_df[feature] = pd.to_numeric(historical_df[feature], errors='coerce').fillna(0)
        else:
            # 없으면 0으로 채움
            historical_df[feature] = 0

        # 미래 데이터 처리
        if feature in future_df.columns:
            future_df[feature] = pd.to_numeric(future_df[feature], errors='coerce').fillna(0)
        else:
            future_df[feature] = 0

    # 이진 날씨 데이터 처리 (isRainy, isSnow)
    for feature in ['isRainy', 'isSnow']:
        # 과거 데이터 처리
        if feature in historical_df.columns:
            # 부울이나 문자열을 0/1 정수로 변환
            historical_df[feature] = historical_df[feature].astype(bool).astype(int)
        else:
            # 없으면 0으로 채움
            historical_df[feature] = 0

        # 미래 데이터 처리
        if feature in future_df.columns:
            future_df[feature] = future_df[feature].astype(bool).astype(int)
        else:
            future_df[feature] = 0

    # 특성 컬럼 확인 - 제외할 컬럼 목록
    exclude_cols = ['ds', 'y', 'holidayName', 'vacationType', 'isSchoolVacation']
    feature_cols = [col for col in historical_df.columns if col not in exclude_cols]

    # 범주형 변수 처리
    for col in feature_cols:
        if historical_df[col].dtype == 'object':
            # 원-핫 인코딩 또는 라벨 인코딩 적용
            if col not in ['ds']:
                historical_df[col] = historical_df[col].astype('category').cat.codes
                if col in future_df.columns:
                    future_df[col] = future_df[col].astype('category').cat.codes

    # 누락된 특성 확인 및 처리
    for col in feature_cols:
        if col not in future_df.columns and col != 'y':
            future_df[col] = 0

    # 데이터 형태 및 통계 로깅
    print(f"학습 데이터 크기: {historical_df.shape}, 예측 데이터 크기: {future_df.shape}")
    print(f"사용 특성 ({len(feature_cols)}개): {feature_cols}")

    # 날씨 데이터 요약 통계
    weather_stats = historical_df[received_weather_features].describe().transpose()
    print("날씨 데이터 통계:")
    print(weather_stats)

    return historical_df, future_df, feature_cols

def density_to_crowd_level(density):
    """밀도를 혼잡도 레벨로 변환"""
    if density < 0.025:
        return 1  # 여유
    elif density < 0.05:
        return 2  # 보통
    elif density < 0.3:
        return 3  # 혼잡
    else:
        return 4  # 매우 혼잡