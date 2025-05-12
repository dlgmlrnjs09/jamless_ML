import pandas as pd

def prepare_data(historical_data, future_dates):
    """데이터 전처리 및 특성 준비"""
    # DataFrame 변환
    historical_df = pd.DataFrame(historical_data)
    future_df = pd.DataFrame(future_dates)

    # 날짜 형식 변환
    historical_df['ds'] = pd.to_datetime(historical_df['ds'])
    future_df['ds'] = pd.to_datetime(future_df['ds'])

    # 특성 컬럼 확인
    feature_cols = [col for col in historical_df.columns
                    if col not in ['ds', 'y', 'holidayName', 'vacationType']]

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