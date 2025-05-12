from prophet import Prophet
import pandas as pd


def train_prophet_model(df):
    """Prophet 모델 학습 - 조건부 계절성 없이"""
    # 기본 데이터 준비
    prophet_df = df[['ds', 'y']].copy()

    # Prophet 모델 설정
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    # 월별 계절성 추가
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 모델 학습
    model.fit(prophet_df)

    return model

# def train_prophet_model(df):
#     """Prophet 모델 학습"""
#     # 기본 데이터 준비
#     prophet_df = df[['ds', 'y']].copy()
#
#     # 휴일 준비
#     holidays = None
#     if 'isHoliday' in df.columns and 'holidayName' in df.columns:
#         holiday_df = df[df['isHoliday'] == 1][['ds', 'holidayName']].copy()
#         if not holiday_df.empty:
#             holiday_df.columns = ['ds', 'holiday']
#             holidays = holiday_df
#
#     # Prophet 모델 설정
#     model = Prophet(
#         yearly_seasonality=True,
#         weekly_seasonality=True,
#         daily_seasonality=False,
#         holidays=holidays,
#         seasonality_mode='multiplicative'  # 관광지에 적합한 모드
#     )
#
#     # 월별 계절성 추가
#     model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
#
#     # 방학 시즌 추가
#     if 'isSchoolVacation' in df.columns:
#         vacation_days = df[df['isSchoolVacation'] == 1]['ds'].dt.dayofyear.tolist()
#         if vacation_days:
#             model.add_seasonality(
#                 name='vacation_season',
#                 period=365.25,
#                 fourier_order=10,
#                 condition_name='is_vacation_season'
#             )
#             prophet_df['is_vacation_season'] = df['isSchoolVacation']
#
#     # 모델 학습
#     model.fit(prophet_df)
#
#     return model

def predict_with_prophet(model, historical_dates, future_dates):
    """Prophet으로 예측 수행"""
    # 과거 데이터 예측
    prophet_historical = model.predict(historical_dates)

    # 미래 데이터 예측
    prophet_future = model.predict(future_dates)

    return prophet_historical, prophet_future