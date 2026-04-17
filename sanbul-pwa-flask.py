"""
Project #1: AI기반 산불 예측 서비스
단계 3: Flask Web App (PWA)

사전 준비: sanbul_step1_2.py 를 먼저 실행하여
  - fires_model.keras
  - full_pipeline.pkl
를 생성해야 합니다.

실행: python sanbul-pwa-flask.py
"""

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:     ", tf.__version__)

import numpy as np
import pandas as pd
import joblib
import os

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, ValidationError

# 유효한 month / day 값 (슬라이드 데이터 기준)
VALID_MONTHS = ['01-Jan','02-Feb','03-Mar','04-Apr','05-May','06-Jun',
                '07-Jul','08-Aug','09-Sep','10-Oct','11-Nov','12-Dec']
VALID_DAYS   = ['00-sun','01-mon','02-tue','03-wed','04-thu','05-fri','06-sat','07-hol']

def validate_month(form, field):
    if field.data.strip() not in VALID_MONTHS:
        raise ValidationError(f"올바른 월 형식 아님. 예: 03-Mar (선택: {', '.join(VALID_MONTHS)})")

def validate_day(form, field):
    if field.data.strip() not in VALID_DAYS:
        raise ValidationError(f"올바른 요일 형식 아님. 예: 06-sat (선택: {', '.join(VALID_DAYS)})")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit

# ──────────────────────────────────────────────
#  Flask 앱 초기화
# ──────────────────────────────────────────────
np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)


# ──────────────────────────────────────────────
#  WTForms 입력 폼 정의
# ──────────────────────────────────────────────
class LabForm(FlaskForm):
    longitude      = StringField('longitude(1-7)',               validators=[DataRequired()])
    latitude       = StringField('latitude(1-7)',                validators=[DataRequired()])
    month          = StringField('month(01-Jan ~ Dec-12)',       validators=[DataRequired(), validate_month])
    day            = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired(), validate_day])
    avg_temp       = StringField('avg_temp',                     validators=[DataRequired()])
    max_temp       = StringField('max_temp',                     validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed',               validators=[DataRequired()])
    avg_wind       = StringField('avg_wind',                     validators=[DataRequired()])
    submit         = SubmitField('Submit')


# ──────────────────────────────────────────────
#  앱 시작 시 파이프라인 & 모델 로드
#  (full_pipeline.pkl 이 없으면 CSV에서 재학습)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, 'sanbul2district-divby100.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'fires_model.keras')
PIPE_PATH  = os.path.join(BASE_DIR, 'full_pipeline.pkl')

NUM_ATTRIBS = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
CAT_ATTRIBS = ['month', 'day']


def build_pipeline(csv_path):
    """CSV 데이터를 읽어 파이프라인을 학습하고 반환합니다."""
    fires = pd.read_csv(csv_path, sep=",")
    fires['burned_area'] = np.log(fires['burned_area'] + 1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, _ in split.split(fires, fires["month"]):
        strat_train_set = fires.loc[train_idx]

    fires_train_data = strat_train_set.drop(["burned_area"], axis=1)

    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, NUM_ATTRIBS),
        ("cat", OneHotEncoder(handle_unknown='ignore'), CAT_ATTRIBS),
    ])
    full_pipeline.fit(fires_train_data)
    return full_pipeline


# 파이프라인 로드 or 생성
if os.path.exists(PIPE_PATH):
    print("파이프라인 로드:", PIPE_PATH)
    full_pipeline = joblib.load(PIPE_PATH)
elif os.path.exists(CSV_PATH):
    print("파이프라인을 CSV에서 새로 학습합니다.")
    full_pipeline = build_pipeline(CSV_PATH)
    joblib.dump(full_pipeline, PIPE_PATH)
else:
    full_pipeline = None
    print("경고: full_pipeline.pkl 과 CSV 파일이 모두 없습니다.")

# 모델 로드
if os.path.exists(MODEL_PATH):
    print("모델 로드:", MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH)
else:
    model = None
    print("경고: fires_model.keras 파일이 없습니다. 먼저 sanbul_step1_2.py 를 실행하세요.")


# ──────────────────────────────────────────────
#  라우트 정의
# ──────────────────────────────────────────────
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()

    if form.validate_on_submit():
        # ① 숫자 변환 (WTForms 검증 통과 후이므로 형식 에러는 여기서만 처리)
        try:
            longitude      = float(form.longitude.data)
            latitude       = float(form.latitude.data)
            avg_temp       = float(form.avg_temp.data)
            max_temp       = float(form.max_temp.data)
            max_wind_speed = float(form.max_wind_speed.data)
            avg_wind       = float(form.avg_wind.data)
        except ValueError as e:
            form.longitude.errors.append(f"숫자 입력 오류: {e}")
            return render_template('prediction.html', form=form)

        month = form.month.data.strip()
        day   = form.day.data.strip()

        # longitude / latitude 범위 검사 (1-7)
        if not (1 <= longitude <= 7) or not (1 <= latitude <= 7):
            form.longitude.errors.append("longitude / latitude 값은 1~7 사이여야 합니다.")
            return render_template('prediction.html', form=form)

        # ② DataFrame 구성
        input_data = pd.DataFrame({
            'longitude':      [longitude],
            'latitude':       [latitude],
            'month':          [month],
            'day':            [day],
            'avg_temp':       [avg_temp],
            'max_temp':       [max_temp],
            'max_wind_speed': [max_wind_speed],
            'avg_wind':       [avg_wind],
        })

        # ③ 모델/파이프라인 존재 확인
        if full_pipeline is None or model is None:
            return render_template('result.html',
                                   burned_area="모델/파이프라인 파일 없음",
                                   error=True)

        # ④ 전처리 → 예측 → 역변환
        input_prepared = full_pipeline.transform(input_data)
        log_pred    = model.predict(input_prepared, verbose=0)[0][0]
        burned_area = round(float(np.exp(log_pred) - 1), 2)
        burned_area = max(burned_area, 0.0)   # 음수 방지

        return render_template('result.html', burned_area=burned_area,
                               inputs=dict(longitude=int(longitude), latitude=int(latitude),
                                           month=month, day=day, avg_temp=avg_temp,
                                           max_temp=max_temp, max_wind_speed=max_wind_speed,
                                           avg_wind=avg_wind))

    return render_template('prediction.html', form=form)


# ──────────────────────────────────────────────
#  실행
# ──────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
