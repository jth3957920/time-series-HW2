import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 시계열 라이브러리 (sktime, statsmodels) ---
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import STLTransformer
from sktime.forecasting.exp_smoothing import ExponentialSmoothing as SktimeExpSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA as SktimeAutoARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller

# --- 시계열 라이브러리 (darts) ---
from darts import TimeSeries
from darts.models import ExponentialSmoothing as DartsExpSmoothing, AutoARIMA as DartsAutoARIMA, NaiveSeasonal as DartsNaiveSeasonal
from darts.metrics import mape as darts_mape

st.set_page_config(layout="wide", page_title="C221061 전태환 프로젝트", page_icon="📈")
st.header("📈 주 단위 다변량 시계열 대시보드 (C221061 전태환)")

# ==========================================
# 1. 원본 유틸리티 함수
# ==========================================
def evaluate_forecast(y_true, y_pred, y_train=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-10
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    if y_train is not None:
        naive_error = np.mean(np.abs(np.diff(np.array(y_train))))
        mase = mae / (naive_error + eps)
    else:
        mase = np.nan
    return {"MAE": mae, "MSE": mse, "MAPE(%)": mape, "MASE": mase}

def safe_to_timestamp(index):
    if hasattr(index, 'to_timestamp'):
        return index.to_timestamp()
    return index

def detect_hampel_filter(series, window_size=5, n_sigmas=3):
    k = 1.4826
    rolling_median = series.rolling(window=2*window_size+1, center=True).median()
    rolling_mad = series.rolling(window=2*window_size+1, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
    threshold = n_sigmas * k * rolling_mad
    difference = np.abs(series - rolling_median)
    outlier_indices = np.where(difference > threshold)[0]
    return outlier_indices

@st.cache_data(show_spinner=False)
def run_sktime_forecasting(y_series, test_size, sp, selected_models):
    y_train, y_test = temporal_train_test_split(y_series, test_size=test_size)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    actual_sp = max(2, sp)

    all_configs = {
        "Naive": NaiveForecaster(strategy="last"),
        "SMA": NaiveForecaster(strategy="mean"),
        "Exp Smoothing": SktimeExpSmoothing(trend=None, seasonal=None),
        "Holt-Winters": SktimeExpSmoothing(trend="add", seasonal="add", sp=actual_sp),
        "STL Forecaster": TransformedTargetForecaster(steps=[("stl", STLTransformer(sp=actual_sp)), ("forecast", NaiveForecaster(strategy="drift"))]),
        "AutoARIMA": SktimeAutoARIMA(sp=actual_sp, seasonal=(actual_sp > 1), stepwise=True, suppress_warnings=True, start_p=0, start_q=0, max_p=1, max_q=1, trace=False)
    }
    model_configs = {name: all_configs[name] for name in selected_models if name in all_configs}
    preds = {}
    for name, model in model_configs.items():
        try:
            model.fit(y_train)
            preds[name] = model.predict(fh)
        except Exception:
            pass
    return y_train, y_test, preds

# ==========================================
# 2. 데이터 업로드 및 전처리 + 달력 특성 공학
# ==========================================
st.sidebar.header("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp949')

    st.sidebar.subheader("📍 컬럼 지정")
    date_col = st.sidebar.selectbox("날짜 열", df.columns, index=0)
    
    # 💡 [핵심] 날짜 기반 주 단위 달력 변수 선제 추출 (사이드바 변수 선택창에 자동 반영되도록 함)
    temp_date = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    df['month'] = temp_date.dt.month
    df['week_of_year'] = temp_date.dt.isocalendar().week.astype(int)

    # 숫자형 컬럼 추출
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 1:
        st.error("⚠️ 시계열 분석을 위해 숫자형 컬럼이 최소 1개 이상 필요합니다.")
        st.stop()

    st.sidebar.subheader("📍 변수 역할 지정")
    target_col = st.sidebar.selectbox("🎯 예측 대상 (Target)", num_cols, index=0)
    
    # 설명 변수 선택 (추출된 month, week_of_year가 자동으로 포함됨)
    available_features = [c for c in num_cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "📊 설명 변수 (다변량 분석용)", 
        options=available_features, 
        default=[f for f in ['month', 'week_of_year'] if f in available_features]
    )
    
    freq_opt = {"주단위 (W)": "W", "일단위 (D)": "D", "월단위 (MS)": "MS", "연단위 (YS)": "YS"}
    selected_freq = st.sidebar.selectbox("데이터 주기", list(freq_opt.keys()), index=0)
    freq_str = freq_opt[selected_freq]

    # 데이터프레임 인덱스 타임스탬프화 및 정렬
    df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
        
    df = df.asfreq(freq_str)
    
    if ('current_file' not in st.session_state or 
        st.session_state.current_file != uploaded_file.name or 
        st.session_state.get('current_freq') != freq_str):
        
        st.session_state.df = df.copy()
        st.session_state.current_file = uploaded_file.name
        st.session_state.current_freq = freq_str
        st.rerun()
        
    working_df = st.session_state.df

    # ==========================================
    # 3. 데이터 정제 및 통계 분석 (ADF / ACF / PACF)
    # ==========================================
    st.subheader("🧹 데이터 정제 및 통계 분석")
    
    all_selected_cols = [target_col] + feature_cols
    clean_target = st.radio("현재 확인/정제할 변수를 선택하세요:", all_selected_cols, horizontal=True)
    current_y = working_df[clean_target]

    if current_y.isnull().any():
        st.warning(f"⚠️ '{clean_target}'에 결측치가 발견되어 선형 보간되었습니다.")
        working_df[clean_target] = current_y.interpolate(method='linear').bfill().ffill()
        current_y = working_df[clean_target]
        st.session_state.df = working_df

    c1, c2 = st.columns([1, 3])
    with c1:
        hampel_window = st.slider("Hampel Window Size", 1, 20, 5)
        hampel_sigma = st.slider("Hampel Sigma (민감도)", 1.0, 5.0, 3.0, step=0.5)
        sp_default = {'W': 52, 'D': 7, 'MS': 12, 'YS': 1}.get(freq_str, 52)
        seasonal_period = st.number_input("계절성 주기 (sp)", value=sp_default, min_value=1)
        
        if st.button("♻️ 현재 변수 원본 복구", use_container_width=True):
            st.session_state.df[clean_target] = df[clean_target]
            st.rerun()

    with c2:
        hampel_outliers = detect_hampel_filter(current_y, window_size=hampel_window, n_sigmas=hampel_sigma)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=safe_to_timestamp(current_y.index), y=current_y, name=clean_target, mode='lines+markers', marker=dict(size=5), line=dict(color="#27AE60")))
        if len(hampel_outliers) > 0:
            fig.add_trace(go.Scatter(x=safe_to_timestamp(current_y.index[hampel_outliers]), y=current_y.iloc[hampel_outliers], mode='markers', name='이상치 의심', marker=dict(color='rgba(0,0,0,0)', size=15, line=dict(color='red', width=2)), hoverinfo='skip'))
        
        selected_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", height=300)

        if selected_data and "selection" in selected_data and len(selected_data["selection"]["points"]) > 0:
            indices = sorted([p["point_index"] for p in selected_data["selection"]["points"]])
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                method = st.radio("보간 방법", ["직전 값", "다음 값", "선형 보간", "1주기 전"], horizontal=True)
                adj_win = st.slider("영향 범위", 1, 20, 1)
            with sc2:
                if st.button("✨ 적용", use_container_width=True):
                    new_y = current_y.copy()
                    for idx in indices:
                        if method == "직전 값" and idx - adj_win >= 0: new_y.iloc[idx] = new_y.iloc[idx - adj_win]
                        elif method == "다음 값" and idx + adj_win < len(new_y): new_y.iloc[idx] = new_y.iloc[idx + adj_win]
                        elif method == "선형 보간": new_y.iloc[idx] = (new_y.iloc[max(0, idx - adj_win)] + new_y.iloc[min(len(new_y) - 1, idx + adj_win)]) / 2
                        elif method == "1주기 전" and idx - (seasonal_period * adj_win) >= 0: new_y.iloc[idx] = new_y.iloc[idx - (seasonal_period * adj_win)]
                    st.session_state.df[clean_target] = new_y
                    st.rerun()

    # --- ADF / ACF / PACF 탭 전환 기능 ---
    with st.expander(f"🔍 {clean_target} 통계 분석 (ADF / ACF / PACF)"):
        valid_y = current_y.dropna()
        if len(valid_y) > 10:
            col_adf, col_corr = st.columns([1, 2])
            with col_adf:
                res = adfuller(valid_y)
                st.metric("ADF p-value", f"{res[1]:.4f}")
                if res[1] < 0.05: st.success("정상(Stationary) 데이터입니다.")
                else: st.error("비정상(Non-Stationary) 데이터입니다.")
            with col_corr:
                y_vals = valid_y.values
                n_lags = min(len(y_vals) // 4, 20)
                if n_lags < 1: n_lags = 1
                
                tab_acf, tab_pacf = st.tabs(["📊 ACF (자기상관함수)", "📊 PACF (부분자기상관함수)"])
                with tab_acf:
                    acf_vals = acf(y_vals, nlags=n_lags)
                    fig_acf = go.Figure(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF', marker_color='#2980B9'))
                    fig_acf.update_layout(height=230, margin=dict(t=10, b=10, l=0, r=0))
                    st.plotly_chart(fig_acf, use_container_width=True)
                with tab_pacf:
                    try:
                        pacf_vals = pacf(y_vals, nlags=n_lags)
                        fig_pacf = go.Figure(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF', marker_color='#E67E22'))
                        fig_pacf.update_layout(height=230, margin=dict(t=10, b=10, l=0, r=0))
                        st.plotly_chart(fig_pacf, use_container_width=True)
                    except Exception as e:
                        st.error(f"PACF 연산 실패: {e}")
        else:
            st.warning("⚠️ 데이터가 부족하여 통계 분석을 건너뜁니다.")

    st.divider()

    # ==========================================
    # 4. 분석 모델링 (sktime 단변량 vs Darts 다변량 기말고사 버전)
    # ==========================================
    st.subheader("🚀 시계열 예측 수행")
    tab1, tab2 = st.tabs(["[sktime] 단일 변수 예측", "[Darts] 다변량(외생변수) 집중 예측 및 백테스팅"])
    
    # --- [sktime] 단일 변수 예측 ---
    with tab1:
        st.markdown(f"**현재 선택된 학습 대상:** `{clean_target}`")
        t1_c1, t1_c2 = st.columns([1, 3])
        
        with t1_c1:
            sk_models = st.multiselect("모델 선택", ["Naive", "SMA", "Exp Smoothing", "Holt-Winters", "STL Forecaster", "AutoARIMA"], default=["Naive", "SMA", "Exp Smoothing", "Holt-Winters", "STL Forecaster", "AutoARIMA"])
            max_test_size = max(2, len(working_df) // 2)
            test_size = st.slider("예측 기간 (Test Size)", 1, max_test_size, min(52, max_test_size))
            run_btn1 = st.button("단일 예측 시작", type="primary", use_container_width=True)
            
        if run_btn1:
            with t1_c2: 
                if len(working_df[clean_target].dropna()) < 10:
                    st.error("🚨 데이터 개수가 너무 적어 예측을 수행할 수 없습니다.")
                else:
                    with st.spinner("단일 예측 중..."):
                        y_train, y_test, preds = run_sktime_forecasting(working_df[clean_target].dropna(), test_size, seasonal_period, sk_models)
                        if not preds:
                            st.warning("⚠️ 선택하신 예측 모델이 학습에 실패했습니다. (Test Size 조절 필요)")
                        else:
                            fig_uni = go.Figure()
                            fig_uni.add_trace(go.Scatter(x=safe_to_timestamp(y_train.index), y=y_train, name='Train', line=dict(color='#BDC3C7')))
                            fig_uni.add_trace(go.Scatter(x=safe_to_timestamp(y_test.index), y=y_test, name='Actual', line=dict(color='#2C3E50', width=2.5)))
                            for n, p in preds.items():
                                fig_uni.add_trace(go.Scatter(x=safe_to_timestamp(p.index), y=p, name=n, line=dict(dash='dash', width=2)))
                            
                            fig_uni.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_uni, use_container_width=True)
                            
                            metrics = {n: evaluate_forecast(y_test, p, y_train) for n, p in preds.items()}
                            st.dataframe(pd.DataFrame(metrics).T.style.highlight_min(axis=0, color='#D5F5E3'), use_container_width=True)

    # --- [Darts] 다변량 집중 예측 (기말고사 업그레이드 버전) ---
    with tab2:
        st.markdown(f"**전략:** `{target_col}`을 예측하기 위해 선택된 설명변수들({feature_cols})을 외생변수(Covariates)로 결합")
        t2_c1, t2_c2 = st.columns([1, 3])
        
        with t2_c1:
            darts_model_name = st.selectbox("Darts 모델", ["AutoARIMA", "Exponential Smoothing", "Naive Seasonal"])
            max_test_size2 = max(2, len(working_df) // 2)
            darts_horizon = st.slider("테스트 기간 (뒷부분 예측 개수)", 1, max_test_size2, min(52, max_test_size2))
            
            do_backtest = st.checkbox("롤링 백테스팅 추가 수행", value=False)
            if do_backtest:
                bt_start = st.slider("백테스트 시작 지점 (%)", 0.5, 0.9, 0.7, 0.05)
                bt_stride = st.number_input("스텝(Stride)", min_value=1, value=1)
            
            run_btn2 = st.button("Darts 모델 분석 시작", type="primary", use_container_width=True)
            
        if run_btn2:
            with t2_c2:
                if len(working_df[target_col].dropna()) < 15:
                    st.error("🚨 Darts 모델을 수행하기에는 데이터가 너무 적습니다.")
                else:
                    with st.spinner("Darts 모델 분석 중..."):
                        model_cls = {"Exponential Smoothing": DartsExpSmoothing, "AutoARIMA": DartsAutoARIMA, "Naive Seasonal": DartsNaiveSeasonal}[darts_model_name]
                        
                        # 1. Target 및 설명 변수들을 Darts TimeSeries 객체로 변환
                        ts_target = TimeSeries.from_series(working_df[target_col].dropna())
                        
                        if feature_cols:
                            # 💡 DataFrame일 때는 from_dataframe을 사용해야 에러가 나지 않습니다.
                            ts_features = TimeSeries.from_dataframe(working_df[feature_cols].ffill().bfill())
                        else:
                            ts_features = None
                        
                        # Train / Validation 분할
                        train_ts = ts_target[:-darts_horizon]
                        val_ts = ts_target[-darts_horizon:]
                        
                        m_target = model_cls()
                        
                        # 💡 AutoARIMA이면서 설명변수가 존재할 때만 '다변량(ARIMAX)' 구조로 학습 진행
                        if darts_model_name == "AutoARIMA" and ts_features is not None:
                            # past_covariates 대신 future_covariates로 변경!
                            m_target.fit(train_ts, future_covariates=ts_features)
                            pred_target = m_target.predict(n=darts_horizon, future_covariates=ts_features)
                            st.info("ℹ️ 요일/월 등 선택된 설명변수들을 반영한 [다변량 AutoARIMA] 모델로 학습되었습니다.")
                        else:
                            m_target.fit(train_ts)
                            pred_target = m_target.predict(n=darts_horizon)
                            if ts_features is not None:
                                st.caption("⚠️ 선택한 모델은 수학적으로 외생변수를 지원하지 않아 단변량으로 학습되었습니다. (AutoARIMA 시 다변량 활성화)")
                        
                        
                        # 결과 시각화
                        fig_multi = go.Figure()
                        fig_multi.add_trace(go.Scatter(x=train_ts.time_index, y=train_ts.values().flatten(), name='Train (학습 데이터)', line=dict(color='#BDC3C7')))
                        fig_multi.add_trace(go.Scatter(x=val_ts.time_index, y=val_ts.values().flatten(), name='Actual (실제 뒷부분)', line=dict(color='#2C3E50', width=2.5)))
                        fig_multi.add_trace(go.Scatter(x=pred_target.time_index, y=pred_target.values().flatten(), name='Prediction (예측치)', line=dict(color='#E74C3C', dash='dash', width=3)))
                        
                        fig_multi.update_layout(height=450, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_multi, use_container_width=True)

                        mape_test = darts_mape(val_ts, pred_target)
                        st.metric("📊 테스트 구간 오차율 (MAPE)", f"{mape_test:.2f}%")

                        if do_backtest:
                            st.divider()
                            st.markdown("##### 🔄 과거 데이터 기반 롤링 백테스팅 결과")
                            if darts_model_name == "AutoARIMA" and ts_features is not None:
                                bt_target = m_target.historical_forecasts(ts_target, past_covariates=ts_features, start=bt_start, forecast_horizon=1, stride=bt_stride, retrain=True)
                            else:
                                bt_target = m_target.historical_forecasts(ts_target, start=bt_start, forecast_horizon=1, stride=bt_stride, retrain=True)
                            
                            intersect_actual = ts_target.slice_intersect(bt_target)
                            mape_bt = darts_mape(intersect_actual, bt_target)
                            st.metric("백테스팅 오차율 (MAPE)", f"{mape_bt:.2f}%")
else:
    st.info("👈 왼쪽 사이드바에서 시계열 CSV 파일을 업로드해주세요.")
