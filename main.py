import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {', '.join(missing)}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 기간: 2025년까지
    # 1908년을 회귀의 기준 연도로 사용하므로 1908년 이후만 회귀에 포함
    df = df[(df["연도"] >= 1908) & (df["연도"] <= 2025)].copy()

    # 연도별 유효 관측일 수가 300일 이상인 해만 사용
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly["연도"] = yearly["연도"].astype(int)
    yearly["연평균기온"] = yearly["연평균기온"].astype(float)

    return yearly.sort_values("연도").reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온을 이용해 연도별 기온의 추세를 살펴보고, 회귀 직선을 이용해 선택한 연도의 예상 기온을 확인합니다.")

try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 문제가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀분석을 수행할 만큼 충분한 연도 데이터가 없습니다.")
    st.stop()

# 독립 변수: 1908년부터 지난 연수
x = yearly["연도"].to_numpy(dtype=float) - 1908.0
y = yearly["연평균기온"].to_numpy(dtype=float)

# 전체 기간 회귀: 1908년부터 지난 연수(연도 - 1908)를 독립 변수로 사용
slope, intercept = np.polyfit(x, y, 1)
corr = np.corrcoef(x, y)[0, 1]

# "100년에 몇 도 오르는가"로 환산
slope_100 = slope * 100

# 최근 20년: 사용 가능한 데이터의 마지막 20개 연도
recent20 = yearly.tail(20).copy()
recent_x = recent20["연도"].to_numpy(dtype=float) - 1908.0
recent_y = recent20["연평균기온"].to_numpy(dtype=float)
recent_slope, recent_intercept = np.polyfit(recent_x, recent_y, 1)
recent_slope_100 = recent_slope * 100

# 예측용 연도 범위
prediction_years = np.arange(1900, 2101)
prediction_x = prediction_years - 1908
prediction_temps = intercept + slope * prediction_x

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - 1908
predicted_temp = intercept + slope * selected_x

st.subheader("기온 변화 속도")

slope_col1, slope_col2 = st.columns(2)
with slope_col1:
    st.metric(
        "전체 기간: 100년에",
        f"{slope_100:+.2f} °C",
        border=True,
        help=f"{yearly['연도'].min()}~{yearly['연도'].max()}년 자료의 회귀 기울기를 100년 단위로 환산했습니다.",
    )
with slope_col2:
    st.metric(
        f"최근 20년 ({recent20['연도'].min()}~{recent20['연도'].max()}년): 100년에",
        f"{recent_slope_100:+.2f} °C",
        border=True,
        help="사용 가능한 연도 데이터의 마지막 20개 연도를 대상으로 계산했습니다.",
    )

st.markdown(
    f"**비교:** 전체 기간의 연간 기울기 {slope:+.4f} °C/년 → "
    f"100년당 {slope_100:+.2f} °C, "
    f"최근 20년의 연간 기울기 {recent_slope:+.4f} °C/년 → "
    f"100년당 {recent_slope_100:+.2f} °C"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("선택 연도", f"{selected_year}년")
with col2:
    st.metric("예상 연평균기온", f"{predicted_temp:.2f} °C")
with col3:
    st.metric("상관계수", f"{corr:.3f}")

st.subheader("연평균기온과 회귀 직선")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        hovertemplate="%{x}년<br>연평균기온: %{y:.2f} °C<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate="%{x}년<br>전체 기간 회귀값: %{y:.2f} °C<extra></extra>",
    )
)

recent_prediction = recent_intercept + recent_slope * prediction_x
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=recent_prediction,
        mode="lines",
        name="최근 20년 회귀 직선",
        line=dict(width=2, dash="dash"),
        hovertemplate="%{x}년<br>최근 20년 회귀값: %{y:.2f} °C<extra></extra>",
    )
)

# 선택한 연도의 예측값을 그래프에서도 강조
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=13, symbol="diamond"),
        hovertemplate=f"{selected_year}년<br>예상 연평균기온: {predicted_temp:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        tickformat="d",
    ),
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=560,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, width="stretch")

st.subheader("회귀에 사용한 데이터")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("사용한 연도 수", f"{len(yearly)}개")
with c2:
    st.metric("시작 연도", f"{yearly['연도'].min()}년")
with c3:
    st.metric("끝 연도", f"{yearly['연도'].max()}년")

st.caption(
    f"전체 기간 회귀식: 연평균기온 = {intercept:.4f} + {slope:.4f} × (연도 - 1908)  |  "
    f"최근 20년 회귀식: 연평균기온 = {recent_intercept:.4f} + {recent_slope:.4f} × (연도 - 1908)"
)

with st.expander("계산에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온 (°C)",
                "관측일수": "관측일수",
            }
        ),
        hide_index=True,
        width="stretch",
    )
