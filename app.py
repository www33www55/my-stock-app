import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="全面技術分析儀表板", layout="wide")
st.title("🔍 全面技術分析儀表板")

st.markdown("### 📥 請輸入股票代號")
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    ticker = st.text_input("股票代號（上市加 .TW；上櫃加 .TWO；美股直接打代號）", "2330.TW")

with col_input2:
    period = st.selectbox("觀看範圍", ["6mo", "1y", "2y"], index=1)

st.markdown("---")


@st.cache_data
def load_data(symbol, p):
    try:
        df = yf.download(symbol, period=p, auto_adjust=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df.dropna()

    except Exception:
        return pd.DataFrame()


df = load_data(ticker, period)


if df is not None and not df.empty and len(df) > 60:

    # ===== 技術指標 =====
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    df["BIAS20"] = ((df["Close"] - df["MA20"]) / df["MA20"]) * 100
    df["MA20_UP"] = df["MA20"] > df["MA20"].shift(3)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["MACD_Signal"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["DIF"] - df["MACD_Signal"]
    df["MACD_Bullish"] = df["DIF"] > df["MACD_Signal"]

    # KD
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = 100 * ((df["Close"] - low_min) / (high_max - low_min))

    k_list = []
    d_list = []
    current_k = 50.0
    current_d = 50.0

    for r in rsv.fillna(50):
        current_k = (2 / 3) * current_k + (1 / 3) * r
        current_d = (2 / 3) * current_d + (1 / 3) * current_k
        k_list.append(current_k)
        d_list.append(current_d)

    df["K"] = k_list
    df["D"] = d_list
    df["KD_Cross"] = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))

    # RSI5
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rs = gain / loss
    df["RSI5"] = 100 - (100 / (1 + rs))

    # KD背離
    df["Is_Peak"] = (
        (df["K"] > df["K"].shift(1))
        & (df["K"] > df["K"].shift(-1))
        & (df["K"] > 60)
    )

    df["Is_Trough"] = (
        (df["K"] < df["K"].shift(1))
        & (df["K"] < df["K"].shift(-1))
        & (df["K"] < 40)
    )

    kd_bear_div = False
    kd_bull_div = False

    peaks = df[df["Is_Peak"]]
    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        if p2["Close"] > p1["Close"] and p2["K"] < p1["K"]:
            kd_bear_div = True

    troughs = df[df["Is_Trough"]]
    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        if t2["Close"] < t1["Close"] and t2["K"] > t1["K"]:
            kd_bull_div = True

    # N字與前高
    recent_high = df["High"].iloc[-60:-1].max()
    recent_low = df["Low"].iloc[-30:].min()
    latest = df.iloc[-1]

    distance_to_high = ((recent_high - latest["Close"]) / latest["Close"]) * 100
    breakout = latest["Close"] > recent_high
    near_breakout = 0 <= distance_to_high <= 5

    stop_loss = min(latest["MA20"], recent_low)
    target_1 = latest["Close"] + (latest["Close"] - stop_loss) * 1.5
    target_2 = latest["Close"] + (latest["Close"] - stop_loss) * 2.5

    df["HIGH_WIN_SIGNAL"] = df["MACD_Bullish"] & df["KD_Cross"]
    df["SUPER_WIN_SIGNAL"] = (
        (df["DIF"] > 0)
        & df["MACD_Bullish"]
        & df["KD_Cross"]
        & (df["Close"] > df["MA20"])
        & (df["MA20_UP"])
    )

    # ===== 評分 =====
    score = 0

    st.subheader("📋 今日技術指標大健檢表")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if (
            latest["Close"] > latest["MA20"]
            and latest["MA5"] > latest["MA10"]
            and latest["MA10"] > latest["MA20"]
            and latest["MA20"] > latest["MA60"]
        ):
            st.success("🟢 均線：標準多頭排列")
            score += 20

        elif (
            latest["Close"] > latest["MA20"]
            and latest["MA5"] > latest["MA10"]
            and latest["MA20_UP"]
        ):
            st.success("🔥 均線：起漲轉強")
            score += 15

        elif latest["Close"] > latest["MA20"]:
            st.warning("🟡 均線：站回月線")
            score += 10

        elif latest["MA5"] > latest["MA10"]:
            st.warning("🟡 均線：短線反彈")
            score += 5

        else:
            st.error("🔴 均線：空頭整理")

    with col2:
        if latest["DIF"] > 0 and latest["MACD_Bullish"]:
            st.success("🟢 MACD：主升段")
            score += 20

        elif latest["MACD_Bullish"]:
            st.warning("🟡 MACD：空頭翻揚")
            score += 10

        else:
            st.error("🔴 MACD：空頭修正")

    with col3:
        if latest["K"] > latest["D"]:
            st.success("🟢 KD：黃金交叉")
            score += 15
        else:
            st.error("🔴 KD：死亡交叉")

    with col4:
        if latest["Close"] > latest["MA60"]:
            st.success("🟢 季線：生命線之上")
            score += 15
        else:
            st.error("🔴 季線：跌破生命線")

    with col5:
        if latest["BIAS20"] > 15:
            st.error(f"🚨 乖離：過高 ({latest['BIAS20']:.1f}%)")
        elif latest["BIAS20"] < -10:
            st.success(f"🟢 乖離：跌深 ({latest['BIAS20']:.1f}%)")
            score += 10
        else:
            st.success(f"🟢 乖離：安全 ({latest['BIAS20']:.1f}%)")
            score += 10

    with col6:
        if kd_bear_div:
            st.error("🚨 背離：高檔背離")
        elif kd_bull_div:
            st.success("🔥 背離：低檔背離")
            score += 10
        else:
            st.success("🟢 背離：目前無背離")

    # N字加分
    if breakout:
        score += 10
    elif near_breakout:
        score += 5

    if latest["Volume"] > latest["VOL20"] * 1.5:
        score += 5

    score = min(score, 100)

    is_overheated = latest["RSI5"] > 85 or latest["BIAS20"] > 15

    if kd_bear_div:
        score = min(score, 35)
    elif is_overheated:
        score = min(score, 65)

    st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")

    # ===== 訊號文字 =====
    if kd_bear_div:
        st.markdown("### 🚦 今日即時訊號：🚨 **KD高檔背離，短線不適合追高，先等回檔。**")
    elif is_overheated:
        st.markdown(f"### 🚦 今日即時訊號：⚠️ **短線偏熱，RSI：{latest['RSI5']:.1f}，適合等回踩。**")
    elif latest["SUPER_WIN_SIGNAL"]:
        st.markdown("### 🚦 今日即時訊號：🌟 **主升段轉強，屬於漂亮買點。**")
    elif breakout:
        st.markdown("### 🚦 今日即時訊號：🔥 **突破前高，N字第二波啟動。**")
    elif near_breakout:
        st.markdown(f"### 🚦 今日即時訊號：🟡 **接近前高，距離突破約 {distance_to_high:.1f}%。**")
    elif latest["HIGH_WIN_SIGNAL"]:
        st.markdown("### 🚦 今日即時訊號：🔥 **MACD與KD同步轉強，可列入觀察。**")
    else:
        st.markdown("### 🚦 今日即時訊號：🍏 **目前沒有危險背離，可續觀察。**")

    # ===== 操作價位 =====
    st.subheader("📌 操作參考價")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("目前收盤價", f"{latest['Close']:.2f}")
    c2.metric("停損參考", f"{stop_loss:.2f}")
    c3.metric("第一目標", f"{target_1:.2f}")
    c4.metric("第二目標", f"{target_2:.2f}")

    st.caption("提醒：這是技術分析輔助工具，不是保證獲利訊號。")

    # ===== 圖表 =====
    st.subheader("📊 完整技術圖表")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_width=[0.15, 0.15, 0.15, 0.55],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K線",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], name="10MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="20MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="60MA"), row=1, col=1)

    normal_signals = df[df["HIGH_WIN_SIGNAL"] & ~df["SUPER_WIN_SIGNAL"]]
    fig.add_trace(
        go.Scatter(
            x=normal_signals.index,
            y=normal_signals["Low"] * 0.97,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12),
            name="🔥高勝率買點",
        ),
        row=1,
        col=1,
    )

    super_signals = df[df["SUPER_WIN_SIGNAL"]]
    fig.add_trace(
        go.Scatter(
            x=super_signals.index,
            y=super_signals["Low"] * 0.96,
            mode="markers",
            marker=dict(symbol="star", size=15),
            name="🌟漂亮買點",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K值"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D值"), row=2, col=1)

    colors = ["red" if val >= 0 else "green" for val in df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Hist"],
            marker_color=colors,
            name="MACD柱狀圖",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI5"], name="RSI5"), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=780,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("請輸入正確股票代號，例如：2330.TW、3591.TWO、AAPL")
