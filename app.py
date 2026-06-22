import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="未來小股神4.0", layout="wide")

st.title("🚀 未來小股神 4.0")

stock = st.text_input("股票代號", "2330.TW")

if st.button("開始分析"):

    df = yf.download(stock, period="1y", auto_adjust=True)

    if df.empty:
        st.error("找不到資料")
        st.stop()

    close = df["Close"]

    # ===== MACD =====
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    dif = ema12 - ema26
    macd = dif.ewm(span=9).mean()
    hist = dif - macd

    # ===== KD =====
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()

    rsv = (close - low9) / (high9 - low9) * 100

    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()

    # ===== RSI =====
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    # ===== 均線 =====
    ma20 = close.rolling(20).mean()

    score = 0

    # MACD
    if dif.iloc[-1] > macd.iloc[-1]:
        score += 25

    # KD
    if k.iloc[-1] > d.iloc[-1]:
        score += 15

    # RSI
    if 40 <= rsi.iloc[-1] <= 80:
        score += 10

    # 均線
    if close.iloc[-1] > ma20.iloc[-1]:
        score += 20

    # 量能
    vol20 = df["Volume"].rolling(20).mean()

    if df["Volume"].iloc[-1] > vol20.iloc[-1]:
        score += 15

    # 型態
    recent_high = close.tail(60).max()

    if close.iloc[-1] > recent_high * 0.95:
        score += 15

    # 星級
    if score >= 90:
        star = "★★★★★"
    elif score >= 80:
        star = "★★★★☆"
    elif score >= 70:
        star = "★★★☆☆"
    elif score >= 60:
        star = "★★☆☆☆"
    else:
        star = "★☆☆☆☆"

    st.subheader("AI評分")

    col1, col2, col3 = st.columns(3)

    col1.metric("總分", score)
    col2.metric("星級", star)
    col3.metric("股價", round(float(close.iloc[-1]), 2))

    st.write("---")

    st.write("### 技術指標")

    st.write(f"MACD DIF：{float(dif.iloc[-1]):.2f}")
    st.write(f"MACD Signal：{float(macd.iloc[-1]):.2f}")
    st.write(f"K值：{float(k.iloc[-1]):.2f}")
    st.write(f"D值：{float(d.iloc[-1]):.2f}")
    st.write(f"RSI：{float(rsi.iloc[-1]):.2f}")

    if dif.iloc[-1] > macd.iloc[-1]:
        st.success("MACD黃金交叉偏多")

    if k.iloc[-1] > d.iloc[-1]:
        st.success("KD黃金交叉")

    if rsi.iloc[-1] > 80:
        st.warning("RSI超買")

    if rsi.iloc[-1] < 20:
        st.warning("RSI超賣")

    target = close.iloc[-1] * 1.10
    stop = close.iloc[-1] * 0.95

    st.write("---")

    st.subheader("AI分析")

    st.write(f"目標價：{target:.2f}")
    st.write(f"停損價：{stop:.2f}")

    # ===== K線 =====
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K線"
        )
    )

    fig.update_layout(
        title=f"{stock} K線圖",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
