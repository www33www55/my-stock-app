import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="技術分析", layout="wide")

st.title("🚀 ｜台股技術分析 App")
st.write("突破、MACD、KD背離、RSI、量能、目標價一次看懂")

# =====================
# 工具函數
# =====================

def fix_symbol(symbol):
    symbol = symbol.strip()
    if symbol.isdigit():
        return symbol + ".TW"
    return symbol

def get_data(symbol):
    data = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_kd(df, period=9):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()

    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def star(score):
    if score >= 90:
        return "★★★★★ 超強勢"
    elif score >= 80:
        return "★★★★☆ 強勢"
    elif score >= 70:
        return "★★★☆☆ 可觀察"
    elif score >= 60:
        return "★★☆☆☆ 普通"
    else:
        return "★☆☆☆☆ 風險高"

def detect_kd_divergence(df):
    recent = df.tail(30)

    price_high_now = recent["Close"].iloc[-1]
    price_high_before = recent["Close"].iloc[:-10].max()

    kd_now = recent["K"].iloc[-1]
    kd_before = recent["K"].iloc[:-10].max()

    price_low_now = recent["Close"].iloc[-1]
    price_low_before = recent["Close"].iloc[:-10].min()

    kd_low_now = recent["K"].iloc[-1]
    kd_low_before = recent["K"].iloc[:-10].min()

    if price_high_now > price_high_before and kd_now < kd_before:
        return "🔴 明顯頂背離：股價創高，但KD沒有跟著創高，追高要小心"
    elif price_low_now < price_low_before and kd_low_now > kd_low_before:
        return "🟢 底背離：股價創低，但KD沒有更低，有機會止跌轉強"
    else:
        return "🟢 無明顯KD背離：目前走勢還算健康"

def analyze(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = last["Close"]
    ma5 = last["MA5"]
    ma10 = last["MA10"]
    ma20 = last["MA20"]
    ma60 = last["MA60"]

    score = 0
    comments = []

    # 均線
    if close > ma5 > ma10 > ma20:
        score += 15
        comments.append("🟢 均線多頭排列，短線趨勢強")
    elif close > ma20:
        score += 8
        comments.append("🟡 股價站上月線，偏多但還不是最強")
    else:
        comments.append("🔴 股價跌破月線，短線偏弱")

    # 季線
    if close > ma60:
        score += 10
        comments.append("🟢 股價站上季線，中線偏多")
    else:
        comments.append("🔴 股價還沒站上季線，中線壓力仍在")

    # 成交量
    if last["Volume"] > last["VOL20"] * 1.8:
        score += 18
        comments.append("🟢 成交量明顯放大，有主力發動味道")
    elif last["Volume"] > last["VOL20"] * 1.2:
        score += 12
        comments.append("🟡 成交量有放大，買氣有增加")
    else:
        score += 5
        comments.append("🟡 量能普通，還不是強攻狀態")

    # MACD
    if last["MACD"] > last["SIGNAL"] and last["MACD"] > 0 and last["HIST"] > prev["HIST"]:
        score += 20
        macd_text = "🟢 MACD主升段啟動：0軸上黃金交叉，動能正在變強"
    elif last["MACD"] > last["SIGNAL"]:
        score += 12
        macd_text = "🟡 MACD黃金交叉：有轉強，但還不是最強主升段"
    elif last["HIST"] < prev["HIST"]:
        score += 3
        macd_text = "🔴 MACD動能變弱：柱狀體縮小，要小心拉回"
    else:
        score += 5
        macd_text = "🟡 MACD普通：方向還不夠明確"

    # KD
    if last["K"] > last["D"] and last["K"] < 80:
        score += 10
        kd_text = "🟢 KD黃金交叉，還沒過熱，健康"
    elif last["K"] > 80:
        score += 5
        kd_text = "🟡 KD超過80，偏熱，追高要小心"
    elif last["K"] < last["D"]:
        score += 2
        kd_text = "🔴 KD死亡交叉，短線轉弱"
    else:
        score += 5
        kd_text = "🟡 KD普通"

    # RSI
    if last["RSI"] > 90:
        rsi_text = "🔴 RSI超過90，極度過熱，不建議追高"
    elif last["RSI"] > 80:
        rsi_text = "🟡 RSI超過80，偏熱，適合等回踩"
    elif last["RSI"] < 30:
        rsi_text = "🟢 RSI低檔，有止跌機會"
    else:
        rsi_text = "🟢 RSI正常，沒有過熱"

    # 型態
    recent_high = df["Close"].tail(30).max()
    old_high = df["Close"].tail(60).iloc[:-30].max()

    if close >= recent_high * 0.98 and close > old_high:
        score += 20
        pattern_text = "🟢 突破前高，可能進入N字第二波"
    elif close > old_high:
        score += 15
        pattern_text = "🟢 已突破平台，型態偏多"
    elif close > ma20 and close > ma60:
        score += 8
        pattern_text = "🟡 還在整理區，等突破會更漂亮"
    else:
        score += 3
        pattern_text = "🔴 型態還沒突破，先觀察"

    kd_div = detect_kd_divergence(df)

    score = min(score, 100)

    # 目標價
    low_30 = df["Low"].tail(30).min()
    high_30 = df["High"].tail(30).max()
    wave = high_30 - low_30

    stop_loss = round(close * 0.94, 2)
    target1 = round(close + wave * 0.5, 2)
    target2 = round(close + wave, 2)

    if score >= 80 and "頂背離" not in kd_div:
        conclusion = "🟢 可以觀察買進 / 已持有可續抱"
    elif score >= 70:
        conclusion = "🟡 等回踩比較安全"
    else:
        conclusion = "🔴 不建議追價"

    summary = f"目前股價趨勢評分 {score} 分，{macd_text}，{kd_div}，整體屬於：{conclusion}"

    return {
        "score": score,
        "star": star(score),
        "comments": comments,
        "macd": macd_text,
        "kd": kd_text,
        "rsi": rsi_text,
        "pattern": pattern_text,
        "kd_div": kd_div,
        "conclusion": conclusion,
        "summary": summary,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
    }

# =====================
# 介面
# =====================

symbol_input = st.text_input("請輸入股票代號，例如 台股:2330、上櫃:7828.TWO)
symbol = fix_symbol(symbol_input)

if st.button("開始分析"):
    df = get_data(symbol)

    if df.empty:
        st.error("抓不到資料，請確認股票代號是否正確")
    else:
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["VOL20"] = df["Volume"].rolling(20).mean()

        df["RSI"] = calc_rsi(df["Close"])
        df["K"], df["D"] = calc_kd(df)
        df["MACD"], df["SIGNAL"], df["HIST"] = calc_macd(df["Close"])

        df = df.dropna()

        result = analyze(df)

        st.subheader(f"📌 {symbol} 分析結果")

        col1, col2, col3 = st.columns(3)
        col1.metric("綜合評分", f"{result['score']} 分")
        col2.metric("強弱等級", result["star"])
        col3.metric("最新收盤價", round(df["Close"].iloc[-1], 2))

        st.divider()

        st.subheader("📈 K線圖")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K線"
            ),
            row=1,
            col=1
        )

        fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], name="MA10"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="MA60"), row=1, col=1)

        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="成交量"),
            row=2,
            col=1
        )

        fig.update_layout(height=700, xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🧠 白話分析")

        for c in result["comments"]:
            st.write(c)

        st.info(result["macd"])
        st.info(result["kd"])
        st.info(result["rsi"])
        st.warning(result["kd_div"])
        st.success(result["pattern"])

        st.divider()

        st.subheader("🎯 操作參考")

        c1, c2, c3 = st.columns(3)
        c1.metric("停損參考", result["stop_loss"])
        c2.metric("第一目標價", result["target1"])
        c3.metric("第二目標價", result["target2"])

        st.subheader("✅ 最後結論")
        st.success(result["conclusion"])
        st.write(result["summary"])
