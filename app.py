streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="未來小股神 v3.0", layout="wide")
st.title("🚀 未來小股神 v3.0 技術選股儀表板")

st.markdown("### 📥 請輸入股票代號")
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    ticker = st.text_input("股票代號：上市加 .TW；上櫃加 .TWO；美股直接打代號", "2330.TW")

with col_input2:
    period = st.selectbox("觀看範圍", ["6mo", "1y", "2y"], index=1)

st.markdown("---"


@st.cache_data(ttl=3600)
def load_data(symbol, p):
    try:
        df = yf.download(symbol, period=p, auto_adjust=False, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        return df

    except Exception:
        return


df = load_data(ticker, period)

if df is not None and not df.empty and len(df) > 80:

    # =========================
    # 1. 技術指標
    # =========================

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    df["MA20_UP"] = df["MA20"] > df["MA20"].shift(3)
    df["MA60_UP"] = df["MA60"] > df["MA60"].shift(5)

    df["BIAS20"] = ((df["Close"] - df["MA20"]) / df["MA20"]) * 100

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["DIF"] = ema12 - ema26
    df["MACD_Signal"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["DIF"] - df["MACD_Signal"]
    df["MACD_Bullish"] = df["DIF"] > df["MACD_Signal"]
    df["MACD_Main"] = (df["DIF"] > 0) & df["MACD_Bullish"]

    # KD
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    rsv = rsv.replace([float("inf"), -float("inf")], 50).fillna(50)

    k_list = []
    d_list = []
    current_k = 50.0
    current_d = 50.0

    for r in rsv:
        current_k = (2 / 3) * current_k + (1 / 3) * r
        current_d = (2 / 3) * current_d + (1 / 3) * current_k
        k_list.append(current_k)
        d_list.append(current_d)

    df["K"] = k_list
    df["D"] = d_list
    df["KD_Golden"] = df["K"] > df["D"]
    df["KD_Cross_Today"] = (df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))

    # RSI：券商常用 Wilder RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 5, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 5, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI5"] = 100 - (100 / (1 + rs))
    df["RSI5"] = df["RSI5"].fillna(50)

    latest = df.iloc[-1]

    # =========================
    # 2. 型態偵測
    # =========================

    recent_high_60 = df["High"].iloc[-60:-1].max()
    recent_low_30 = df["Low"].iloc[-30:].min()

    distance_to_high = ((recent_high_60 - latest["Close"]) / latest["Close"]) * 100

    breakout = latest["Close"] > recent_high_60
    near_breakout = 0 <= distance_to_high <= 5

    pullback_ma20 = (
        latest["Close"] > latest["MA20"]
        and df["Low"].iloc[-1] <= latest["MA20"] * 1.02
        and latest["MA20_UP"]
    )

    volume_break = latest["Volume"] > latest["VOL20"] * 1.5

    fake_breakout = (
        latest["High"] > recent_high_60
        and latest["Close"] < recent_high_60
        and latest["Close"] < latest["Open"]
        and volume_break
    )

    # KD 背離
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
        p1 = peaks.iloc[-2]
        p2 = peaks.iloc[-1]

        if p2["Close"] > p1["Close"] and p2["K"] < p1["K"]:
            kd_bear_div = True

    troughs = df[df["Is_Trough"]]
    if len(troughs) >= 2:
        t1 = troughs.iloc[-2]
        t2 = troughs.iloc[-1]

        if t2["Close"] < t1["Close"] and t2["K"] > t1["K"]:
            kd_bull_div = True

    # =========================
    # 3. 支撐、停損、目標
    # =========================

    support_1 = latest["MA5"]
    support_2 = latest["MA10"]
    support_3 = latest["MA20"]

    stop_loss = min(latest["MA20"], recent_low_30)

    risk = latest["Close"] - stop_loss

    if risk <= 0:
        target_1 = latest["Close"] * 1.05
        target_2 = latest["Close"] * 1.10
    else:
        target_1 = latest["Close"] + risk * 1.5
        target_2 = latest["Close"] + risk * 2.5

    # =========================
    # 4. 未來小股神評分
    # =========================

    score = 0
    reasons_good = []
    reasons_bad = []

    # 趨勢 30
    if latest["Close"] > latest["MA20"]:
        score += 8
        reasons_good.append("股價站上月線")
    else:
        reasons_bad.append("股價尚未站上月線")

    if latest["MA20_UP"]:
        score += 8
        reasons_good.append("月線上彎")
    else:
        reasons_bad.append("月線尚未上彎")

    if latest["Close"] > latest["MA60"]:
        score += 7
        reasons_good.append("股價站上季線")
    else:
        reasons_bad.append("股價跌破季線")

    if latest["MA5"] > latest["MA10"] > latest["MA20"]:
        score += 7
        reasons_good.append("短中期均線多頭排列")
    else:
        reasons_bad.append("均線尚未完整多頭排列")

    # 動能 25
    if latest["MACD_Main"]:
        score += 10
        reasons_good.append("MACD主升段")
    elif latest["MACD_Bullish"]:
        score += 5
        reasons_good.append("MACD空頭翻揚")
    else:
        reasons_bad.append("MACD仍偏弱")

    if latest["KD_Golden"]:
        score += 8
        reasons_good.append("KD黃金交叉")
    else:
        reasons_bad.append("KD死亡交叉")

    if 45 <= latest["RSI5"] <= 75:
        score += 7
        reasons_good.append("RSI處於健康強勢區")
    elif latest["RSI5"] > 85:
        reasons_bad.append("RSI過熱")
    elif latest["RSI5"] < 35:
        reasons_bad.append("RSI偏弱")

    # 型態 25
    if breakout:
        score += 10
        reasons_good.append("突破前高，N字第二波啟動")
    elif near_breakout:
        score += 7
        reasons_good.append("接近前高，準備挑戰突破")

    if pullback_ma20:
        score += 8
        reasons_good.append("回踩月線不破")
    else:
        reasons_bad.append("尚未出現漂亮回踩月線")

    if volume_break:
        score += 7
        reasons_good.append("爆量轉強")
    else:
        reasons_bad.append("量能尚未明顯放大")

    # 風險修正
    if kd_bull_div:
        score += 5
        reasons_good.append("KD低檔背離")

    if fake_breakout:
        score = min(score, 45)
        reasons_bad.append("疑似假突破")

    if kd_bear_div:
        score = min(score, 40)
        reasons_bad.append("KD高檔背離")

    if latest["BIAS20"] > 15:
        score = min(score, 70)
        reasons_bad.append("乖離過大")

    if latest["RSI5"] > 85:
        score = min(score, 70)
        reasons_bad.append("短線過熱")

    score = min(score, 100)

    # =========================
    # 5. 等級
    # =========================

    if score >= 90:
        grade = "🌟 神級漂亮股"
        action = "可以列入優先觀察，等回踩不破或突破量確認。"
    elif score >= 80:
        grade = "🔥 強勢股"
        action = "型態不錯，但不要追高，等5MA或10MA回踩。"
    elif score >= 70:
        grade = "🟡 可觀察"
        action = "有轉強跡象，但還沒到最漂亮買點。"
    elif score >= 55:
        grade = "🟠 普通觀察"
        action = "可以放清單，但先不要急著進。"
    else:
        grade = "🔴 暫時放棄"
        action = "型態不夠漂亮，先找更強的股票。"

    # =========================
    # 6. 顯示結果
    # =========================

    st.subheader("🎯 未來小股神總評")

    c1, c2, c3 = st.columns(3)

    c1.metric("總分", f"{score} / 100")
    c2.metric("評級", grade)
    c3.metric("RSI5", f"{latest['RSI5']:.1f}")

    st.markdown(f"### 🧠 人話結論：{action}")

    if fake_breakout:
        st.error("🚨 假突破警報：有爆量衝高但收不回前高，短線小心。")
    elif kd_bear_div:
        st.error("🚨 KD高檔背離：不適合追高，等回檔。")
    elif breakout:
        st.success("🔥 N字第二波：已突破前高，攻擊訊號成立。")
    elif near_breakout:
        st.warning(f"🟡 前高壓力：距離突破前高約 {distance_to_high:.1f}%。")
    elif pullback_ma20:
        st.success("🟢 回踩月線不破：這是你喜歡的觀察型態。")
    else:
        st.info("🍏 目前沒有危險訊號，但也還不是最強攻擊點。")

    # =========================
    # 7. 技術健檢卡
    # =========================

    st.subheader("📋 技術健檢卡")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if latest["MA5"] > latest["MA10"] > latest["MA20"]:
            st.success("🟢 均線：多頭")
        elif latest["Close"] > latest["MA20"]:
            st.warning("🟡 均線：站月線")
        else:
            st.error("🔴 均線：偏弱")

    with col2:
        if latest["MACD_Main"]:
            st.success("🟢 MACD：主升段")
        elif latest["MACD_Bullish"]:
            st
