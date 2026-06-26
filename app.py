import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="未來小股神 AI 選股系統 V20 PRO", layout="wide")

FAITH_STOCK = "7828"
DEFAULT_POOL = [
    "7828", "1714", "2409", "2303", "6271", "6191", "3557", "3037", "2382",
    "2313", "2344", "2359", "3060", "8923", "6272", "5468", "6259"
]

def tw_symbol(code: str) -> str:
    code = str(code).strip()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code
    return f"{code}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def load_price(code: str, period="6mo"):
    # TW first, if empty try TWO
    for suffix in [".TW", ".TWO"]:
        ticker = code if code.endswith((".TW", ".TWO")) else f"{code}{suffix}"
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if df is not None and len(df) > 40:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df["Ticker"] = ticker
                return df
        except Exception:
            pass
    return pd.DataFrame()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def estimate_launch_time(score, dist_pct, rsi_v, volume_ratio, trend_ok):
    """預估發動時間：越接近突破、量能健康、分數高，時間越短。"""
    if not trend_ok or score < 60:
        return "未成熟"
    if dist_pct <= 1.0 and score >= 88 and 50 <= rsi_v <= 75 and volume_ratio >= 1.1:
        return "1～3天"
    if dist_pct <= 2.5 and score >= 78:
        return "2～5天"
    if dist_pct <= 4.0 and score >= 70:
        return "5～10天"
    return "觀察中"

def risk_level(rsi_v, dist_ma20, volume_ratio):
    if rsi_v >= 82 or dist_ma20 > 18:
        return "🔴 高"
    if rsi_v >= 75 or dist_ma20 > 12 or volume_ratio > 2.8:
        return "🟡 中"
    return "🟢 低"

def analyze(code: str):
    df = load_price(code)
    if df.empty:
        return None, None
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["RSI"] = rsi(close)
    df["DIF"], df["DEA"], df["MACD_H"] = macd(close)
    df["Vol20"] = vol.rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = high.tail(20).max()
    platform_low = low.tail(20).min()
    price = float(last["Close"])
    ma5, ma10, ma20, ma60 = [float(last[x]) for x in ["MA5", "MA10", "MA20", "MA60"]]
    rsi_v = float(last["RSI"]) if pd.notna(last["RSI"]) else 50
    vol_ratio = float(last["Volume"] / last["Vol20"]) if last["Vol20"] else 1
    dist_pct = max(0, float((recent_high - price) / price * 100))
    dist_ma20 = float((price - ma20) / ma20 * 100) if ma20 else 0
    trend_ok = ma5 > ma10 > ma20 and price >= ma20
    macd_ok = last["DIF"] > last["DEA"] and last["DIF"] > 0
    vol_ok = vol_ratio >= 1.1
    platform_ok = (recent_high - platform_low) / price <= 0.18

    score = 0
    score += 24 if trend_ok else 0
    score += 18 if macd_ok else 0
    score += 15 if 50 <= rsi_v <= 72 else (8 if 72 < rsi_v <= 80 else 3)
    score += 15 if vol_ok else 6
    score += 18 if dist_pct <= 2.5 else (10 if dist_pct <= 5 else 3)
    score += 10 if platform_ok else 0
    score = int(min(score, 100))

    launch_time = estimate_launch_time(score, dist_pct, rsi_v, vol_ratio, trend_ok)
    risk = risk_level(rsi_v, dist_ma20, vol_ratio)
    action = "🟢 可分批布局" if score >= 88 and risk != "🔴 高" else ("🟡 等突破" if score >= 76 else "🟠 觀察")
    if risk == "🔴 高":
        action = "🔴 不追高"

    stop = min(ma20, float(low.tail(10).min()))
    box_height = recent_high - platform_low
    target1 = recent_high
    target2 = recent_high + box_height * 0.5
    target3 = recent_high + box_height

    ai_note = f"爆發指數{score}，距離20日高點約{dist_pct:.2f}%，預估發動時間：{launch_time}。"
    if code == FAITH_STOCK:
        ai_note = "❤️7828信仰股｜" + ai_note
    if trend_ok and macd_ok:
        ai_note += " 多頭排列＋MACD偏多，線型仍可觀察。"
    elif not trend_ok:
        ai_note += " 均線未完全多頭，先等轉強。"

    result = {
        "股票代號": code,
        "現價": round(price, 2),
        "爆發指數": score,
        "預估發動時間": launch_time,
        "距離突破%": round(dist_pct, 2),
        "RSI": round(rsi_v, 1),
        "量比": round(vol_ratio, 2),
        "風險": risk,
        "建議": action,
        "停損參考": round(stop, 2),
        "第一目標": round(target1, 2),
        "第二目標": round(target2, 2),
        "第三目標": round(target3, 2),
        "AI一句話": ai_note,
    }
    return result, df

def draw_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"))
    for ma in ["MA5", "MA10", "MA20", "MA60"]:
        if ma in df:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[ma], mode="lines", name=ma))
    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.title("🚀 未來小股神 AI 選股系統 V20 PRO")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}｜❤️ 信仰股固定：7828")

# Faith card
faith_result, faith_df = analyze(FAITH_STOCK)
if faith_result:
    st.subheader("❤️ 7828 信仰股專區")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("現價", faith_result["現價"])
    c2.metric("爆發指數", faith_result["爆發指數"])
    c3.metric("預估發動時間", faith_result["預估發動時間"])
    c4.metric("風險", faith_result["風險"])
    c5.metric("建議", faith_result["建議"])
    st.info(faith_result["AI一句話"])
else:
    st.warning("7828 資料暫時抓不到，可能要確認 Yahoo 代號是 .TW 或 .TWO。")

st.divider()

st.subheader("🔍 首頁尋找股票")
query = st.text_input("輸入股票代號，例如 1714、2409、6271、7828", value="")
if query:
    r, d = analyze(query)
    if r:
        st.dataframe(pd.DataFrame([r]), use_container_width=True)
        draw_chart(d.tail(120), f"{query} 技術線圖")
    else:
        st.error("抓不到資料，請確認代號。")

st.divider()
st.subheader("🔥 今日 AI TOP 掃描")
custom = st.text_area("股票池，可自行增減，用逗號分隔", value=", ".join(DEFAULT_POOL), height=80)
if st.button("🚀 掃描股票池", type="primary"):
    codes = [x.strip() for x in custom.replace("\n", ",").split(",") if x.strip()]
    rows = []
    bar = st.progress(0)
    for i, code in enumerate(codes):
        res, _ = analyze(code)
        if res:
            rows.append(res)
        bar.progress((i + 1) / len(codes))
    if rows:
        out = pd.DataFrame(rows).sort_values(["爆發指數", "距離突破%"], ascending=[False, True])
        cols = ["股票代號", "現價", "爆發指數", "預估發動時間", "距離突破%", "RSI", "量比", "風險", "建議", "停損參考", "第一目標", "第二目標", "第三目標", "AI一句話"]
        st.dataframe(out[cols], use_container_width=True, hide_index=True)
        st.download_button("下載掃描結果 CSV", out[cols].to_csv(index=False).encode("utf-8-sig"), "ai_scan_result.csv", "text/csv")
    else:
        st.warning("沒有成功抓到資料。")

with st.expander("欄位說明：預估發動時間"):
    st.write("預估發動時間會依照爆發指數、距離突破、RSI、量比、均線多頭排列判斷，分成：1～3天、2～5天、5～10天、觀察中、未成熟。")
