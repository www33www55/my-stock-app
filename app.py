# 🚀 未來小股神 AI－發動前雷達 Ultimate 13.0
# 使用方式：
# 1) pip install -r requirements.txt
# 2) streamlit run app.py
#
# 注意：這是技術面篩選工具，不是保證獲利或投資建議。
# 資料來源 yfinance / twstock 可能延遲或缺漏，請下單前再用券商資料確認。

import warnings
warnings.filterwarnings("ignore")

import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import twstock
except Exception:
    twstock = None

st.set_page_config(page_title="未來小股神 AI－發動前雷達 13.0", layout="wide")

# -----------------------------
# 股票池
# -----------------------------
MY_POOL = {
    "1714": "和桐",
    "1717": "長興",
    "1773": "勝一",
    "1815": "富喬",
    "2313": "華通",
    "2409": "友達",
    "2413": "環科",
    "2489": "瑞軒",
    "3693": "營邦",
    "4967": "十銓",
    "7828": "創新服務",
}

DEFAULT_WATCH = "1714,1717,1773,1815,2313,2409,2413,2489,3693,4967,7828"

def get_tw_stock_list(limit_all=True):
    if twstock is None:
        return list(MY_POOL.keys())
    codes = []
    try:
        for code, info in twstock.codes.items():
            if not code.isdigit():
                continue
            if len(code) != 4:
                continue
            # 普通上市櫃股票為主
            if getattr(info, "type", "") == "股票":
                codes.append(code)
    except Exception:
        codes = list(MY_POOL.keys())
    return sorted(set(codes))

def get_name(code):
    if code in MY_POOL:
        return MY_POOL[code]
    if twstock is not None:
        try:
            return twstock.codes.get(code).name if twstock.codes.get(code) else ""
        except Exception:
            return ""
    return ""

# -----------------------------
# 資料與指標
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(code, period="9mo"):
    suffixes = [".TW", ".TWO"]
    last_err = None
    for suf in suffixes:
        try:
            df = yf.download(f"{code}{suf}", period=period, interval="1d", auto_adjust=False, progress=False)
            if df is not None and not df.empty and len(df) > 40:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.dropna()
                df["Code"] = code
                return df
        except Exception as e:
            last_err = e
    return pd.DataFrame()

def calc_indicators(df):
    d = df.copy()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA10"] = d["Close"].rolling(10).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA60"] = d["Close"].rolling(60).mean()
    d["VOL5"] = d["Volume"].rolling(5).mean()
    d["VOL20"] = d["Volume"].rolling(20).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["MACD_H"] = d["DIF"] - d["DEA"]

    d["HH20"] = d["High"].rolling(20).max()
    d["LL20"] = d["Low"].rolling(20).min()
    d["HH60"] = d["High"].rolling(60).max()
    return d

def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100

def score_stock(df):
    if df.empty or len(df) < 80:
        return None

    d = calc_indicators(df)
    r = d.iloc[-1]
    prev = d.iloc[-2]
    close = float(r["Close"])
    high20 = float(r["HH20"])
    high60 = float(r["HH60"])
    low20 = float(r["LL20"])

    reasons = []
    score = 0

    # 1 型態成熟度 25
    range20 = (high20 - low20) / close if close else 9
    near_high20 = (high20 - close) / close if close else 9
    if range20 <= 0.16 and near_high20 <= 0.035:
        score += 25
        reasons.append("平台整理後接近突破")
    elif near_high20 <= 0.05:
        score += 16
        reasons.append("接近20日高點")
    elif range20 <= 0.18:
        score += 10
        reasons.append("區間收斂中")

    # 2 均線 20
    ma_ok = r["MA5"] > r["MA10"] > r["MA20"]
    ma_up = r["MA5"] > prev["MA5"] and r["MA10"] >= prev["MA10"]
    if ma_ok and ma_up:
        score += 20
        reasons.append("5/10/20日均線多頭上彎")
    elif r["MA5"] > r["MA10"] and r["Close"] > r["MA20"]:
        score += 12
        reasons.append("短均轉強")
    elif r["Close"] > r["MA20"]:
        score += 6
        reasons.append("站上月線")

    # 3 MACD 15
    macd_cross_recent = (d["DIF"].iloc[-5:] > d["DEA"].iloc[-5:]).sum()
    if r["DIF"] > r["DEA"] and r["MACD_H"] > prev["MACD_H"] and macd_cross_recent <= 4:
        score += 15
        reasons.append("MACD剛翻多")
    elif r["DIF"] > r["DEA"]:
        score += 9
        reasons.append("MACD多方")
    elif r["MACD_H"] > prev["MACD_H"]:
        score += 5
        reasons.append("MACD柱狀體改善")

    # 4 RSI 10
    rsi = float(r["RSI"]) if not pd.isna(r["RSI"]) else 0
    if 55 <= rsi <= 72:
        score += 10
        reasons.append(f"RSI甜蜜區 {rsi:.1f}")
    elif 45 <= rsi < 55:
        score += 5
        reasons.append(f"RSI未過熱 {rsi:.1f}")
    elif 72 < rsi <= 80:
        score += 3
        reasons.append(f"RSI偏熱 {rsi:.1f}")
    elif rsi > 80:
        score -= 8
        reasons.append(f"RSI過熱 {rsi:.1f}")

    # 5 量能 15
    vol_ratio = r["Volume"] / r["VOL20"] if r["VOL20"] and not pd.isna(r["VOL20"]) else np.nan
    if 1.3 <= vol_ratio <= 2.8:
        score += 15
        reasons.append(f"量能溫和放大 {vol_ratio:.1f}倍")
    elif 0.75 <= vol_ratio < 1.3:
        score += 7
        reasons.append("量能正常")
    elif vol_ratio > 2.8:
        score += 5
        reasons.append(f"爆量，需防震盪 {vol_ratio:.1f}倍")

    # 6 距離突破 10
    dist_break = (high20 - close) / close * 100
    if 0 <= dist_break <= 3:
        score += 10
        reasons.append(f"距離突破約 {dist_break:.1f}%")
    elif 3 < dist_break <= 6:
        score += 5
        reasons.append(f"距離突破約 {dist_break:.1f}%")

    # 7 風險扣分：離60日高太近但已過熱、跌破月線
    if close < r["MA20"]:
        score -= 10
        reasons.append("跌破月線扣分")
    if pct(close, r["MA5"]) > 8:
        score -= 8
        reasons.append("乖離5日線過大扣分")

    score = int(max(0, min(100, score)))

    if score >= 90:
        signal = "🟢 A級：快發動"
        days = "0～2天"
    elif score >= 80:
        signal = "🟡 B級：接近成熟"
        days = "2～5天"
    elif score >= 70:
        signal = "🔵 C級：繼續等"
        days = "5～10天"
    else:
        signal = "⚪ 暫時觀察"
        days = "等待型態"

    support = min(float(r["MA5"]), float(r["MA10"])) if not pd.isna(r["MA10"]) else close * 0.95
    stop = support * 0.97
    target1 = high20 * 1.06
    target2 = high60 * 1.12

    return {
        "代號": str(r["Code"]),
        "名稱": get_name(str(r["Code"])),
        "收盤": round(close, 2),
        "發動率": score,
        "燈號": signal,
        "預估發動": days,
        "距突破%": round(dist_break, 2),
        "RSI": round(rsi, 1),
        "量比": round(float(vol_ratio), 2) if not pd.isna(vol_ratio) else np.nan,
        "MA5": round(float(r["MA5"]), 2) if not pd.isna(r["MA5"]) else np.nan,
        "MA10": round(float(r["MA10"]), 2) if not pd.isna(r["MA10"]) else np.nan,
        "MA20": round(float(r["MA20"]), 2) if not pd.isna(r["MA20"]) else np.nan,
        "建議買點": f"{round(close*0.99,2)}～{round(close*1.01,2)}",
        "停損參考": round(stop, 2),
        "目標1": round(target1, 2),
        "目標2": round(target2, 2),
        "入選原因": "、".join(reasons[:6]),
        "_df": d,
    }

def make_chart(d, code, name):
    tail = d.tail(120)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.58, 0.22, 0.20])
    fig.add_trace(go.Candlestick(
        x=tail.index, open=tail["Open"], high=tail["High"],
        low=tail["Low"], close=tail["Close"], name="K線"
    ), row=1, col=1)
    for ma in ["MA5", "MA10", "MA20", "MA60"]:
        fig.add_trace(go.Scatter(x=tail.index, y=tail[ma], mode="lines", name=ma), row=1, col=1)
    fig.add_trace(go.Bar(x=tail.index, y=tail["Volume"], name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tail.index, y=tail["DIF"], mode="lines", name="DIF"), row=3, col=1)
    fig.add_trace(go.Scatter(x=tail.index, y=tail["DEA"], mode="lines", name="DEA"), row=3, col=1)
    fig.add_trace(go.Bar(x=tail.index, y=tail["MACD_H"], name="MACD柱"), row=3, col=1)
    fig.update_layout(title=f"{code} {name}", xaxis_rangeslider_visible=False, height=760)
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🚀 未來小股神 AI－發動前雷達 Ultimate 13.0")
st.caption("專找：盤整尾端、圓弧底、N字、突破前、MACD剛翻多、RSI未過熱。")

with st.sidebar:
    st.header("⚙️ 掃描設定")
    mode = st.radio("掃描模式", ["我的持股池", "自訂代號", "全台股掃描（較慢）"], index=0)
    custom_codes = st.text_area("自訂股票代號，用逗號分隔", DEFAULT_WATCH, height=100)
    topn = st.slider("顯示前幾名", 5, 50, 15)
    run = st.button("🔥 開始掃描", type="primary")

if mode == "我的持股池":
    codes = list(MY_POOL.keys())
elif mode == "自訂代號":
    codes = [c.strip() for c in custom_codes.replace("，", ",").split(",") if c.strip()]
else:
    codes = get_tw_stock_list()
    st.warning("全台股掃描會比較久，第一次可能需要數分鐘。")

if run:
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, code in enumerate(codes):
        status.write(f"掃描中：{code} {get_name(code)}")
        df = load_data(code)
        res = score_stock(df)
        if res:
            rows.append(res)
        progress.progress((i + 1) / len(codes))
        time.sleep(0.02)

    progress.empty()
    status.empty()

    if not rows:
        st.error("沒有抓到資料。可以稍後重試，或改用自訂代號。")
    else:
        result = pd.DataFrame([{k:v for k,v in r.items() if k != "_df"} for r in rows])
        result = result.sort_values(["發動率", "距突破%"], ascending=[False, True]).head(topn).reset_index(drop=True)

        st.subheader("🔥 今日發動前排行榜")
        st.dataframe(result, use_container_width=True, hide_index=True)

        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載排行榜 CSV", csv, "future_stock_god_radar.csv", "text/csv")

        st.subheader("🏆 前三名")
        cols = st.columns(min(3, len(result)))
        for idx, col in enumerate(cols):
            r = result.iloc[idx]
            with col:
                st.metric(f"🥇 No.{idx+1} {r['代號']} {r['名稱']}", f"{r['發動率']}%", r["燈號"])
                st.write(f"⏳ 預估：{r['預估發動']}")
                st.write(f"🎯 買點：{r['建議買點']}")
                st.write(f"🛡 停損：{r['停損參考']}")

        st.subheader("📈 K線健檢")
        selected = st.selectbox("選一檔看圖", result["代號"].tolist())
        full = next((x for x in rows if x["代號"] == selected), None)
        if full:
            st.plotly_chart(make_chart(full["_df"], selected, full["名稱"]), use_container_width=True)
            st.info(f"入選原因：{full['入選原因']}")
            st.write({
                "建議買點": full["建議買點"],
                "停損參考": full["停損參考"],
                "第一目標": full["目標1"],
                "第二目標": full["目標2"],
            })
else:
    st.info("左邊按「開始掃描」。先用『我的持股池』最快。")
