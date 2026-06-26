import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="未來小股神 AI 操盤中心 V33.1", layout="wide")

FAITH_CODE = "7828"
FAITH_NAME = "創新服務"

# -----------------------------
# Utils
# -----------------------------
def clean_num(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace(",", "").replace("--", "").strip()
        return float(x)
    except Exception:
        return np.nan


def safe_last(s, default=np.nan):
    try:
        if s is None or len(s) == 0:
            return default
        v = s.dropna().iloc[-1]
        return v
    except Exception:
        return default


def stars(score: float) -> str:
    if score >= 90:
        return "⭐⭐⭐⭐⭐"
    if score >= 75:
        return "⭐⭐⭐⭐"
    if score >= 60:
        return "⭐⭐⭐"
    if score >= 40:
        return "⭐⭐"
    return "⭐"


def confidence(score: float) -> str:
    if score >= 95:
        c = 95
    elif score >= 90:
        c = 90
    elif score >= 85:
        c = 85
    elif score >= 80:
        c = 80
    elif score >= 75:
        c = 75
    else:
        c = max(50, int(score))
    return f"{c}%"


def estimate_trigger(score: float, row: Dict) -> str:
    if row.get("risk_high", False):
        return "高風險"
    if row.get("breakout", False):
        return "已發動"
    if score >= 85:
        return "1～3天"
    if score >= 70:
        return "2～5天"
    if score >= 55:
        return "5～10天"
    return "觀察中"

# -----------------------------
# Universe loader: true listed + OTC pool
# -----------------------------
@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_universe() -> pd.DataFrame:
    frames = []
    sources = [("上市", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"),
               ("上櫃", "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4")]
    headers = {"User-Agent": "Mozilla/5.0"}
    for market, url in sources:
        try:
            html = requests.get(url, headers=headers, timeout=20).content
            tables = pd.read_html(html, encoding="big5")
            df = tables[0].copy()
            df.columns = df.iloc[0]
            df = df.iloc[1:].copy()
            first_col = df.columns[0]
            # 只保留股票，排除 ETF/權證/受益證券等
            if "有價證券代號及名稱" not in str(first_col):
                first_col = df.columns[0]
            work = df[[first_col]].copy()
            work["raw"] = work[first_col].astype(str)
            work = work[work["raw"].str.match(r"^\d{4}\s+.+")]
            work["代號"] = work["raw"].str.extract(r"^(\d{4})")
            work["名稱"] = work["raw"].str.replace(r"^\d{4}\s+", "", regex=True).str.strip()
            # 若原始表有有價證券別，盡量只保留股票
            if "有價證券別" in df.columns:
                types = df.loc[work.index, "有價證券別"].astype(str)
                work = work[types.str.contains("股票", na=False)]
            work["市場"] = market
            work["類型"] = "股票"
            frames.append(work[["代號", "名稱", "市場", "類型"]])
        except Exception:
            pass
    if frames:
        uni = pd.concat(frames, ignore_index=True).drop_duplicates("代號")
    else:
        uni = pd.DataFrame([
            ["2330", "台積電", "上市", "股票"], ["2303", "聯電", "上市", "股票"],
            ["2409", "友達", "上市", "股票"], ["6271", "同欣電", "上市", "股票"],
            ["6191", "精成科", "上市", "股票"], ["3567", "逸昌", "上櫃", "股票"],
            ["7828", "創新服務", "上櫃", "股票"]
        ], columns=["代號", "名稱", "市場", "類型"])
    if FAITH_CODE not in set(uni["代號"]):
        uni = pd.concat([uni, pd.DataFrame([[FAITH_CODE, FAITH_NAME, "上櫃", "股票"]], columns=uni.columns)], ignore_index=True)
    return uni.sort_values("代號").reset_index(drop=True)


def code_to_yahoo(code: str, market: str) -> str:
    return f"{code}.TW" if market == "上市" else f"{code}.TWO"

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_one(ticker: str, period="6mo") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()

# -----------------------------
# Indicators + scoring
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"] if "High" in out else close
    low = out["Low"] if "Low" in out else close
    vol = out["Volume"] if "Volume" in out else pd.Series(index=out.index, data=np.nan)
    for n in [5, 10, 20, 60]:
        out[f"MA{n}"] = close.rolling(n).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    out["K"] = rsv.ewm(com=2, adjust=False).mean()
    out["D"] = out["K"].ewm(com=2, adjust=False).mean()
    out["VOL_MA20"] = vol.rolling(20).mean()
    out["量比"] = vol / out["VOL_MA20"].replace(0, np.nan)
    out["HIGH20"] = high.rolling(20).max()
    out["HIGH60"] = high.rolling(60).max()
    out["LOW20"] = low.rolling(20).min()
    return out


def score_stock(code: str, name: str, market: str, df: pd.DataFrame) -> Dict:
    if df is None or df.empty or len(df) < 60:
        return {"代號": code, "名稱": name, "市場": market, "現價": np.nan, "爆發指數": 0, "AI信心": "50%", "發動時間": "資料不足", "評等": "⭐", "RSI": np.nan, "MACD": np.nan, "量比": np.nan, "AI解讀": "資料不足", "risk_high": False, "breakout": False}
    d = add_indicators(df)
    last = d.iloc[-1]
    prev = d.iloc[-2]
    close = clean_num(last.get("Close"))
    ma5, ma10, ma20, ma60 = [clean_num(last.get(f"MA{n}")) for n in [5, 10, 20, 60]]
    rsi = clean_num(last.get("RSI"))
    macd = clean_num(last.get("MACD"))
    sig = clean_num(last.get("MACD_SIGNAL"))
    hist = clean_num(last.get("MACD_HIST"))
    hist_prev = clean_num(prev.get("MACD_HIST"))
    k = clean_num(last.get("K")); dd = clean_num(last.get("D"))
    vol_ratio = clean_num(last.get("量比"))
    high20_prev = clean_num(d["High"].rolling(20).max().shift(1).iloc[-1])
    high60 = clean_num(last.get("HIGH60"))
    low20 = clean_num(last.get("LOW20"))

    score = 0
    reasons = []
    risks = []
    # 技術面 40
    if all(np.isfinite([ma5, ma10, ma20])) and ma5 > ma10 > ma20:
        score += 12; reasons.append("均線多頭")
    if np.isfinite(close) and np.isfinite(ma20) and close > ma20:
        score += 8; reasons.append("站上月線")
    if np.isfinite(close) and np.isfinite(high60) and high60 > 0:
        dist_high60 = (high60 - close) / high60 * 100
        if dist_high60 <= 3:
            score += 10; reasons.append("接近60日高")
        elif dist_high60 <= 8:
            score += 5; reasons.append("接近波段高")
    if np.isfinite(macd) and np.isfinite(sig):
        if macd > sig and macd > 0:
            score += 10; reasons.append("MACD多頭")
        elif macd > sig:
            score += 6; reasons.append("MACD翻揚")
    if np.isfinite(k) and np.isfinite(dd) and k > dd:
        score += 5; reasons.append("KD偏多")
    # RSI 基礎加分，不要太嚴格
    if np.isfinite(rsi):
        if 50 <= rsi <= 70:
            score += 10; reasons.append("RSI健康")
        elif 45 <= rsi < 50 or 70 < rsi <= 78:
            score += 5; reasons.append("RSI可接受")
    # 量價 20
    if np.isfinite(vol_ratio):
        if 1.2 <= vol_ratio <= 3.5:
            score += 10; reasons.append("量能放大")
        elif 0.8 <= vol_ratio < 1.2:
            score += 4; reasons.append("量能穩定")
    breakout = bool(np.isfinite(close) and np.isfinite(high20_prev) and close >= high20_prev)
    if breakout:
        score += 10; reasons.append("突破20日高")
    # 型態 10：平台整理、箱型靠近上緣
    platform = False
    if np.isfinite(high20_prev) and np.isfinite(low20) and low20 > 0:
        rng = (high20_prev - low20) / low20 * 100
        near_top = close >= high20_prev * 0.97 if np.isfinite(close) else False
        if rng <= 18 and near_top:
            platform = True
            score += 8; reasons.append("平台尾端")
        elif rng <= 25:
            score += 4; reasons.append("整理中")
    # 籌碼目前未串接真法人，先不假造，只給中性，不加分

    risk_high = False
    if np.isfinite(rsi) and rsi > 80:
        score -= 10; risks.append("RSI過熱"); risk_high = True
    if all(np.isfinite([close, ma5])) and close < ma5:
        score -= 8; risks.append("跌破5MA")
    if np.isfinite(macd) and np.isfinite(sig) and macd < sig:
        score -= 8; risks.append("MACD轉弱")
    if np.isfinite(vol_ratio) and vol_ratio > 5 and np.isfinite(close) and close < clean_num(prev.get("Close")):
        score -= 18; risks.append("爆量長黑"); risk_high = True

    score = int(max(0, min(100, round(score))))
    temp_row = {"risk_high": risk_high, "breakout": breakout}
    trig = estimate_trigger(score, temp_row)
    ai = "、".join(reasons[:4]) if reasons else "條件未成熟"
    if risks:
        ai += "；風險：" + "、".join(risks[:2])
    return {
        "代號": code, "名稱": name, "市場": market, "現價": round(close, 2) if np.isfinite(close) else np.nan,
        "爆發指數": score, "AI信心": confidence(score), "發動時間": trig, "評等": stars(score),
        "RSI": round(rsi, 1) if np.isfinite(rsi) else np.nan,
        "MACD": round(macd, 3) if np.isfinite(macd) else np.nan,
        "量比": round(vol_ratio, 2) if np.isfinite(vol_ratio) else np.nan,
        "AI解讀": ai,
        "risk_high": risk_high, "breakout": breakout,
    }

@st.cache_data(ttl=60*20, show_spinner=False)
def market_eval(ticker: str, name: str) -> Dict:
    df = fetch_one(ticker, period="1y")
    if df.empty:
        return {"市場": name, "收盤": "-", "AI分數": "-", "RSI": "-", "MACD": "-", "技術評估": "資料不足"}
    r = score_stock(ticker, name, "指數", df)
    score = r["爆發指數"]
    if score >= 75:
        txt = "偏多，適合找平台整理股"
    elif score >= 55:
        txt = "震盪選股，不宜追高"
    elif score >= 35:
        txt = "偏弱震盪，降低部位"
    else:
        txt = "保守觀望"
    return {"市場": name, "收盤": r["現價"], "AI分數": score, "RSI": r["RSI"], "MACD": r["MACD"], "技術評估": txt}

# -----------------------------
# UI
# -----------------------------
st.title("🚀 未來小股神 AI 操盤中心 V33.1")
st.caption("修正：真全池掃描｜真評分不全100｜AI大盤技術評估")

universe = load_universe()
st.success(f"已載入股票池：{len(universe)} 檔（上市＋上櫃；含名稱與市場別）")

tab1, tab2, tab3 = st.tabs(["🔥 全池TOP20", "🔍 單股掃描", "📋 股票池"])

with tab1:
    st.subheader("📊 AI 大盤技術評估")
    m1, m2 = market_eval("^TWII", "加權指數"), market_eval("^TWOII", "櫃買OTC")
    st.dataframe(pd.DataFrame([m1, m2]), use_container_width=True, hide_index=True)

    st.subheader("🔥 今日 AI TOP20（全市場）")
    st.caption("會逐檔下載資料並評分。Streamlit Cloud 免費版可能需要幾分鐘；抓不到的股票會跳過，不會整個掛掉。")
    col_a, col_b, col_c = st.columns([1,1,2])
    max_scan = col_a.number_input("最多掃描檔數（0=全池）", min_value=0, max_value=int(len(universe)), value=0, step=50)
    top_n = col_b.slider("顯示前幾名", 10, 100, 20, 10)
    start = col_a.button("開始全池掃描 / 更新TOP20", type="primary")

    if start:
        pool = universe.copy()
        if max_scan and max_scan > 0:
            pool = pool.head(int(max_scan))
        progress = st.progress(0)
        status = st.empty()
        results = []
        total = len(pool)
        for i, row in pool.iterrows():
            code, name, market = row["代號"], row["名稱"], row["市場"]
            ticker = code_to_yahoo(code, market)
            status.write(f"AI掃描中：{len(results)}筆有效 / {i+1} / {total}　目前：{code} {name}")
            df = fetch_one(ticker, period="6mo")
            if df.empty and market == "上櫃":
                # 少數上櫃 Yahoo 可能掛，補試上市格式
                df = fetch_one(f"{code}.TW", period="6mo")
            if not df.empty:
                res = score_stock(code, name, market, df)
                results.append(res)
            progress.progress(min(1.0, (i+1)/total))
        status.write(f"掃描完成：有效 {len(results)} 檔 / 股票池 {total} 檔")
        if results:
            out = pd.DataFrame(results)
            out = out.sort_values(["爆發指數", "RSI"], ascending=[False, True]).reset_index(drop=True)
            st.session_state["last_scan"] = out
        else:
            st.warning("本次沒有抓到有效資料，請稍後再試。")

    if "last_scan" in st.session_state:
        out = st.session_state["last_scan"]
        show_cols = ["代號", "名稱", "市場", "現價", "爆發指數", "AI信心", "發動時間", "評等", "RSI", "MACD", "量比", "AI解讀"]
        st.dataframe(out[show_cols].head(top_n), use_container_width=True, hide_index=True)
        st.download_button("下載本次掃描CSV", out[show_cols].to_csv(index=False).encode("utf-8-sig"), file_name=f"v33_1_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    else:
        st.info("請按上方紅色按鈕開始掃描。")

with tab2:
    st.subheader("❤️ 7828 信仰股 / 單股 AI 掃描")
    code_input = st.text_input("輸入代號或名稱", value=FAITH_CODE)
    if st.button("分析單股", type="primary"):
        q = code_input.strip()
        target = universe[(universe["代號"] == q) | (universe["名稱"].str.contains(q, na=False))]
        if target.empty and q == FAITH_CODE:
            target = pd.DataFrame([[FAITH_CODE, FAITH_NAME, "上櫃", "股票"]], columns=universe.columns)
        if target.empty:
            st.error("找不到股票，請確認代號或名稱。")
        else:
            r0 = target.iloc[0]
            ticker = code_to_yahoo(r0["代號"], r0["市場"])
            df = fetch_one(ticker, period="1y")
            if df.empty and r0["市場"] == "上櫃":
                df = fetch_one(f"{r0['代號']}.TW", period="1y")
            res = score_stock(r0["代號"], r0["名稱"], r0["市場"], df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("爆發指數", res["爆發指數"])
            c2.metric("AI信心", res["AI信心"])
            c3.metric("發動時間", res["發動時間"])
            c4.metric("RSI", res["RSI"])
            st.write("### AI解讀")
            st.info(res["AI解讀"])
            if not df.empty:
                d = add_indicators(df)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="K線"))
                for ma in ["MA5", "MA20", "MA60"]:
                    fig.add_trace(go.Scatter(x=d.index, y=d[ma], mode="lines", name=ma))
                fig.update_layout(height=520, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("此股票目前抓不到足夠歷史資料。")

with tab3:
    st.subheader("📋 股票池檢查")
    st.caption("這裡只是股票池清單，不是評分結果。評分請到『全池TOP20』按開始掃描。")
    st.dataframe(universe, use_container_width=True, hide_index=True)
