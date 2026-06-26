import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="未來小股神 AI 操盤中心 V31.1 REAL", layout="wide")

# ---------- 基礎資料 ----------
NAME_MAP = {
    "7828": "創新服務", "1714": "和桐", "2409": "友達", "2303": "聯電", "6271": "同欣電",
    "6191": "精成科", "3557": "逸昌", "3037": "欣興", "2382": "廣達", "2313": "華通",
    "2344": "華邦電", "2359": "所羅門", "3060": "銘異", "8923": "時報", "6272": "驊陞",
    "5468": "凱鈺", "6259": "百徽", "5211": "蒙恬", "1730": "花仙子", "3567": "逸昌",
    "4183": "福永生技", "5469": "瀚宇博", "3064": "泰偉", "2643": "捷迅", "8183": "精星",
    "1817": "凱撒衛", "4554": "橙的", "2013": "中鋼構", "1731": "美吾華", "1240": "茂生農經",
    "2739": "寒舍", "3479": "安勤", "3717": "聯嘉投控"
}

DEFAULT_POOL = [
    "7828", "1714", "2409", "2303", "6271", "6191", "3557", "3037", "2382", "2313",
    "2344", "2359", "3060", "8923", "6272", "5468", "6259", "5211", "1730", "3567",
    "4183", "5469", "3064", "2643", "8183", "1817", "4554", "2013", "1731", "1240",
    "2739", "3479", "3717"
]

# 這裡放常見台股池，避免一次掃 1900 檔過慢；使用者可自行貼更多代號。
LARGE_POOL = list(dict.fromkeys(DEFAULT_POOL + [
    "2330", "2317", "2454", "2412", "2881", "2882", "2891", "2886", "2884", "2885",
    "2603", "2609", "2615", "2618", "2610", "2002", "1301", "1303", "1326", "1101",
    "1102", "1216", "2207", "2308", "2327", "2379", "2383", "2395", "3008", "3017",
    "3231", "3443", "3661", "3711", "6669", "8046", "8069", "2376", "2356", "2474",
    "1504", "1513", "1514", "1519", "1522", "1536", "1589", "1590", "1597", "2049",
    "2301", "2353", "2360", "2368", "2371", "2377", "2385", "2392", "2408", "2439",
    "2449", "2455", "2481", "2492", "2498", "3013", "3022", "3034", "3044", "3059",
    "3189", "3324", "3376", "3481", "3532", "3533", "3653", "3665", "3702", "4938",
]))

# ---------- 工具函數 ----------
def stock_name(code: str) -> str:
    return NAME_MAP.get(str(code), "")


def ticker_candidates(code: str) -> List[str]:
    code = str(code).strip()
    if code.startswith("^"):
        return [code]
    # 台股上市/上櫃兩種都試，不讓 7828.TWO/TW 錯了就直接死
    return [f"{code}.TW", f"{code}.TWO"]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_history(code: str, period: str = "9mo") -> Tuple[pd.DataFrame, str, str]:
    last_err = ""
    for t in ticker_candidates(code):
        try:
            df = yf.download(t, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                last_err = "空資料"
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(set(df.columns)):
                last_err = f"缺欄位 {needed - set(df.columns)}"
                continue
            df = df.dropna(subset=["Close"]).copy()
            if len(df) < 35:
                last_err = "資料不足"
                continue
            return df, t, "OK"
        except Exception as e:
            last_err = str(e)
    return pd.DataFrame(), "", last_err or "無資料"


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = c.rolling(n).mean()
    out["RSI"] = rsi(c)
    low9 = l.rolling(9).min(); high9 = h.rolling(9).max()
    out["K"] = ((c - low9) / (high9 - low9).replace(0, np.nan) * 100).ewm(com=2).mean()
    out["D"] = out["K"].ewm(com=2).mean()
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    out["MACD"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["OSC"] = out["DIF"] - out["MACD"]
    out["BB_MID"] = c.rolling(20).mean()
    out["BB_STD"] = c.rolling(20).std()
    out["BB_UP"] = out["BB_MID"] + 2 * out["BB_STD"]
    out["BB_LOW"] = out["BB_MID"] - 2 * out["BB_STD"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    plus_dm = h.diff().clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / out["ATR"].replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / out["ATR"].replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["ADX"] = dx.rolling(14).mean()
    out["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    typical = (h + l + c) / 3
    out["VWAP"] = (typical * v).cumsum() / v.replace(0, np.nan).cumsum()
    out["ROC"] = c.pct_change(10) * 100
    out["VOL_MA20"] = v.rolling(20).mean()
    return out


def pattern_text(df: pd.DataFrame) -> str:
    d = df.dropna().copy()
    if len(d) < 30:
        return "資料不足"
    c = d["Close"]
    last = c.iloc[-1]
    high20 = d["High"].tail(20).max()
    low20 = d["Low"].tail(20).min()
    range_pct = (high20 - low20) / max(last, 1) * 100
    ma_ok = d["MA5"].iloc[-1] > d["MA10"].iloc[-1] > d["MA20"].iloc[-1]
    near_high = (high20 - last) / max(high20, 1) * 100
    if range_pct < 12 and near_high < 3:
        return "平台整理接近突破"
    if ma_ok and d["MA20"].iloc[-1] > d["MA20"].iloc[-5]:
        return "主升段整理"
    if c.iloc[-1] > c.iloc[-10] and c.iloc[-10] < c.iloc[-20]:
        return "疑似圓弧底翻揚"
    return "觀察中"


def analyze_stock(code: str) -> Optional[Dict]:
    df, ticker, status = fetch_history(code)
    if df.empty:
        return {"股票代號": code, "名稱": stock_name(code), "狀態": f"無資料：{status}", "爆發指數": 0}
    df = add_indicators(df)
    d = df.dropna().copy()
    if len(d) < 30:
        return {"股票代號": code, "名稱": stock_name(code), "狀態": "資料不足", "爆發指數": 0}

    last = d.iloc[-1]
    close = float(last["Close"])
    high20 = float(d["High"].tail(20).max())
    low20 = float(d["Low"].tail(20).min())
    dist_break = max(0.0, (high20 - close) / high20 * 100) if high20 else 999
    vol_ratio = float(last["Volume"] / last["VOL_MA20"]) if last["VOL_MA20"] and not pd.isna(last["VOL_MA20"]) else 0
    r = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50
    ma_score = 20 if last["MA5"] > last["MA10"] > last["MA20"] else 8
    macd_score = 20 if last["DIF"] > last["MACD"] and last["DIF"] > 0 else (12 if last["DIF"] > last["MACD"] else 4)
    rsi_score = 15 if 55 <= r <= 72 else (10 if 45 <= r < 80 else 3)
    vol_score = min(15, max(0, vol_ratio * 8))
    break_score = 20 if dist_break <= 1 else (16 if dist_break <= 3 else (10 if dist_break <= 6 else 3))
    trend_score = 10 if close > last["MA20"] else 2
    score = int(min(100, ma_score + macd_score + rsi_score + vol_score + break_score + trend_score))
    if score >= 96 and dist_break <= 1:
        fire_time = "今天～1天"
    elif score >= 92:
        fire_time = "1～3天"
    elif score >= 85:
        fire_time = "2～5天"
    elif score >= 75:
        fire_time = "5～10天"
    else:
        fire_time = "觀察中"
    pat = pattern_text(d)
    support1 = float(d["Low"].tail(10).min())
    support2 = float(d["Low"].tail(20).min())
    stop = min(support1, float(last["MA20"])) * 0.985
    target1 = close + (high20 - low20) * 0.5
    target2 = close + (high20 - low20) * 1.0
    target3 = close + (high20 - low20) * 1.5
    risk = "低" if r < 72 and close < high20 * 1.03 else ("中" if r < 82 else "高")
    ai = f"{pat}，RSI {r:.1f}，量比 {vol_ratio:.2f}，距突破 {dist_break:.2f}%，預估發動 {fire_time}。"
    return {
        "股票代號": code, "名稱": stock_name(code), "Ticker": ticker, "現價": round(close, 2),
        "爆發指數": score, "預估發動時間": fire_time, "距離突破%": round(dist_break, 2),
        "RSI": round(r, 1), "K": round(float(last["K"]), 1), "D": round(float(last["D"]), 1),
        "MACD狀態": "多" if last["DIF"] > last["MACD"] else "弱", "量比": round(vol_ratio, 2),
        "MA5": round(float(last["MA5"]), 2), "MA10": round(float(last["MA10"]), 2), "MA20": round(float(last["MA20"]), 2),
        "型態": pat, "支撐1": round(support1, 2), "支撐2": round(support2, 2), "停損參考": round(stop, 2),
        "目標1": round(target1, 2), "目標2": round(target2, 2), "目標3": round(target3, 2),
        "風險": risk, "AI一句話": ai, "狀態": "OK", "_df": d
    }


def make_k_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"))
    for ma in ["MA5", "MA10", "MA20"]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode="lines", name=ma))
    fig.update_layout(title=title, height=480, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


@st.cache_data(ttl=60 * 20, show_spinner=False)
def scan_pool(codes: List[str], limit: int = 80) -> pd.DataFrame:
    rows = []
    for i, code in enumerate(codes[:limit]):
        res = analyze_stock(str(code).strip())
        if res and res.get("狀態") == "OK":
            rows.append({k: v for k, v in res.items() if not k.startswith("_")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["爆發指數", "距離突破%"], ascending=[False, True]).reset_index(drop=True)


def market_analysis() -> pd.DataFrame:
    items = [("加權指數", "^TWII"), ("櫃買OTC", "^TWOII")]
    rows = []
    for name, code in items:
        df, _, status = fetch_history(code, period="6mo")
        if df.empty:
            rows.append({"市場": name, "收盤": "-", "AI分數": "-", "RSI": "-", "MACD": "-", "AI解讀": "資料暫無"})
            continue
        d = add_indicators(df).dropna()
        if d.empty:
            rows.append({"市場": name, "收盤": "-", "AI分數": "-", "RSI": "-", "MACD": "-", "AI解讀": "資料不足"})
            continue
        last = d.iloc[-1]
        score = 50
        if last["Close"] > last["MA20"]: score += 15
        if last["MA5"] > last["MA10"] > last["MA20"]: score += 15
        if last["DIF"] > last["MACD"]: score += 10
        if 45 <= last["RSI"] <= 72: score += 10
        score = min(100, score)
        view = "偏多，可正常找突破股" if score >= 75 else ("中性，精選個股" if score >= 60 else "保守，降低追高")
        rows.append({"市場": name, "收盤": round(float(last["Close"]), 2), "AI分數": int(score), "RSI": round(float(last["RSI"]), 1), "MACD": "多" if last["DIF"] > last["MACD"] else "弱", "AI解讀": view})
    return pd.DataFrame(rows)

# ---------- UI ----------
st.title("🚀 未來小股神 AI 操盤中心 V31.1 REAL")
st.caption("重點：能動、不 empty、單檔抓不到不會整個爆掉。")

with st.sidebar:
    st.header("⚙️ 模式")
    page = st.radio("選單", ["首頁", "單股掃描", "全池掃描", "歷史回測示範"], index=0)
    pool_mode = st.selectbox("股票池", ["示範池（較快）", "擴充池（較慢）", "自訂"])
    custom_codes = st.text_area("自訂股票池，用逗號分隔", ",".join(DEFAULT_POOL[:20]), height=100)
    max_scan = st.slider("最多掃描檔數", 10, 200, 60, step=10)

if pool_mode == "示範池（較快）":
    pool = DEFAULT_POOL
elif pool_mode == "擴充池（較慢）":
    pool = LARGE_POOL
else:
    pool = [x.strip() for x in custom_codes.replace("\n", ",").split(",") if x.strip()]

if page == "首頁":
    st.subheader("❤️ 7828 信仰股")
    faith = analyze_stock("7828")
    if faith and faith.get("狀態") == "OK":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", faith["現價"])
        c2.metric("爆發指數", faith["爆發指數"])
        c3.metric("發動時間", faith["預估發動時間"])
        c4.metric("風險", faith["風險"])
        st.info(faith["AI一句話"])
    else:
        st.warning(f"7828 暫時抓不到資料：{faith.get('狀態') if faith else '無資料'}")

    st.subheader("📊 AI 大盤分析")
    st.dataframe(market_analysis(), use_container_width=True, hide_index=True)

    st.subheader("🔥 今日 AI TOP20")
    if st.button("快速產生 TOP20"):
        with st.spinner("AI 掃描中，抓不到的股票會自動跳過..."):
            top = scan_pool(pool, max_scan).head(20)
        if top.empty:
            st.error("目前抓不到可用資料，請稍後重試或改用自訂池。")
        else:
            show_cols = ["股票代號", "名稱", "現價", "爆發指數", "預估發動時間", "距離突破%", "RSI", "量比", "型態", "停損參考", "目標1", "AI一句話"]
            st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

elif page == "單股掃描":
    st.subheader("🔍 單股 AI 掃描")
    code = st.text_input("輸入股票代號", value="1714")
    if st.button("AI 分析"):
        res = analyze_stock(code)
        if not res or res.get("狀態") != "OK":
            st.error(f"{code} 抓不到可分析資料：{res.get('狀態') if res else '無資料'}")
        else:
            st.success(res["AI一句話"])
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("現價", res["現價"])
            c2.metric("爆發指數", res["爆發指數"])
            c3.metric("發動時間", res["預估發動時間"])
            c4.metric("距突破%", res["距離突破%"])
            c5.metric("風險", res["風險"])
            detail = pd.DataFrame([{k:v for k,v in res.items() if not k.startswith("_") and k not in ["AI一句話", "Ticker", "狀態"]}])
            st.dataframe(detail, use_container_width=True, hide_index=True)
            st.plotly_chart(make_k_chart(res["_df"].tail(120), f"{code} {res['名稱']} K線"), use_container_width=True)

elif page == "全池掃描":
    st.subheader("🌏 全池 AI 掃描")
    st.write(f"目前股票池：{len(pool)} 檔；本次最多掃描：{max_scan} 檔。")
    if st.button("開始全池掃描"):
        with st.spinner("掃描中，請等一下..."):
            df = scan_pool(pool, max_scan)
        if df.empty:
            st.error("掃描完成，但沒有取得可用資料。")
        else:
            tabs = st.tabs(["TOP20", "TOP50", "全部結果"])
            cols = ["股票代號", "名稱", "現價", "爆發指數", "預估發動時間", "距離突破%", "RSI", "量比", "型態", "停損參考", "目標1", "目標2", "AI一句話"]
            tabs[0].dataframe(df.head(20)[cols], use_container_width=True, hide_index=True)
            tabs[1].dataframe(df.head(50)[cols], use_container_width=True, hide_index=True)
            tabs[2].dataframe(df[cols], use_container_width=True, hide_index=True)

else:
    st.subheader("📚 歷史回測示範")
    st.info("這頁先提供可運作的示範：正式回測會依每日TOP20紀錄累積後計算。")
    sample = pd.DataFrame({
        "推薦日": [datetime.now().strftime("%Y-%m-%d")],
        "股票": ["1714 和桐"],
        "推薦原因": ["平台整理/主升段觀察"],
        "5日最高漲幅": ["待累積"],
        "10日最高漲幅": ["待累積"],
        "結果": ["觀察中"]
    })
    st.dataframe(sample, use_container_width=True, hide_index=True)
