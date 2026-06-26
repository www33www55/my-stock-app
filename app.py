import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="未來小股神 AI 操盤中心 Restore", layout="wide")

FALLBACK_POOL = [
    {"code":"2330","name":"台積電","market":"TW"}, {"code":"2303","name":"聯電","market":"TW"},
    {"code":"2409","name":"友達","market":"TW"}, {"code":"2313","name":"華通","market":"TW"},
    {"code":"2382","name":"廣達","market":"TW"}, {"code":"3037","name":"欣興","market":"TW"},
    {"code":"6271","name":"同欣電","market":"TW"}, {"code":"6272","name":"驊陞","market":"TW"},
    {"code":"6191","name":"精成科","market":"TW"}, {"code":"3557","name":"逸昌","market":"TWO"},
    {"code":"3567","name":"逸昌","market":"TWO"}, {"code":"1714","name":"和桐","market":"TW"},
    {"code":"7828","name":"創新服務","market":"TWO"}, {"code":"5468","name":"凱鈺","market":"TWO"},
    {"code":"6259","name":"百徽","market":"TWO"}, {"code":"5211","name":"蒙恬","market":"TWO"},
    {"code":"1730","name":"花仙子","market":"TW"}, {"code":"4183","name":"福永生技","market":"TWO"},
    {"code":"5469","name":"瀚宇博","market":"TW"}, {"code":"8183","name":"精星","market":"TW"},
    {"code":"2643","name":"捷迅","market":"TW"}, {"code":"1817","name":"凱撒衛","market":"TW"},
    {"code":"2013","name":"中鋼構","market":"TW"}, {"code":"1731","name":"美吾華","market":"TW"},
    {"code":"3479","name":"安勤","market":"TWO"}, {"code":"2739","name":"寒舍","market":"TW"},
    {"code":"4554","name":"橙的","market":"TWO"}, {"code":"1240","name":"茂生農經","market":"TWO"},
]

@st.cache_data(ttl=3600, show_spinner=False)
def get_twse_stock() -> List[Dict[str, str]]:
    """上市股票池：使用 TWSE OpenAPI。"""
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        data = requests.get(url, timeout=15).json()
        stocks = []
        for x in data:
            code = str(x.get("Code", "")).strip()
            name = str(x.get("Name", "")).strip()
            if code.isdigit() and len(code) == 4 and name:
                stocks.append({"code": code, "name": name, "market": "TW"})
        return stocks
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_tpex_stock() -> List[Dict[str, str]]:
    """上櫃股票池：使用 TPEX OpenAPI。欄位有時會異動，所以做多欄位備援。"""
    urls = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
    ]
    for url in urls:
        try:
            data = requests.get(url, timeout=15).json()
            stocks = []
            for x in data:
                code = str(
                    x.get("SecuritiesCompanyCode") or x.get("SecuritiesCode") or x.get("Code") or x.get("代號") or ""
                ).strip()
                name = str(
                    x.get("CompanyName") or x.get("SecuritiesCompanyName") or x.get("Name") or x.get("名稱") or ""
                ).strip()
                if code.isdigit() and len(code) == 4 and name:
                    stocks.append({"code": code, "name": name, "market": "TWO"})
            if stocks:
                return stocks
        except Exception:
            pass
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def load_tw_market() -> List[Dict[str, str]]:
    twse = get_twse_stock()
    tpex = get_tpex_stock()
    pool = twse + tpex
    # 去重
    seen = set()
    uniq = []
    for s in pool:
        k = (s["code"], s["market"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq if len(uniq) >= 500 else FALLBACK_POOL

def yf_symbol(code: str, market: str) -> str:
    return f"{code}.TW" if market == "TW" else f"{code}.TWO"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price(code: str, market: str, period: str = "6mo") -> pd.DataFrame:
    symbol = yf_symbol(code, market)
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def kd(df: pd.DataFrame, n: int = 9) -> Tuple[pd.Series, pd.Series]:
    low = df["Low"].rolling(n).min()
    high = df["High"].rolling(n).max()
    rsv = (df["Close"] - low) / (high - low).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d

def analyze_stock(code: str, name: str, market: str) -> Dict:
    df = fetch_price(code, market)
    if df.empty or len(df) < 35:
        return {"代號": code, "名稱": name, "市場": market, "狀態": "資料不足", "爆發指數": 0, "AI信心": "50%", "發動時間": "觀察中"}

    c = df["Close"]
    v = df["Volume"] if "Volume" in df else pd.Series(index=df.index, data=0)
    close = float(c.iloc[-1])
    ma5 = c.rolling(5).mean().iloc[-1]
    ma10 = c.rolling(10).mean().iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1] if len(c) >= 60 else np.nan
    r = float(rsi(c).iloc[-1]) if not np.isnan(rsi(c).iloc[-1]) else 50.0
    dif, dea, hist = macd(c)
    macd_val = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
    k, d = kd(df)
    k_last = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0
    d_last = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
    vol_ma20 = v.rolling(20).mean().iloc[-1] if len(v) >= 20 else np.nan
    vol_ratio = float(v.iloc[-1] / vol_ma20) if vol_ma20 and not np.isnan(vol_ma20) and vol_ma20 > 0 else 1.0
    high60 = c.rolling(60).max().iloc[-1] if len(c) >= 60 else c.max()
    distance_high = float((high60 - close) / high60 * 100) if high60 else 999

    score = 0
    reasons = []
    risks = []

    if ma5 > ma10 > ma20:
        score += 20; reasons.append("均線多頭")
    elif close > ma20:
        score += 8; reasons.append("站上月線")

    if 45 <= r <= 72:
        score += 12; reasons.append("RSI健康")
    elif 72 < r <= 80:
        score += 5; reasons.append("RSI偏熱")

    if macd_val > 0:
        score += 15; reasons.append("MACD偏多")
    if k_last > d_last and k_last < 85:
        score += 8; reasons.append("KD偏多")
    if vol_ratio >= 1.5:
        score += 15; reasons.append("量能放大")
    elif vol_ratio >= 1.1:
        score += 7; reasons.append("量能溫和")
    if distance_high <= 3:
        score += 12; reasons.append("接近60日高")
    elif distance_high <= 8:
        score += 6; reasons.append("接近壓力區")

    # 平台整理粗略判斷：20日波動區間不過大且接近區間高點
    recent20 = c.tail(20)
    range_pct = (recent20.max() - recent20.min()) / recent20.min() * 100 if recent20.min() else 999
    if range_pct <= 12 and close >= recent20.quantile(0.65):
        score += 10; reasons.append("平台整理")

    # 風險扣分
    if r > 80:
        score -= 12; risks.append("RSI過熱")
    if close < ma5:
        score -= 8; risks.append("跌破5MA")
    if macd_val < 0:
        score -= 8; risks.append("MACD偏弱")
    if vol_ratio > 3 and close < float(df["Open"].iloc[-1]):
        score -= 15; risks.append("爆量長黑")

    score = int(max(0, min(100, score)))
    confidence = int(max(50, min(95, score + 5 if score >= 70 else score)))
    if score >= 90:
        timing = "1～3天"
        stars = "⭐⭐⭐⭐⭐"
    elif score >= 75:
        timing = "2～5天"
        stars = "⭐⭐⭐⭐"
    elif score >= 60:
        timing = "5～10天"
        stars = "⭐⭐⭐"
    elif score >= 40:
        timing = "觀察中"
        stars = "⭐⭐"
    else:
        timing = "觀察中"
        stars = "⭐"

    ai_text = "、".join(reasons[:4]) if reasons else "等待型態確認"
    if risks:
        ai_text += "；風險：" + "、".join(risks[:2])

    return {
        "代號": code, "名稱": name, "市場": market, "收盤": round(close, 2),
        "爆發指數": score, "AI信心": f"{confidence}%", "發動時間": timing, "評等": stars,
        "RSI": round(r, 1), "MACD": round(macd_val, 3), "量比": round(vol_ratio, 2),
        "MA5": round(float(ma5), 2) if not np.isnan(ma5) else None,
        "MA10": round(float(ma10), 2) if not np.isnan(ma10) else None,
        "MA20": round(float(ma20), 2) if not np.isnan(ma20) else None,
        "距60高%": round(distance_high, 2), "AI解讀": ai_text, "狀態": "OK"
    }

def market_row(label: str, symbol: str) -> Dict:
    try:
        df = yf.download(symbol, period="6mo", progress=False, threads=False)
        if df is None or df.empty or len(df) < 35:
            return {"市場": label, "收盤": "-", "AI分數": "-", "RSI": "-", "MACD": "-", "AI解讀": "資料不足"}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        c = df["Close"].dropna()
        ma5, ma20 = c.rolling(5).mean().iloc[-1], c.rolling(20).mean().iloc[-1]
        r = float(rsi(c).iloc[-1]) if not np.isnan(rsi(c).iloc[-1]) else 50.0
        _, _, hist = macd(c)
        m = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
        score = 50 + (10 if c.iloc[-1] > ma20 else -5) + (10 if ma5 > ma20 else 0) + (10 if m > 0 else -5) + (5 if 45 <= r <= 70 else 0)
        score = int(max(0, min(100, score)))
        text = "偏多震盪，可選股" if score >= 70 else "中性震盪，保守選股" if score >= 55 else "偏弱，降低追高"
        return {"市場": label, "收盤": round(float(c.iloc[-1]), 2), "AI分數": score, "RSI": round(r, 1), "MACD": round(m, 3), "AI解讀": text}
    except Exception:
        return {"市場": label, "收盤": "-", "AI分數": "-", "RSI": "-", "MACD": "-", "AI解讀": "資料不足"}

def scan_pool(pool: List[Dict[str, str]], limit: int) -> pd.DataFrame:
    if limit and limit > 0:
        pool = pool[:limit]
    total = len(pool)
    progress = st.progress(0)
    status = st.empty()
    rows = []
    start = time.time()
    for i, s in enumerate(pool, start=1):
        status.write(f"掃描中：{i}/{total}　{s['code']} {s['name']}")
        row = analyze_stock(s["code"], s["name"], s["market"])
        if row.get("狀態") == "OK":
            rows.append(row)
        progress.progress(i / total if total else 1.0)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["爆發指數", "RSI"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "排名", range(1, len(df) + 1))
    st.success(f"掃描完成：有效 {len(rows)} 檔 / 股票池 {total} 檔，耗時 {time.time()-start:.1f} 秒")
    return df

# ---- UI ----
st.title("🚀 未來小股神 AI 操盤中心 Restore")
st.caption("穩定版：真股票池、單股掃描、全池 TOP20、大盤技術評估。")

tabs = st.tabs(["🔥 全池TOP20", "🔍 單股掃描", "📋 股票池", "❤️ 7828信仰股", "📊 大盤"])

with tabs[0]:
    st.header("🌏 全池 AI 掃描")
    stock_pool = load_tw_market()
    limit = st.number_input("掃描上限（0=全池；建議免費雲先用100～200）", min_value=0, max_value=2500, value=200, step=50)
    st.write(f"目前股票池：{len(stock_pool)} 檔；掃描上限：{limit if limit else '全池'}")
    if st.button("開始全池掃描 / 更新TOP20", type="primary"):
        df = scan_pool(stock_pool, int(limit))
        if df.empty:
            st.warning("沒有有效資料，可能是資料源暫時抓不到。")
        else:
            st.dataframe(df.head(20), use_container_width=True)
            st.download_button("下載本次掃描CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="ai_top20.csv", mime="text/csv")

with tabs[1]:
    st.header("🔍 單股 AI 掃描")
    pool = load_tw_market()
    code = st.text_input("輸入股票代號", value="7828")
    if st.button("分析單股"):
        match = next((s for s in pool if s["code"] == code.strip()), None)
        if not match:
            market = st.selectbox("找不到股票池資料，請選市場", ["TW", "TWO"])
            match = {"code": code.strip(), "name": code.strip(), "market": market}
        row = analyze_stock(match["code"], match["name"], match["market"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("爆發指數", row.get("爆發指數"))
        c2.metric("AI信心", row.get("AI信心"))
        c3.metric("發動時間", row.get("發動時間"))
        c4.metric("RSI", row.get("RSI", "-"))
        st.write(row.get("AI解讀", ""))
        st.dataframe(pd.DataFrame([row]), use_container_width=True)

with tabs[2]:
    st.header("📋 股票池")
    pool = load_tw_market()
    st.success(f"已載入股票池：{len(pool)} 檔（上市 + 上櫃；抓不到時使用備援池）")
    st.dataframe(pd.DataFrame(pool).head(200), use_container_width=True)

with tabs[3]:
    st.header("❤️ 7828 信仰股")
    pool = load_tw_market()
    match = next((s for s in pool if s["code"] == "7828"), {"code":"7828", "name":"創新服務", "market":"TWO"})
    row = analyze_stock(match["code"], match["name"], match["market"])
    c1, c2, c3 = st.columns(3)
    c1.metric("爆發", row.get("爆發指數"))
    c2.metric("信心", row.get("AI信心"))
    c3.metric("發動", row.get("發動時間"))
    st.write(row.get("AI解讀", ""))
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

with tabs[4]:
    st.header("📊 AI 大盤技術評估")
    mdf = pd.DataFrame([
        market_row("加權指數", "^TWII"),
        market_row("櫃買OTC", "^TWOII"),
    ])
    st.dataframe(mdf, use_container_width=True)
    st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
