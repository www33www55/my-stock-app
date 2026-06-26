import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    import twstock
except Exception:
    twstock = None

st.set_page_config(page_title="未來小股神 AI 操盤中心 V31 REAL", layout="wide")

FALLBACK_STOCKS = {
    "7828":"創新服務", "1714":"和桐", "2409":"友達", "2303":"聯電", "6271":"同欣電",
    "6191":"精成科", "3557":"逸昌", "3037":"欣興", "2382":"廣達", "2313":"華通",
    "2344":"華邦電", "2359":"所羅門", "3060":"銘異", "8923":"時報", "6272":"驊陞",
    "5468":"凱鈺", "6259":"百徽", "5211":"蒙恬", "1730":"花仙子", "3567":"逸昌",
    "4183":"福永生技", "5469":"瀚宇博", "3064":"泰偉", "2643":"捷迅", "8183":"精星",
    "1817":"凱撒衛", "4554":"橙的", "2013":"中鋼構", "1731":"美吾華", "1240":"茂生農經",
    "2739":"寒舍", "3479":"安勤", "3717":"聯嘉投控"
}

@st.cache_data(ttl=86400)
def get_stock_universe() -> Dict[str, str]:
    data = dict(FALLBACK_STOCKS)
    if twstock:
        try:
            for code, info in twstock.codes.items():
                if len(code) == 4 and code.isdigit() and getattr(info, "type", "") == "股票":
                    data[code] = info.name
        except Exception:
            pass
    return dict(sorted(data.items()))

def suffix(code: str) -> str:
    # yfinance: many OTC work with .TWO, listed with .TW. Try .TW first then .TWO
    return f"{code}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def load_price(code: str, period="9mo") -> pd.DataFrame:
    for suf in [".TW", ".TWO"]:
        try:
            df = yf.download(f"{code}{suf}", period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
            if df is not None and len(df) > 40:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna().copy()
                df["Code"] = code
                return df
        except Exception:
            continue
    return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"].replace(0, np.nan)
    for n in [5,10,20,60,120,240]:
        d[f"MA{n}"] = c.rolling(n).mean()
    # RSI
    delta = c.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + rs))
    # KD
    low9 = l.rolling(9).min(); high9 = h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9) * 100
    d["K"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1/3, adjust=False).mean()
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26; d["MACD"] = d["DIF"].ewm(span=9, adjust=False).mean(); d["OSC"] = d["DIF"] - d["MACD"]
    # Bollinger
    d["BB_M"] = c.rolling(20).mean(); std = c.rolling(20).std(); d["BB_U"] = d["BB_M"] + 2*std; d["BB_L"] = d["BB_M"] - 2*std
    # ATR
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    # ADX simplified
    up = h.diff(); down = -l.diff(); plus_dm = np.where((up > down) & (up > 0), up, 0); minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr = d["ATR"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).sum() / atr
    d["ADX"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(14).mean()
    # OBV, VWAP, ROC, Volume ratio
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    d["VWAP"] = ((h+l+c)/3 * d["Volume"]).rolling(20).sum() / d["Volume"].rolling(20).sum()
    d["ROC"] = c.pct_change(12) * 100
    d["VolRatio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return d

def detect_patterns(d: pd.DataFrame) -> Tuple[List[str], str]:
    if len(d) < 60: return [], "資料不足"
    x = d.tail(30); c = x["Close"]
    patterns = []
    high20 = d["High"].rolling(20).max().iloc[-2]
    low20 = d["Low"].rolling(20).min().iloc[-2]
    close = d["Close"].iloc[-1]
    box_range = (high20-low20)/max(low20,1)
    if box_range < 0.16 and close >= low20 + (high20-low20)*0.55:
        patterns.append("平台整理")
    if close >= high20*0.985:
        patterns.append("接近突破")
    if d["MA5"].iloc[-1] > d["MA10"].iloc[-1] > d["MA20"].iloc[-1]:
        patterns.append("多頭排列")
    if d["Close"].iloc[-1] > d["MA20"].iloc[-1] and d["Low"].iloc[-1] <= d["MA10"].iloc[-1]*1.03:
        patterns.append("回踩不破")
    if d["Close"].tail(30).idxmin() < d.tail(10).index[0] and d["MA20"].iloc[-1] > d["MA20"].iloc[-8]:
        patterns.append("圓弧底/轉強")
    if len(patterns) == 0:
        patterns.append("觀察中")
    return patterns, "、".join(patterns)

def score_stock(code: str, name: str) -> Dict:
    raw = load_price(code)
    if raw.empty:
        return {"股票代號": code, "名稱": name, "錯誤": "抓不到資料"}
    d = add_indicators(raw).dropna()
    if len(d) < 30:
        return {"股票代號": code, "名稱": name, "錯誤": "資料不足"}
    r = d.iloc[-1]
    patterns, pattern_text = detect_patterns(d)
    score = 0; reasons = []
    # trend
    if r.MA5 > r.MA10 > r.MA20: score += 18; reasons.append("均線多頭")
    if r.Close > r.MA20: score += 8; reasons.append("站上月線")
    if r.MA20 > d["MA20"].iloc[-6]: score += 8; reasons.append("月線上揚")
    # momentum
    if r.DIF > r.MACD and r.DIF > 0: score += 15; reasons.append("MACD主升")
    if 50 <= r.RSI <= 72: score += 12; reasons.append("RSI健康")
    elif 72 < r.RSI <= 80: score += 5; reasons.append("RSI偏熱")
    elif r.RSI > 80: score -= 8; reasons.append("RSI過熱扣分")
    if r.K > r.D and r.K < 85: score += 8; reasons.append("KD偏多")
    # volume and breakout
    if r.VolRatio >= 1.2: score += 10; reasons.append("量能放大")
    high20 = d["High"].rolling(20).max().iloc[-2]
    dist = max((high20 - r.Close) / r.Close * 100, 0)
    if dist <= 1: score += 14; reasons.append("距突破1%內")
    elif dist <= 3: score += 9; reasons.append("距突破3%內")
    elif dist <= 6: score += 4
    for p in patterns:
        if p in ["平台整理", "接近突破", "回踩不破", "圓弧底/轉強"]:
            score += 6
    # risk
    if r.Close > r.MA20 * 1.18: score -= 8; reasons.append("距月線偏遠")
    score = int(max(0, min(100, score)))
    if score >= 96 and dist <= 1: launch = "今天～1天"
    elif score >= 92 and dist <= 2.5: launch = "1～3天"
    elif score >= 85 and dist <= 5: launch = "2～5天"
    elif score >= 75: launch = "5～10天"
    else: launch = "觀察中"
    stop = min(r.MA20, d["Low"].tail(10).min())
    box = max(high20 - d["Low"].rolling(20).min().iloc[-2], r.ATR*2)
    t1 = r.Close + box*0.8; t2 = r.Close + box*1.3; t3 = r.Close + box*2.0
    ai = f"{pattern_text}，爆發指數{score}，預估{launch}。" + ("線型偏多，可列入觀察。" if score >= 85 else "尚未成熟，等待確認。")
    return {
        "股票代號": code, "名稱": name, "現價": round(float(r.Close),2), "爆發指數": score,
        "預估發動時間": launch, "距離突破%": round(float(dist),2), "RSI": round(float(r.RSI),1),
        "K": round(float(r.K),1), "D": round(float(r.D),1), "MACD狀態": "多" if r.DIF > r.MACD else "弱",
        "量比": round(float(r.VolRatio),2), "MA5": round(float(r.MA5),2), "MA10": round(float(r.MA10),2), "MA20": round(float(r.MA20),2),
        "型態": pattern_text, "停損參考": round(float(stop),2), "目標1": round(float(t1),2), "目標2": round(float(t2),2), "目標3": round(float(t3),2),
        "AI一句話": ai, "入選原因": "、".join(reasons[:6])
    }

def market_summary() -> pd.DataFrame:
    rows = []
    for name, ticker in [("加權指數", "^TWII"), ("櫃買OTC", "^TWOII")]:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            d = add_indicators(df.dropna()).dropna(); r=d.iloc[-1]
            score = 50 + (15 if r.Close>r.MA20 else -10) + (15 if r.MA5>r.MA10>r.MA20 else 0) + (10 if r.DIF>r.MACD else -5) + (10 if 45<r.RSI<75 else -5)
            score = max(0,min(100,int(score)))
            rows.append({"市場":name,"收盤":round(float(r.Close),2),"AI分數":score,"RSI":round(float(r.RSI),1),"MACD":"多" if r.DIF>r.MACD else "弱","AI解讀":"偏多，可正常選股" if score>=70 else "偏保守，降低追價"})
        except Exception:
            rows.append({"市場":name,"收盤":"-","AI分數":"-","RSI":"-","MACD":"-","AI解讀":"資料暫無"})
    return pd.DataFrame(rows)

def k_chart(code: str):
    df = load_price(code)
    if df.empty: return None
    d = add_indicators(df).tail(90)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="K線"))
    for ma in ["MA5","MA10","MA20"]:
        fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=ma, mode="lines"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

universe = get_stock_universe()

st.title("🚀 未來小股神 AI 操盤中心 V31 REAL")
st.caption("真資料版：單股掃描、全池TOP、技術分析、發動時間、股票名稱、7828信仰股")

tab_home, tab_single, tab_pool, tab_market, tab_backtest = st.tabs(["🏠首頁", "🔍單股掃描", "🌏全池掃描", "📊大盤", "📚回測紀錄"])

with tab_home:
    st.subheader("❤️ 7828 信仰股")
    faith = score_stock("7828", universe.get("7828","創新服務"))
    if "錯誤" in faith:
        st.warning(f"7828 暫時抓不到資料：{faith['錯誤']}")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("爆發指數", faith["爆發指數"])
        c2.metric("預估發動", faith["預估發動時間"])
        c3.metric("現價", faith["現價"])
        c4.metric("停損參考", faith["停損參考"])
        st.info(faith["AI一句話"])
    st.subheader("📊 AI 大盤分析")
    st.dataframe(market_summary(), use_container_width=True, hide_index=True)
    st.subheader("🔥 今日 AI TOP20")
    default_codes = list(FALLBACK_STOCKS.keys())[:30]
    if st.button("快速產生 TOP20（示範池）", key="home_top"):
        rows=[]; bar=st.progress(0)
        for i,code in enumerate(default_codes):
            rows.append(score_stock(code, universe.get(code, code))); bar.progress((i+1)/len(default_codes))
        res = pd.DataFrame([r for r in rows if "錯誤" not in r]).sort_values("爆發指數", ascending=False).head(20)
        st.dataframe(res, use_container_width=True, hide_index=True)

with tab_single:
    st.header("🔍 單股 AI 掃描")
    q = st.text_input("輸入股票代號或名稱", value="7828")
    code = q.strip()
    if not code.isdigit():
        matches = [c for c,n in universe.items() if q in n]
        code = matches[0] if matches else code
    if st.button("AI 分析單股"):
        name = universe.get(code, code)
        r = score_stock(code, name)
        if "錯誤" in r:
            st.error(r["錯誤"])
        else:
            st.subheader(f"{code} {name}")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("現價", r["現價"]); c2.metric("爆發指數", r["爆發指數"]); c3.metric("發動時間", r["預估發動時間"]); c4.metric("距突破%", r["距離突破%"]); c5.metric("RSI", r["RSI"])
            st.info(r["AI一句話"])
            st.dataframe(pd.DataFrame([r]), use_container_width=True, hide_index=True)
            fig = k_chart(code)
            if fig: st.plotly_chart(fig, use_container_width=True)

with tab_pool:
    st.header("🌏 全池 AI 掃描")
    mode = st.radio("掃描模式", ["快速示範池", "上市上櫃全池"], horizontal=True)
    topn = st.selectbox("顯示", [20,50,100,300], index=0)
    max_scan = st.slider("最多掃描檔數（手機建議100～300；全池可拉到全部）", 30, len(universe), min(200, len(universe)), 10)
    custom = st.text_area("可貼自訂股票池（逗號分隔）；留空則依模式掃描", "")
    if st.button("🚀 開始掃描"):
        if custom.strip():
            codes = [x.strip() for x in custom.replace("\n",",").split(",") if x.strip()]
        elif mode == "快速示範池":
            codes = list(FALLBACK_STOCKS.keys())
        else:
            codes = list(universe.keys())[:max_scan]
        rows=[]; bar=st.progress(0); status=st.empty()
        for i,code in enumerate(codes):
            status.write(f"掃描中 {i+1}/{len(codes)}：{code} {universe.get(code,'')}")
            rows.append(score_stock(code, universe.get(code, code)))
            bar.progress((i+1)/len(codes))
        df = pd.DataFrame([r for r in rows if "錯誤" not in r])
        if df.empty:
            st.error("沒有成功取得資料，請稍後再試或改自訂股票池。")
        else:
            df = df.sort_values(["爆發指數","距離突破%"], ascending=[False, True]).head(topn)
            st.success(f"完成：成功分析 {len(df)} 檔，顯示 TOP{topn}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("下載CSV", df.to_csv(index=False).encode("utf-8-sig"), "ai_top.csv", "text/csv")

with tab_market:
    st.header("📊 AI 大盤中心")
    st.dataframe(market_summary(), use_container_width=True, hide_index=True)
    st.caption("三大法人完整自動抓取需要接 TWSE/TPEX 官方資料 API；本REAL版先完成大盤技術面，後續可接法人資料。")

with tab_backtest:
    st.header("📚 歷史回測紀錄")
    st.info("本版先保留頁面與CSV匯出架構。回測資料會在下一版接入每日TOP20紀錄後自動累積。")
