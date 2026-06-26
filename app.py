# -*- coding: utf-8 -*-
"""
未來小股神 AI 操盤中心 - Serious Project v1
目標：穩定、可部署、不中斷。先把資料、單股、全池、評分、大盤做對。
"""
from __future__ import annotations

import io
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="未來小股神 AI 操盤中心", layout="wide", page_icon="📈")

APP_VERSION = "Serious Project v1.0"
FAITH_CODE = "7828"

def load_tw_market():

    twse = get_twse_stock()
    tpex = get_tpex_stock()

    return twse + tpex


stock_pool = load_tw_market()

# -----------------------------
# 資料層：股票池 / 歷史價格 / 大盤
# -----------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_twse_list() -> pd.DataFrame:
    """上市清單。失敗時回傳空表，避免 App 爆掉。"""
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = []
        for x in data:
            code = str(x.get("公司代號", "")).strip()
            name = str(x.get("公司簡稱", x.get("公司名稱", ""))).strip()
            if code.isdigit() and len(code) == 4:
                rows.append({"代號": code, "名稱": name, "市場": "TW"})
        return pd.DataFrame(rows).drop_duplicates("代號")
    except Exception:
        return pd.DataFrame(columns=["代號", "名稱", "市場"])

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_tpex_list() -> pd.DataFrame:
    """上櫃清單。TPEX 來源常變，提供多層防呆。"""
    urls = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_company",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            rows = []
            for x in data:
                code = str(x.get("SecuritiesCompanyCode", x.get("公司代號", ""))).strip()
                name = str(x.get("CompanyShortName", x.get("公司簡稱", x.get("公司名稱", "")))).strip()
                if code.isdigit() and len(code) == 4:
                    rows.append({"代號": code, "名稱": name, "市場": "TWO"})
            df = pd.DataFrame(rows).drop_duplicates("代號")
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["代號", "名稱", "市場"])

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_stock_pool() -> pd.DataFrame:
    twse = load_twse_list()
    tpex = load_tpex_list()
    df = pd.concat([twse, tpex], ignore_index=True).drop_duplicates("代號")
    # 若官方清單暫時抓不到，至少保留核心池，讓系統不會空白
    fallback = pd.DataFrame(DEFAULT_POOL, columns=["代號", "名稱", "市場"])
    if df.empty or len(df) < 500:
        df = fallback
    else:
        # 確保 7828 等常用代號存在
        df = pd.concat([df, fallback], ignore_index=True).drop_duplicates("代號")
    return df.sort_values("代號").reset_index(drop=True)

def yf_symbol(code: str, market: str) -> str:
    return f"{code}.TW" if market == "TW" else f"{code}.TWO"

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_price(code: str, market: str, period: str = "6mo") -> pd.DataFrame:
    """抓價格。先用市場別，再雙後綴備援。永遠回傳 DataFrame，不拋例外。"""
    suffixes = [yf_symbol(code, market), f"{code}.TW", f"{code}.TWO"]
    seen = []
    for symbol in suffixes:
        if symbol in seen:
            continue
        seen.append(symbol)
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns and df["Close"].dropna().shape[0] >= 35:
                return df.dropna(subset=["Close"]).copy()
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_index(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"]) if "Close" in df.columns else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# -----------------------------
# 指標層
# -----------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def kd(df: pd.DataFrame, n: int = 9) -> Tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(n).min()
    high_max = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def compute_features(df: pd.DataFrame) -> Dict:
    c = df["Close"]
    v = df["Volume"] if "Volume" in df.columns else pd.Series([0]*len(df), index=df.index)
    out = {}
    for n in [5, 10, 20, 60, 120, 240]:
        out[f"MA{n}"] = safe_float(c.rolling(n).mean().iloc[-1]) if len(c) >= n else safe_float(c.mean())
    out["Close"] = safe_float(c.iloc[-1])
    out["PrevClose"] = safe_float(c.iloc[-2]) if len(c) >= 2 else out["Close"]
    out["RSI"] = safe_float(rsi(c).iloc[-1], 50)
    k, d = kd(df)
    out["K"] = safe_float(k.iloc[-1], 50)
    out["D"] = safe_float(d.iloc[-1], 50)
    dif, dea, hist = macd(c)
    out["DIF"] = safe_float(dif.iloc[-1])
    out["DEA"] = safe_float(dea.iloc[-1])
    out["MACDHist"] = safe_float(hist.iloc[-1])
    out["MACDHistPrev"] = safe_float(hist.iloc[-2]) if len(hist) >= 2 else out["MACDHist"]
    out["Vol"] = safe_float(v.iloc[-1])
    out["VolMA20"] = safe_float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else safe_float(v.mean())
    out["VolRatio"] = out["Vol"] / out["VolMA20"] if out["VolMA20"] else 0
    out["High60"] = safe_float(c.rolling(60).max().iloc[-1]) if len(c) >= 60 else safe_float(c.max())
    out["DistHigh60"] = (out["High60"] - out["Close"]) / out["High60"] * 100 if out["High60"] else 999
    out["Ret1"] = (out["Close"] - out["PrevClose"]) / out["PrevClose"] * 100 if out["PrevClose"] else 0
    # 平台：20日高低區間不要太寬
    if len(c) >= 20:
        h20, l20 = c.tail(20).max(), c.tail(20).min()
        out["Range20Pct"] = safe_float((h20 - l20) / l20 * 100 if l20 else 999)
    else:
        out["Range20Pct"] = 999
    return out

# -----------------------------
# AI 評分層：不再全部 100，也不再 0 分太嚴格
# -----------------------------
def score_stock(feat: Dict) -> Dict:
    score = 0
    reasons: List[str] = []
    risks: List[str] = []

    close = feat["Close"]
    ma5, ma10, ma20, ma60 = feat["MA5"], feat["MA10"], feat["MA20"], feat["MA60"]
    r = feat["RSI"]
    k, d = feat["K"], feat["D"]
    dif, dea, hist, hist_prev = feat["DIF"], feat["DEA"], feat["MACDHist"], feat["MACDHistPrev"]
    vol_ratio = feat["VolRatio"]
    dist60 = feat["DistHigh60"]
    range20 = feat["Range20Pct"]

    # 技術 45
    if ma5 > ma10 > ma20:
        score += 15; reasons.append("均線多頭排列")
    elif close > ma20:
        score += 8; reasons.append("站上月線")
    if close > ma60:
        score += 5; reasons.append("站上季線")
    if 45 <= r <= 72:
        score += 12; reasons.append(f"RSI健康 {r:.1f}")
    elif 72 < r <= 80:
        score += 5; reasons.append(f"RSI偏熱但可接受 {r:.1f}")
    if dif > dea:
        score += 10; reasons.append("MACD多方")
    if hist > hist_prev:
        score += 5; reasons.append("MACD動能增加")
    if k > d:
        score += 5; reasons.append("KD偏多")

    # 量價 25
    if 1.2 <= vol_ratio <= 3.5:
        score += 12; reasons.append(f"量能放大 {vol_ratio:.2f}倍")
    elif 0.8 <= vol_ratio < 1.2:
        score += 5; reasons.append("量能穩定")
    if dist60 <= 3:
        score += 8; reasons.append("接近60日高點")
    elif dist60 <= 8:
        score += 4; reasons.append("距高點不遠")
    if feat["Ret1"] > 0 and vol_ratio >= 1:
        score += 5; reasons.append("價漲量穩")

    # 型態 20
    if range20 <= 10:
        score += 10; reasons.append("平台整理")
    elif range20 <= 18:
        score += 5; reasons.append("區間整理")
    if ma5 >= ma10 and ma20 >= ma60:
        score += 5; reasons.append("主升段結構")
    if close >= ma5 and dist60 <= 10:
        score += 5; reasons.append("靠近突破區")

    # 籌碼 10（v1 無外部法人資料，先用 OBV/量價代理，不假裝有真法人）
    if feat["Ret1"] >= 0 and vol_ratio >= 1.1:
        score += 6; reasons.append("資金承接跡象")
    if close >= ma20:
        score += 4; reasons.append("籌碼未明顯轉弱")

    # 扣分
    if r > 85:
        score -= 15; risks.append("RSI過熱")
    elif r > 80:
        score -= 8; risks.append("RSI偏高")
    if close < ma20:
        score -= 12; risks.append("跌破月線")
    if close < ma5:
        score -= 6; risks.append("跌破5日線")
    if feat["Ret1"] < -3 and vol_ratio > 1.5:
        score -= 18; risks.append("放量長黑風險")
    if vol_ratio > 5:
        score -= 8; risks.append("爆量過大需確認")

    score = int(max(0, min(100, round(score))))

    # AI信心獨立計算：分數 + 風險數校正，不直接複製分數
    confidence = max(45, min(96, int(score * 0.82 + 18 - len(risks) * 4)))

    if score >= 88:
        launch = "1~3天"
    elif score >= 72:
        launch = "2~5天"
    elif score >= 58:
        launch = "5~10天"
    elif r > 82:
        launch = "高風險"
    else:
        launch = "觀察中"

    stars = "⭐" * max(1, min(5, math.ceil(score / 20)))
    summary = "、".join(reasons[:3]) if reasons else "資料偏弱，先觀察"
    if risks:
        summary += "；風險：" + "、".join(risks[:2])

    return {
        "爆發指數": score,
        "AI信心": confidence,
        "發動時間": launch,
        "星級": stars,
        "AI解讀": summary,
        "加分原因": "、".join(reasons),
        "風險": "、".join(risks) if risks else "低",
    }

def analyze_one(code: str, name: str, market: str) -> Dict:
    df = fetch_price(code, market)
    if df.empty:
        return {"代號": code, "名稱": name, "市場": market, "狀態": "資料不足"}
    feat = compute_features(df)
    sc = score_stock(feat)
    return {
        "代號": code,
        "名稱": name,
        "市場": market,
        "收盤": round(feat["Close"], 2),
        "爆發指數": sc["爆發指數"],
        "AI信心": sc["AI信心"],
        "發動時間": sc["發動時間"],
        "星級": sc["星級"],
        "RSI": round(feat["RSI"], 1),
        "量比": round(feat["VolRatio"], 2),
        "MACD": "多" if feat["DIF"] > feat["DEA"] else "空",
        "AI解讀": sc["AI解讀"],
        "風險": sc["風險"],
        "狀態": "OK",
    }

def market_tech(symbol: str, label: str) -> Dict:
    df = fetch_index(symbol)
    if df.empty:
        return {"市場": label, "狀態": "資料不足"}
    feat = compute_features(df)
    score = 0
    notes = []
    if feat["Close"] > feat["MA20"]: score += 25; notes.append("站上月線")
    if feat["MA5"] > feat["MA10"] > feat["MA20"]: score += 25; notes.append("短均多頭")
    if 45 <= feat["RSI"] <= 70: score += 20; notes.append("RSI健康")
    if feat["DIF"] > feat["DEA"]: score += 20; notes.append("MACD偏多")
    if feat["VolRatio"] >= 0.8: score += 10; notes.append("量能正常")
    temp = "偏多" if score >= 70 else "中性偏多" if score >= 55 else "保守"
    return {
        "市場": label,
        "收盤": round(feat["Close"], 2),
        "技術分": int(score),
        "RSI": round(feat["RSI"], 1),
        "MACD": "多" if feat["DIF"] > feat["DEA"] else "空",
        "量比": round(feat["VolRatio"], 2),
        "AI判斷": f"{temp}：{'、'.join(notes[:3])}",
        "狀態": "OK",
    }

# -----------------------------
# UI
# -----------------------------
st.title("🚀 未來小股神 AI 操盤中心")
st.caption(f"{APP_VERSION}｜先求穩定，再求完整。不是全部100分，不是假全池。")

pool = load_stock_pool()

with st.sidebar:
    st.header("設定")
    st.write(f"股票池載入：**{len(pool)} 檔**")
    max_scan = st.number_input("最多掃描檔數（0=全池，雲端很慢）", min_value=0, max_value=3000, value=200, step=50)
    min_score = st.slider("最低爆發指數", 0, 100, 0)
    st.info("建議先掃 200～500 檔確認穩定；全池可設 0，但會很慢。")

tab_home, tab_single, tab_pool, tab_market = st.tabs(["🏠 首頁", "🔍 單股掃描", "🌏 全池掃描", "📊 大盤技術"])

with tab_home:
    st.subheader("❤️ 7828 信仰股")
    faith_row = pool[pool["代號"] == FAITH_CODE]
    if faith_row.empty:
        faith = analyze_one("7828", "創新服務", "TWO")
    else:
        r0 = faith_row.iloc[0]
        faith = analyze_one(r0["代號"], r0["名稱"], r0["市場"])
    if faith.get("狀態") == "OK":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("爆發指數", faith["爆發指數"])
        c2.metric("AI信心", f"{faith['AI信心']}%")
        c3.metric("發動時間", faith["發動時間"])
        c4.metric("RSI", faith["RSI"])
        st.write(f"**AI：** {faith['AI解讀']}")
    else:
        st.warning("7828 暫時資料不足；系統不會爆掉，請稍後重試或確認資料源。")

    st.subheader("📊 AI 大盤快速評估")
    m1, m2 = market_tech("^TWII", "加權指數"), market_tech("^TWOII", "櫃買OTC")
    st.dataframe(pd.DataFrame([m1, m2]), use_container_width=True)

with tab_single:
    st.subheader("🔍 單股 AI 掃描")
    q = st.text_input("輸入股票代號或名稱", value="7828")
    if st.button("開始分析", type="primary"):
        q = q.strip()
        hit = pool[(pool["代號"] == q) | (pool["名稱"].str.contains(q, na=False))]
        if hit.empty and q.isdigit():
            hit = pd.DataFrame([{"代號": q, "名稱": q, "市場": "TW"}])
        if hit.empty:
            st.error("找不到股票")
        else:
            row = hit.iloc[0]
            res = analyze_one(row["代號"], row["名稱"], row["市場"])
            if res.get("狀態") != "OK":
                st.warning(f"{row['代號']} {row['名稱']} 資料不足")
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("爆發指數", res["爆發指數"])
                c2.metric("AI信心", f"{res['AI信心']}%")
                c3.metric("發動時間", res["發動時間"])
                c4.metric("RSI", res["RSI"])
                c5.metric("量比", res["量比"])
                st.write(f"### {res['代號']} {res['名稱']} {res['星級']}")
                st.write(res["AI解讀"])
                dfp = fetch_price(row["代號"], row["市場"])
                if not dfp.empty:
                    st.line_chart(dfp["Close"].tail(90))
                st.dataframe(pd.DataFrame([res]), use_container_width=True)

with tab_pool:
    st.subheader("🌏 全池 AI 掃描")
    st.write(f"目前股票池：**{len(pool)} 檔**。掃描上限：**{'全池' if max_scan == 0 else max_scan}**")
    if st.button("開始全池掃描", type="primary"):
        scan_df = pool.copy()
        if max_scan and max_scan > 0:
            scan_df = scan_df.head(int(max_scan))
        total = len(scan_df)
        progress = st.progress(0)
        status = st.empty()
        rows = []
        start = time.time()
        for i, row in scan_df.iterrows():
            idx = len(rows) + 1
            status.write(f"掃描中：{i+1}/{total}　{row['代號']} {row['名稱']}")
            res = analyze_one(row["代號"], row["名稱"], row["市場"])
            if res.get("狀態") == "OK" and res.get("爆發指數", 0) >= min_score:
                rows.append(res)
            progress.progress(min(1.0, (i + 1) / total))
        elapsed = time.time() - start
        result = pd.DataFrame(rows)
        if result.empty:
            st.warning("沒有符合條件的股票，請降低最低爆發指數或稍後重試資料源。")
        else:
            result = result.sort_values(["爆發指數", "AI信心"], ascending=False).reset_index(drop=True)
            result.insert(0, "排名", result.index + 1)
            st.success(f"掃描完成：有效 {len(result)} 檔 / 掃描 {total} 檔，耗時 {elapsed:.1f} 秒")
            st.dataframe(result.head(20), use_container_width=True, height=600)
            csv = result.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下載完整掃描結果 CSV", csv, "future_stock_ai_scan.csv", "text/csv")

with tab_market:
    st.subheader("📊 大盤技術評估")
    data = pd.DataFrame([market_tech("^TWII", "加權指數"), market_tech("^TWOII", "櫃買OTC")])
    st.dataframe(data, use_container_width=True)
    st.caption("大盤資料來源為 Yahoo 指數；若雲端暫時抓不到，會顯示資料不足，不會讓 App 掛掉。")

st.divider()
st.caption("投資有風險。本工具為資料整理與策略輔助，不構成買賣建議。")
