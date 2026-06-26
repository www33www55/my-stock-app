import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="未來小股神 AI 操盤中心 V32.2", layout="wide")

# ---------------------------
# 基本設定
# ---------------------------
DEFAULT_WATCH = "7828, 1714, 2409, 2303, 6271, 6191, 3557, 3037, 2382, 2313, 2344, 2359, 3060, 8923, 6272, 5468, 6259"
HEADERS = {"User-Agent": "Mozilla/5.0"}

FALLBACK_NAME = {
    "7828": "創新服務", "1714": "和桐", "2409": "友達", "2303": "聯電", "6271": "同欣電",
    "6272": "驊陞", "6191": "精成科", "3557": "嘉威", "3037": "欣興", "2382": "廣達",
    "2313": "華通", "2344": "華邦電", "2359": "所羅門", "3060": "銘異", "8923": "時報",
    "5468": "凱鈺", "6259": "百徽", "5211": "蒙恬", "1730": "花仙子", "4183": "福永生技",
    "5469": "瀚宇博", "8183": "精星", "2643": "捷迅", "1817": "凱撒衛", "2013": "中鋼構",
    "1731": "美吾華", "3479": "安勤", "2739": "寒舍", "4554": "橙的", "1240": "茂生農經"
}

# ---------------------------
# 小工具
# ---------------------------
def _num(x, default=np.nan):
    if x is None:
        return default
    try:
        s = str(x).replace(",", "").replace("--", "").replace("-", "").strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _pct(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{x:.2f}%"
    except Exception:
        return "-"


def stock_type(code: str) -> str:
    """簡單判斷上市/上櫃。若不確定，單股會自動 TW/TWO 都試。"""
    code = str(code).strip()
    if code in {"6271", "6191", "3557", "3037", "2382", "2313", "2409", "2303", "1714", "2344", "2359", "3060"}:
        return "TW"
    if code in {"7828", "8923", "5468", "6259", "6272", "5211", "4183", "5469", "8183", "3479", "2739", "4554", "1240"}:
        return "TWO"
    return "AUTO"

# ---------------------------
# 官方日行情：全池清單
# ---------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_twse_all() -> pd.DataFrame:
    urls = [
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=json",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, dict) and "data" in data and "fields" in data:
                df = pd.DataFrame(data["data"], columns=data["fields"])
            else:
                df = pd.DataFrame(data)
            if not df.empty:
                return normalize_daily(df, "上市")
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_tpex_all() -> pd.DataFrame:
    urls = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_esb_latest_statistics",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            df = pd.DataFrame(data)
            if not df.empty:
                return normalize_daily(df, "上櫃")
        except Exception:
            pass
    return pd.DataFrame()


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    # 模糊找
    for c in cols:
        for key in candidates:
            if key.lower() in str(c).lower():
                return c
    return None


def normalize_daily(df: pd.DataFrame, market: str) -> pd.DataFrame:
    code_col = pick_col(df, ["Code", "SecuritiesCompanyCode", "SecuritiesCode", "股票代號", "有價證券代號", "代號"])
    name_col = pick_col(df, ["Name", "CompanyName", "SecuritiesCompanyName", "股票名稱", "有價證券名稱", "名稱"])
    close_col = pick_col(df, ["ClosingPrice", "Close", "收盤價", "收盤", "LatestPrice"])
    open_col = pick_col(df, ["OpeningPrice", "Open", "開盤價", "開盤"])
    high_col = pick_col(df, ["HighestPrice", "High", "最高價", "最高"])
    low_col = pick_col(df, ["LowestPrice", "Low", "最低價", "最低"])
    vol_col = pick_col(df, ["TradeVolume", "TradingShares", "成交股數", "成交量", "Volume"])
    change_col = pick_col(df, ["Change", "漲跌價差", "漲跌", "ChangeAmount"])

    if code_col is None or close_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["代號"] = df[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
    out = out[out["代號"].notna()].copy()
    out["名稱"] = df[name_col].astype(str).values[: len(out)] if name_col else out["代號"].map(FALLBACK_NAME).fillna("")
    out["市場"] = market
    out["現價"] = df.loc[out.index, close_col].apply(_num)
    out["開盤"] = df.loc[out.index, open_col].apply(_num) if open_col else np.nan
    out["最高"] = df.loc[out.index, high_col].apply(_num) if high_col else np.nan
    out["最低"] = df.loc[out.index, low_col].apply(_num) if low_col else np.nan
    out["成交量"] = df.loc[out.index, vol_col].apply(_num) if vol_col else np.nan
    out["漲跌"] = df.loc[out.index, change_col].apply(_num) if change_col else np.nan
    out = out.dropna(subset=["現價"])
    out = out[out["現價"] > 0]
    return out.reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_full_market() -> pd.DataFrame:
    twse = fetch_twse_all()
    tpex = fetch_tpex_all()
    df = pd.concat([twse, tpex], ignore_index=True)
    if df.empty:
        # 最後備援：至少讓 App 能動，但會明確標示是示範池
        rows = []
        for c, n in FALLBACK_NAME.items():
            rows.append({"代號": c, "名稱": n, "市場": "示範池", "現價": np.nan, "開盤": np.nan, "最高": np.nan, "最低": np.nan, "成交量": np.nan, "漲跌": np.nan})
        df = pd.DataFrame(rows)
    df["名稱"] = df.apply(lambda r: r["名稱"] if str(r["名稱"]).strip() else FALLBACK_NAME.get(str(r["代號"]), ""), axis=1)
    return df.drop_duplicates("代號").reset_index(drop=True)

# ---------------------------
# Yahoo chart：單股深度 K 線
# ---------------------------
@st.cache_data(ttl=30 * 60, show_spinner=False)
def yahoo_history(symbol: str, range_: str = "6mo") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval=1d&events=history"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        js = r.json()
        res = js.get("chart", {}).get("result")
        if not res:
            return pd.DataFrame()
        item = res[0]
        ts = item.get("timestamp", [])
        q = item.get("indicators", {}).get("quote", [{}])[0]
        df = pd.DataFrame(q)
        if df.empty or not ts:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(ts, unit="s").tz_localize("UTC").dt.tz_convert("Asia/Taipei").dt.date
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna().reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_stock_history(code: str) -> Tuple[pd.DataFrame, str]:
    code = str(code).strip()
    candidates = []
    typ = stock_type(code)
    if typ == "TW":
        candidates = [f"{code}.TW", f"{code}.TWO"]
    elif typ == "TWO":
        candidates = [f"{code}.TWO", f"{code}.TW"]
    else:
        candidates = [f"{code}.TW", f"{code}.TWO"]
    for sym in candidates:
        df = yahoo_history(sym)
        if len(df) >= 30:
            return df, sym
    return pd.DataFrame(), candidates[0]

# ---------------------------
# 技術分析
# ---------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in [5, 10, 20, 60, 120, 240]:
        d[f"MA{n}"] = c.rolling(n).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    low9 = d["Low"].rolling(9).min()
    high9 = d["High"].rolling(9).max()
    rsv = (d["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    d["K"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1/3, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26
    d["MACD"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["MACD柱"] = d["DIF"] - d["MACD"]
    d["BB_MID"] = c.rolling(20).mean()
    d["BB_UP"] = d["BB_MID"] + 2 * c.rolling(20).std()
    d["BB_LOW"] = d["BB_MID"] - 2 * c.rolling(20).std()
    return d


def analyze_stock(code: str, name: str = "") -> Dict:
    hist, symbol = get_stock_history(code)
    if hist.empty or len(hist) < 30:
        return {"代號": code, "名稱": name or FALLBACK_NAME.get(code, ""), "狀態": "資料不足", "symbol": symbol}
    d = add_indicators(hist)
    last = d.iloc[-1]
    prev = d.iloc[-2]

    score = 50
    reasons = []
    if last["Close"] > last.get("MA5", np.nan) > last.get("MA10", np.nan) > last.get("MA20", np.nan):
        score += 20; reasons.append("均線多頭排列")
    elif last["Close"] > last.get("MA20", np.nan):
        score += 10; reasons.append("站上月線")
    rsi = last.get("RSI", np.nan)
    if 50 <= rsi <= 72:
        score += 12; reasons.append("RSI健康")
    elif rsi > 80:
        score -= 8; reasons.append("RSI過熱")
    if last.get("DIF", 0) > last.get("MACD", 0):
        score += 10; reasons.append("MACD偏多")
    if last.get("MACD柱", 0) > prev.get("MACD柱", 0):
        score += 5; reasons.append("動能增加")
    vol20 = d["Volume"].rolling(20).mean().iloc[-1]
    vol_ratio = last["Volume"] / vol20 if vol20 and not pd.isna(vol20) else np.nan
    if not pd.isna(vol_ratio) and vol_ratio > 1.3:
        score += 8; reasons.append("量能放大")
    high20 = d["High"].rolling(20).max().iloc[-1]
    dist = (high20 - last["Close"]) / high20 * 100 if high20 else np.nan
    if not pd.isna(dist) and dist <= 2.5:
        score += 10; reasons.append("接近20日高點")
    score = int(max(0, min(100, score)))

    if score >= 92:
        launch = "1～3天"
    elif score >= 84:
        launch = "2～5天"
    elif score >= 75:
        launch = "5～10天"
    else:
        launch = "觀察中"

    support = np.nanmin([last.get("MA10", np.nan), last.get("MA20", np.nan), d["Low"].tail(10).min()])
    resistance = d["High"].tail(20).max()
    target1 = last["Close"] * 1.08
    target2 = last["Close"] * 1.16

    return {
        "代號": code,
        "名稱": name or FALLBACK_NAME.get(code, ""),
        "現價": round(float(last["Close"]), 2),
        "爆發指數": score,
        "AI信心": f"{min(99, max(60, score-1))}%",
        "預估發動時間": launch,
        "RSI": round(float(rsi), 1) if not pd.isna(rsi) else np.nan,
        "K": round(float(last.get("K", np.nan)), 1) if not pd.isna(last.get("K", np.nan)) else np.nan,
        "D": round(float(last.get("D", np.nan)), 1) if not pd.isna(last.get("D", np.nan)) else np.nan,
        "MACD": "多" if last.get("DIF", 0) > last.get("MACD", 0) else "弱",
        "距離突破%": round(float(dist), 2) if not pd.isna(dist) else np.nan,
        "支撐": round(float(support), 2) if not pd.isna(support) else np.nan,
        "壓力": round(float(resistance), 2) if not pd.isna(resistance) else np.nan,
        "停損參考": round(float(support * 0.98), 2) if not pd.isna(support) else np.nan,
        "目標1": round(float(target1), 2),
        "目標2": round(float(target2), 2),
        "AI解讀": "、".join(reasons[:4]) if reasons else "資料正常，等待表態",
        "symbol": symbol,
        "狀態": "OK",
        "hist": d,
    }


def quick_score_fullmarket(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 官方全池快速分數：不靠逐檔歷史K，才能掃全股不超時
    change_abs = d["漲跌"].abs().fillna(0)
    vol_rank = d["成交量"].fillna(0).rank(pct=True)
    price_ok = d["現價"].between(8, 300).astype(int)
    intraday_strength = ((d["現價"] - d["最低"]) / (d["最高"] - d["最低"]).replace(0, np.nan)).fillna(0.5)
    d["爆發指數"] = (55 + vol_rank * 25 + intraday_strength * 15 + price_ok * 5).round().clip(0, 100).astype(int)
    d["AI信心"] = d["爆發指數"].apply(lambda x: f"{min(99, max(55, x-1))}%")
    d["預估發動時間"] = pd.cut(
        d["爆發指數"], bins=[-1, 74, 84, 91, 100], labels=["觀察中", "5～10天", "2～5天", "1～3天"]
    ).astype(str)
    d["AI解讀"] = np.where(d["爆發指數"] >= 90, "量價強勢，優先觀察", np.where(d["爆發指數"] >= 80, "型態接近表態", "觀察中"))
    return d.sort_values(["爆發指數", "成交量"], ascending=[False, False]).reset_index(drop=True)

# ---------------------------
# UI
# ---------------------------
st.title("🌍 未來小股神 AI 操盤中心 V32.2 Full Market")
st.caption("重點：全池抓上市＋上櫃，不再只有示範股票池。單股抓不到也不會整個爆掉。")

tab1, tab2, tab3 = st.tabs(["🌍 全池 AI 掃描", "🔍 單股掃描", "❤️ 7828 信仰股"])

with tab1:
    st.header("🌍 全池 AI 掃描")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        topn = st.slider("顯示前幾名", 20, 500, 100, step=20)
    with colB:
        min_score = st.slider("最低爆發指數", 0, 100, 70)
    with colC:
        refresh = st.button("🔄 重新抓取全市場")
    if refresh:
        st.cache_data.clear()
        st.rerun()

    with st.spinner("正在抓 TWSE / TPEX 官方全市場資料..."):
        full = get_full_market()
        ranked = quick_score_fullmarket(full)
    source_note = "官方全市場" if "示範池" not in set(ranked["市場"].astype(str)) else "⚠️ 官方資料暫時失敗，目前是備援示範池"
    st.success(f"取得 {len(ranked)} 檔資料｜來源：{source_note}")

    show = ranked[ranked["爆發指數"] >= min_score].head(topn)
    st.subheader("🔥 今日 AI TOP")
    st.dataframe(
        show[["代號", "名稱", "市場", "現價", "爆發指數", "AI信心", "預估發動時間", "成交量", "AI解讀"]],
        use_container_width=True,
        hide_index=True,
        height=600,
    )
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下載目前排行榜 CSV", csv, "ai_top_full_market.csv", "text/csv")

with tab2:
    st.header("🔍 單股 AI 掃描")
    code = st.text_input("輸入股票代號", "7828")
    if st.button("開始單股分析"):
        market_df = get_full_market()
        name_map = dict(zip(market_df["代號"].astype(str), market_df["名稱"].astype(str))) if not market_df.empty else {}
        info = analyze_stock(code, name_map.get(code, FALLBACK_NAME.get(code, "")))
        if info.get("狀態") != "OK":
            st.warning(f"{code} 暫時抓不到足夠歷史K線。可換 .TW/.TWO 或稍後再試。")
        else:
            st.subheader(f"{info['代號']} {info['名稱']}｜{info['現價']}｜{info['symbol']}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("爆發指數", info["爆發指數"])
            c2.metric("AI信心", info["AI信心"])
            c3.metric("預估發動時間", info["預估發動時間"])
            c4.metric("距離突破%", info["距離突破%"])
            st.info(info["AI解讀"])
            st.dataframe(pd.DataFrame([{k:v for k,v in info.items() if k not in ["hist"]}]), use_container_width=True)
            d = info["hist"].tail(90)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="K"))
            for ma in ["MA5", "MA10", "MA20"]:
                fig.add_trace(go.Scatter(x=d["Date"], y=d[ma], mode="lines", name=ma))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("❤️ 7828 信仰股")
    info = analyze_stock("7828", "創新服務")
    if info.get("狀態") != "OK":
        st.warning("7828 暫時抓不到足夠歷史K線；但 App 不會爆掉。若 7828 是興櫃或資料源暫無，請用單股掃描改試其他代號。")
    else:
        st.metric("現價", info["現價"])
        st.metric("爆發指數", info["爆發指數"])
        st.metric("預估發動時間", info["預估發動時間"])
        st.info(info["AI解讀"])
