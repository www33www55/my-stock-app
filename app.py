import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="主力法人選股器 6.0", layout="wide")
st.title("🔥 主力法人選股器 6.0｜三大法人＋技術面＋主力動向")

st.warning("提醒：這是選股輔助工具，不是保證獲利。法人買超 ≠ 一定上漲，要搭配型態、量、均線。")

FINMIND_TOKEN = st.sidebar.text_input("FinMind Token（可不填，但有填比較穩）", type="password")

stock_input = st.text_area(
    "輸入股票代號，用逗號分開",
    value="2409,2303,6271,1714,3037,2313,2382,2330,3060"
)

days = st.sidebar.slider("抓幾天法人資料", 5, 30, 10)

def finmind_get(dataset, stock_id=None, start_date=None, end_date=None):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {"dataset": dataset}
    if stock_id:
        params["data_id"] = stock_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "data" not in data:
        return pd.DataFrame()
    return pd.DataFrame(data["data"])

def get_price(stock_id):
    symbol = f"{stock_id}.TW"
    df = yf.download(symbol, period="6mo", progress=False, auto_adjust=False)

    if df.empty:
        symbol = f"{stock_id}.TWO"
        df = yf.download(symbol, period="6mo", progress=False, auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

def calc_tech(df):
    if df.empty or len(df) < 60:
        return None

    close = df["Close"]
    volume = df["Volume"]

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    k = 100 * (close - low9) / (high9 - low9)
    d = k.rolling(3).mean()

    latest_close = close.iloc[-1]
    latest_volume = volume.iloc[-1]
    avg_volume = volume.rolling(5).mean().iloc[-1]

    score = 0
    signals = []

    if latest_close > ma5.iloc[-1] > ma20.iloc[-1]:
        score += 15
        signals.append("多頭排列")

    if latest_close > ma60.iloc[-1]:
        score += 10
        signals.append("站上季線")

    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
        score += 20
        signals.append("MACD主升")

    if rsi.iloc[-1] < 75:
        score += 10
        signals.append("RSI未過熱")

    if latest_volume > avg_volume * 1.5:
        score += 15
        signals.append("量放大")

    if latest_close >= close.rolling(60).max().iloc[-1] * 0.97:
        score += 15
        signals.append("接近60日高")

    if k.iloc[-1] > d.iloc[-1]:
        score += 10
        signals.append("KD偏多")

    return {
        "收盤價": round(latest_close, 2),
        "技術分數": score,
        "技術訊號": "、".join(signals),
        "RSI": round(rsi.iloc[-1], 1),
        "MACD": round(macd.iloc[-1], 2),
        "成交量": int(latest_volume)
    }

def get_institution(stock_id):
    end = datetime.today().date()
    start = end - timedelta(days=days * 2)

    df = finmind_get(
        "TaiwanStockInstitutionalInvestorsBuySell",
        stock_id=stock_id,
        start_date=str(start),
        end_date=str(end)
    )

    if df.empty:
        return {
            "外資": 0,
            "投信": 0,
            "自營商": 0,
            "三大法人": 0,
            "法人分數": 0,
            "法人訊號": "無資料"
        }

    if "buy" not in df.columns or "sell" not in df.columns:
        return {
            "外資": 0,
            "投信": 0,
            "自營商": 0,
            "三大法人": 0,
            "法人分數": 0,
            "法人訊號": "欄位異常"
        }

    df["net"] = df["buy"] - df["sell"]

    name_col = "name" if "name" in df.columns else "institutional_investors"

    latest_date = df["date"].max()
    today_df = df[df["date"] == latest_date]

    foreign = today_df[today_df[name_col].astype(str).str.contains("Foreign|外資", case=False, regex=True)]["net"].sum()
    trust = today_df[today_df[name_col].astype(str).str.contains("Investment|投信", case=False, regex=True)]["net"].sum()
    dealer = today_df[today_df[name_col].astype(str).str.contains("Dealer|自營", case=False, regex=True)]["net"].sum()

    total = foreign + trust + dealer

    score = 0
    signals = []

    if foreign > 0:
        score += 20
        signals.append("外資買超")
    if trust > 0:
        score += 25
        signals.append("投信買超")
    if dealer > 0:
        score += 10
        signals.append("自營商買超")
    if total > 1000:
        score += 25
        signals.append("三大法人強買")
    elif total > 0:
        score += 10
        signals.append("法人偏買")

    return {
        "外資": int(foreign),
        "投信": int(trust),
        "自營商": int(dealer),
        "三大法人": int(total),
        "法人分數": score,
        "法人訊號": "、".join(signals) if signals else "法人普通"
    }

def grade(score):
    if score >= 90:
        return "🔥 強勢觀察"
    elif score >= 75:
        return "⭐ 偏多"
    elif score >= 60:
        return "👀 可觀察"
    else:
        return "普通"

if st.button("開始掃描"):
    stock_list = [s.strip() for s in stock_input.split(",") if s.strip()]
    results = []

    progress = st.progress(0)

    for i, stock_id in enumerate(stock_list):
        progress.progress((i + 1) / len(stock_list))

        price_df = get_price(stock_id)
        tech = calc_tech(price_df)

        if tech is None:
            continue

        inst = get_institution(stock_id)

        total_score = tech["技術分數"] + inst["法人分數"]

        results.append({
            "股票": stock_id,
            "收盤價": tech["收盤價"],
            "外資": inst["外資"],
            "投信": inst["投信"],
            "自營商": inst["自營商"],
            "三大法人": inst["三大法人"],
            "技術分數": tech["技術分數"],
            "法人分數": inst["法人分數"],
            "總分": total_score,
            "評價": grade(total_score),
            "技術訊號": tech["技術訊號"],
            "法人訊號": inst["法人訊號"],
            "RSI": tech["RSI"],
            "MACD": tech["MACD"],
            "成交量": tech["成交量"]
        })

    if results:
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values("總分", ascending=False)

        st.subheader("🔥 掃描結果")
        st.dataframe(df_result, use_container_width=True)

        st.subheader("⭐ 今日優先觀察")
        st.dataframe(df_result[df_result["總分"] >= 75], use_container_width=True)
    else:
        st.error("沒有抓到資料，可能是代號錯誤、資料源限制，或今天法人資料尚未更新。")
