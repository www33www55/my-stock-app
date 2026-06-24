import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="全股主力法人選股器 Ultimate", layout="wide")

st.title("🔥 全股主力法人選股器 Ultimate")
st.caption("一鍵掃描上市櫃普通股：三大法人 + 技術面 + 量價 + 型態評分")

st.warning("這是選股輔助工具，不是買賣保證。全股掃描 API 量大，強烈建議輸入 FinMind Token。")

FINMIND_TOKEN = st.sidebar.text_input("FinMind Token", type="password")
TOP_N = st.sidebar.slider("顯示前幾名", 10, 100, 30)
MAX_STOCKS = st.sidebar.slider("最多掃描幾檔", 50, 1800, 500)
MIN_PRICE = st.sidebar.number_input("最低股價", 0.0, 500.0, 10.0)
MIN_VOLUME = st.sidebar.number_input("最低成交量", 0, 200000, 1000)

@st.cache_data(ttl=3600)
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

    try:
        r = requests.get(url, params=params, timeout=20)
        js = r.json()
        return pd.DataFrame(js.get("data", []))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_list():
    df = finmind_get("TaiwanStockInfo")

    if df.empty:
        return pd.DataFrame()

    keep_cols = [c for c in ["stock_id", "stock_name", "type", "industry_category"] if c in df.columns]
    df = df[keep_cols].copy()

    df["stock_id"] = df["stock_id"].astype(str)

    # 排除 ETF、權證、特殊代號
    df = df[df["stock_id"].str.match(r"^\d{4}$")]
    df = df[~df["stock_name"].astype(str).str.contains("ETF|ETN|指數|期貨|權證|牛|熊", regex=True)]

    # 上市上櫃
    if "type" in df.columns:
        df = df[df["type"].isin(["twse", "tpex"])]

    return df.drop_duplicates("stock_id")

@st.cache_data(ttl=3600)
def get_price(stock_id):
    for suffix in [".TW", ".TWO"]:
        try:
            df = yf.download(
                f"{stock_id}{suffix}",
                period="8mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                return df
        except Exception:
            pass

    return pd.DataFrame()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_kd(df):
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    k = 100 * (df["Close"] - low9) / (high9 - low9)
    d = k.rolling(3).mean()
    return k, d

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def detect_n_pattern(close):
    if len(close) < 40:
        return False

    recent = close.iloc[-40:]
    low1 = recent.iloc[:15].min()
    high1 = recent.iloc[:25].max()
    low2 = recent.iloc[15:35].min()
    now = recent.iloc[-1]

    return high1 > low1 * 1.08 and low2 > low1 * 1.03 and now >= high1 * 0.98

def detect_platform_breakout(close, volume):
    if len(close) < 30:
        return False

    box_high = close.iloc[-25:-1].max()
    box_low = close.iloc[-25:-1].min()
    box_range = (box_high - box_low) / box_low
    vol_ok = volume.iloc[-1] > volume.iloc[-6:-1].mean() * 1.3

    return box_range < 0.15 and close.iloc[-1] > box_high and vol_ok

def detect_round_bottom(close):
    if len(close) < 60:
        return False

    a = close.iloc[-60:-40].mean()
    b = close.iloc[-40:-20].mean()
    c = close.iloc[-20:].mean()

    return b < a and c > b and close.iloc[-1] > close.iloc[-20:].mean()

def calc_tech(df):
    if df.empty or len(df) < 80:
        return None

    close = df["Close"]
    volume = df["Volume"]

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    macd, signal, hist = calc_macd(close)
    rsi = calc_rsi(close)
    k, d = calc_kd(df)

    price = float(close.iloc[-1])
    vol = int(volume.iloc[-1])
    avg_vol5 = volume.rolling(5).mean().iloc[-1]

    score = 0
    reasons = []

    if price > ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
        score += 12
        reasons.append("短均多頭")

    if price > ma60.iloc[-1]:
        score += 8
        reasons.append("站上季線")

    if ma20.iloc[-1] > ma20.iloc[-5]:
        score += 8
        reasons.append("月線上彎")

    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
        score += 15
        reasons.append("MACD主升")

    if hist.iloc[-1] > hist.iloc[-2]:
        score += 5
        reasons.append("MACD柱增強")

    if 45 <= rsi.iloc[-1] <= 75:
        score += 8
        reasons.append("RSI健康")

    if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 85:
        score += 8
        reasons.append("KD偏多未過熱")

    if vol > avg_vol5 * 1.4:
        score += 12
        reasons.append("量能放大")

    if price >= close.rolling(60).max().iloc[-1] * 0.97:
        score += 10
        reasons.append("接近60日高")

    if detect_platform_breakout(close, volume):
        score += 15
        reasons.append("平台突破")

    if detect_n_pattern(close):
        score += 12
        reasons.append("N字型態")

    if detect_round_bottom(close):
        score += 8
        reasons.append("圓弧底")

    return {
        "收盤價": round(price, 2),
        "成交量": vol,
        "技術分": min(score, 100),
        "RSI": round(float(rsi.iloc[-1]), 1),
        "MACD": round(float(macd.iloc[-1]), 2),
        "技術原因": "、".join(reasons)
    }

@st.cache_data(ttl=3600)
def get_institution(stock_id, days=14):
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
            "外資": 0, "投信": 0, "自營商": 0,
            "三大法人": 0, "法人分": 0,
            "法人原因": "法人無資料"
        }

    if "buy" not in df.columns or "sell" not in df.columns:
        return {
            "外資": 0, "投信": 0, "自營商": 0,
            "三大法人": 0, "法人分": 0,
            "法人原因": "法人欄位異常"
        }

    df["net"] = df["buy"] - df["sell"]
    name_col = "name" if "name" in df.columns else "institutional_investors"

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    foreign = latest[latest[name_col].astype(str).str.contains("Foreign|外資", case=False, regex=True)]["net"].sum()
    trust = latest[latest[name_col].astype(str).str.contains("Investment|投信", case=False, regex=True)]["net"].sum()
    dealer = latest[latest[name_col].astype(str).str.contains("Dealer|自營", case=False, regex=True)]["net"].sum()

    total = foreign + trust + dealer

    daily_total = df.groupby("date")["net"].sum().tail(5)
    buy_days = int((daily_total > 0).sum())

    score = 0
    reasons = []

    if foreign > 0:
        score += 18
        reasons.append("外資買")
    if trust > 0:
        score += 22
        reasons.append("投信買")
    if dealer > 0:
        score += 8
        reasons.append("自營買")
    if total > 1000:
        score += 22
        reasons.append("法人強買")
    elif total > 0:
        score += 10
        reasons.append("法人偏買")
    if buy_days >= 4:
        score += 18
        reasons.append("法人連買")
    elif buy_days >= 3:
        score += 10
        reasons.append("法人偏連買")

    return {
        "外資": int(foreign),
        "投信": int(trust),
        "自營商": int(dealer),
        "三大法人": int(total),
        "法人分": min(score, 100),
        "法人原因": "、".join(reasons) if reasons else "法人普通"
    }

def final_grade(score):
    if score >= 85:
        return "🔥 強勢票選"
    elif score >= 75:
        return "⭐ 優先觀察"
    elif score >= 65:
        return "👀 可追蹤"
    else:
        return "普通"

def draw_chart(stock_id, name):
    df = get_price(stock_id)
    if df.empty:
        st.error("抓不到K線")
        return

    ma5 = df["Close"].rolling(5).mean()
    ma20 = df["Close"].rolling(20).mean()
    ma60 = df["Close"].rolling(60).mean()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="K線"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=ma5, name="MA5"))
    fig.add_trace(go.Scatter(x=df.index, y=ma20, name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=ma60, name="MA60"))

    fig.update_layout(
        title=f"{stock_id} {name} K線圖",
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

def scan_all():
    stock_df = get_stock_list()

    if stock_df.empty:
        st.error("抓不到股票清單，請確認網路或 FinMind Token。")
        return pd.DataFrame()

    stock_df = stock_df.head(MAX_STOCKS)
    results = []

    progress = st.progress(0)
    status = st.empty()

    total_count = len(stock_df)

    for idx, row in stock_df.iterrows():
        stock_id = row["stock_id"]
        name = row.get("stock_name", "")

        status.write(f"掃描中：{stock_id} {name}")

        df_price = get_price(stock_id)
        tech = calc_tech(df_price)

        if tech is None:
            progress.progress(min((len(results) + 1) / total_count, 1.0))
            continue

        if tech["收盤價"] < MIN_PRICE:
            continue

        if tech["成交量"] < MIN_VOLUME:
            continue

        inst = get_institution(stock_id)

        total_score = round(tech["技術分"] * 0.6 + inst["法人分"] * 0.4, 1)

        results.append({
            "股票": stock_id,
            "名稱": name,
            "收盤價": tech["收盤價"],
            "成交量": tech["成交量"],
            "外資": inst["外資"],
            "投信": inst["投信"],
            "自營商": inst["自營商"],
            "三大法人": inst["三大法人"],
            "技術分": tech["技術分"],
            "法人分": inst["法人分"],
            "總分": total_score,
            "評價": final_grade(total_score),
            "RSI": tech["RSI"],
            "MACD": tech["MACD"],
            "技術原因": tech["技術原因"],
            "法人原因": inst["法人原因"],
        })

        progress.progress(min((idx + 1) / total_count, 1.0))

    status.write("掃描完成")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values("總分", ascending=False)

tab1, tab2 = st.tabs(["🔥 全股掃描票選", "📈 單股K線檢查"])

with tab1:
    st.subheader("一鍵全股掃描")

    if st.button("🚀 開始掃描全上市櫃", use_container_width=True):
        result = scan_all()

        if result.empty:
            st.error("沒有掃到符合條件的股票。")
        else:
            st.success("掃描完成")

            st.subheader(f"🏆 今日票選前 {TOP_N} 名")
            st.dataframe(result.head(TOP_N), use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載完整掃描結果 CSV",
                data=csv,
                file_name="stock_scan_result.csv",
                mime="text/csv"
            )

            st.subheader("🔥 強勢票選")
            st.dataframe(result[result["總分"] >= 85], use_container_width=True)

            st.subheader("⭐ 優先觀察")
            st.dataframe(result[(result["總分"] >= 75) & (result["總分"] < 85)], use_container_width=True)

with tab2:
    stock_id = st.text_input("輸入股票代號看K線", value="2409")
    stock_name = st.text_input("股票名稱，可不填", value="友達")

    if st.button("畫K線"):
        draw_chart(stock_id, stock_name)
