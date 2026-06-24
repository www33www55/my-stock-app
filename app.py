import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="未來小股神 10.0 全股主力掃描器",
    layout="wide"
)

st.title("🔥 未來小股神 10.0｜全股主力法人 AI 掃描器")
st.caption("一鍵掃描上市櫃普通股，自動票選：主力方向、三大法人、技術面、量價、N字、平台突破。")

st.warning("這是選股輔助，不是買賣保證。全股掃描建議輸入 FinMind Token，資料會比較穩。")

# =========================
# 側邊設定
# =========================

FINMIND_TOKEN = st.sidebar.text_input("FinMind Token", type="password")

TOP_N = st.sidebar.slider("顯示前幾名", 10, 100, 30)
MAX_STOCKS = st.sidebar.slider("最多掃描幾檔", 50, 1800, 500)
MIN_PRICE = st.sidebar.number_input("最低股價", min_value=0.0, value=10.0)
MIN_VOLUME = st.sidebar.number_input("最低成交量", min_value=0, value=1000)
MIN_SCORE = st.sidebar.slider("最低總分", 0, 100, 60)

SCAN_DELAY = st.sidebar.slider("掃描延遲秒數，避免API過快", 0.0, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("建議第一次測試：")
st.sidebar.write("最多掃描 300～500 檔")
st.sidebar.write("跑順後再拉到 1800 檔")

# =========================
# FinMind API
# =========================

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
        data = js.get("data", [])
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# =========================
# 股票清單
# =========================

@st.cache_data(ttl=3600)
def get_stock_list():
    df = finmind_get("TaiwanStockInfo")

    if df.empty:
        return pd.DataFrame()

    df["stock_id"] = df["stock_id"].astype(str)

    df = df[df["stock_id"].str.match(r"^\d{4}$")]

    if "stock_name" in df.columns:
        df = df[~df["stock_name"].astype(str).str.contains(
            "ETF|ETN|權證|指數|期貨|牛|熊|特別股|受益證券",
            regex=True
        )]

    if "type" in df.columns:
        df = df[df["type"].isin(["twse", "tpex"])]

    keep_cols = []
    for c in ["stock_id", "stock_name", "type", "industry_category"]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].drop_duplicates("stock_id")
    df = df.sort_values("stock_id")

    return df

# =========================
# 股價資料
# =========================

@st.cache_data(ttl=3600)
def get_price(stock_id):
    symbols = [f"{stock_id}.TW", f"{stock_id}.TWO"]

    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                period="8mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.dropna()

                if len(df) >= 80:
                    return df
        except Exception:
            pass

    return pd.DataFrame()

# =========================
# 技術指標
# =========================

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_kd(df):
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    k = 100 * (df["Close"] - low9) / (high9 - low9).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)

# =========================
# 型態偵測
# =========================

def detect_platform_breakout(close, volume):
    if len(close) < 35:
        return False

    box_high = close.iloc[-31:-1].max()
    box_low = close.iloc[-31:-1].min()

    if box_low <= 0:
        return False

    box_range = (box_high - box_low) / box_low
    vol_ok = volume.iloc[-1] > volume.iloc[-6:-1].mean() * 1.3
    price_ok = close.iloc[-1] > box_high

    return box_range <= 0.18 and vol_ok and price_ok

def detect_n_pattern(close):
    if len(close) < 45:
        return False

    recent = close.iloc[-45:]

    low1 = recent.iloc[:15].min()
    high1 = recent.iloc[10:30].max()
    low2 = recent.iloc[25:40].min()
    now = recent.iloc[-1]

    cond1 = high1 > low1 * 1.08
    cond2 = low2 > low1 * 1.03
    cond3 = now >= high1 * 0.97

    return cond1 and cond2 and cond3

def detect_round_bottom(close):
    if len(close) < 70:
        return False

    a = close.iloc[-70:-45].mean()
    b = close.iloc[-45:-20].mean()
    c = close.iloc[-20:].mean()

    return b < a and c > b and close.iloc[-1] > close.iloc[-20:].mean()

def detect_pullback_hold(close):
    if len(close) < 30:
        return False

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()

    recent_low = close.iloc[-5:].min()
    now = close.iloc[-1]

    return recent_low >= ma20.iloc[-1] * 0.98 and now > ma5.iloc[-1]

# =========================
# 技術評分
# =========================

def calc_tech_score(df):
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
    avg_vol5 = volume.iloc[-6:-1].mean()
    avg_vol20 = volume.iloc[-21:-1].mean()

    score = 0
    reasons = []

    if price > ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
        score += 12
        reasons.append("短均多頭")

    if price > ma20.iloc[-1] > ma60.iloc[-1]:
        score += 10
        reasons.append("站上月季線")

    if ma20.iloc[-1] > ma20.iloc[-5]:
        score += 8
        reasons.append("月線上彎")

    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
        score += 14
        reasons.append("MACD主升")

    if hist.iloc[-1] > hist.iloc[-2]:
        score += 6
        reasons.append("MACD動能增強")

    if 45 <= rsi.iloc[-1] <= 75:
        score += 8
        reasons.append("RSI健康")

    if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 85:
        score += 8
        reasons.append("KD偏多")

    if vol > avg_vol5 * 1.4:
        score += 10
        reasons.append("量能放大")

    if vol > avg_vol20 * 1.2:
        score += 6
        reasons.append("量高於20日均量")

    if price >= close.rolling(60).max().iloc[-1] * 0.97:
        score += 10
        reasons.append("接近60日高")

    if detect_platform_breakout(close, volume):
        score += 14
        reasons.append("平台突破")

    if detect_n_pattern(close):
        score += 12
        reasons.append("N字第二波")

    if detect_round_bottom(close):
        score += 8
        reasons.append("圓弧底")

    if detect_pullback_hold(close):
        score += 8
        reasons.append("回踩不破")

    return {
        "收盤價": round(price, 2),
        "成交量": vol,
        "技術分": min(score, 100),
        "RSI": round(float(rsi.iloc[-1]), 1),
        "MACD": round(float(macd.iloc[-1]), 2),
        "技術原因": "、".join(reasons) if reasons else "技術普通"
    }

# =========================
# 法人籌碼
# =========================

@st.cache_data(ttl=3600)
def get_institution_score(stock_id):
    end = datetime.today().date()
    start = end - timedelta(days=30)

    df = finmind_get(
        "TaiwanStockInstitutionalInvestorsBuySell",
        stock_id=stock_id,
        start_date=str(start),
        end_date=str(end)
    )

    empty = {
        "外資": 0,
        "投信": 0,
        "自營商": 0,
        "三大法人": 0,
        "法人分": 0,
        "法人買天": 0,
        "法人原因": "法人無資料"
    }

    if df.empty:
        return empty

    if "buy" not in df.columns or "sell" not in df.columns:
        return empty

    df["net"] = df["buy"] - df["sell"]

    name_col = None
    for c in ["name", "institutional_investors", "institutional_investors_name"]:
        if c in df.columns:
            name_col = c
            break

    if name_col is None:
        return empty

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    names = latest[name_col].astype(str)

    foreign = latest[names.str.contains("Foreign|外資", case=False, regex=True)]["net"].sum()
    trust = latest[names.str.contains("Investment|投信", case=False, regex=True)]["net"].sum()
    dealer = latest[names.str.contains("Dealer|自營", case=False, regex=True)]["net"].sum()

    total = foreign + trust + dealer

    daily_total = df.groupby("date")["net"].sum().tail(5)
    buy_days = int((daily_total > 0).sum())

    score = 0
    reasons = []

    if foreign > 0:
        score += 16
        reasons.append("外資買超")

    if trust > 0:
        score += 22
        reasons.append("投信買超")

    if dealer > 0:
        score += 8
        reasons.append("自營買超")

    if total > 3000:
        score += 24
        reasons.append("法人強買")

    elif total > 1000:
        score += 18
        reasons.append("法人明顯買")

    elif total > 0:
        score += 10
        reasons.append("法人偏買")

    if buy_days >= 5:
        score += 20
        reasons.append("法人連5買")

    elif buy_days >= 4:
        score += 16
        reasons.append("法人連4買")

    elif buy_days >= 3:
        score += 10
        reasons.append("法人連3買")

    return {
        "外資": int(foreign),
        "投信": int(trust),
        "自營商": int(dealer),
        "三大法人": int(total),
        "法人分": min(score, 100),
        "法人買天": buy_days,
        "法人原因": "、".join(reasons) if reasons else "法人普通"
    }

# =========================
# 主力推估
# =========================

def calc_main_force_score(tech, inst):
    score = 0
    reasons = []

    if inst["三大法人"] > 0 and tech["成交量"] > MIN_VOLUME:
        score += 15
        reasons.append("資金流入")

    if inst["投信"] > 0:
        score += 20
        reasons.append("投信加持")

    if inst["外資"] > 1000:
        score += 15
        reasons.append("外資明顯買")

    if "量能放大" in tech["技術原因"]:
        score += 15
        reasons.append("量能啟動")

    if "平台突破" in tech["技術原因"]:
        score += 15
        reasons.append("突破平台")

    if "N字第二波" in tech["技術原因"]:
        score += 15
        reasons.append("第二波型態")

    if "回踩不破" in tech["技術原因"]:
        score += 10
        reasons.append("回踩守住")

    return {
        "主力分": min(score, 100),
        "主力原因": "、".join(reasons) if reasons else "主力普通"
    }

# =========================
# 評價
# =========================

def final_grade(score):
    if score >= 88:
        return "🔥 五星強勢"
    elif score >= 78:
        return "⭐ 優先觀察"
    elif score >= 68:
        return "👀 可追蹤"
    elif score >= 60:
        return "普通偏多"
    else:
        return "普通"

def ai_comment(row):
    reasons = []

    if row["主力分"] >= 70:
        reasons.append("主力動能強")
    if row["法人分"] >= 60:
        reasons.append("法人籌碼偏多")
    if row["技術分"] >= 70:
        reasons.append("技術面強")
    if row["投信"] > 0:
        reasons.append("投信有買")
    if row["三大法人"] > 1000:
        reasons.append("法人買超明顯")
    if "N字第二波" in row["技術原因"]:
        reasons.append("N字第二波")
    if "平台突破" in row["技術原因"]:
        reasons.append("平台突破")
    if "回踩不破" in row["技術原因"]:
        reasons.append("回踩不破")

    if not reasons:
        return "條件普通，先觀察"

    return "、".join(reasons)

# =========================
# 掃描主程式
# =========================

def scan_all():
    stock_df = get_stock_list()

    if stock_df.empty:
        st.error("抓不到股票清單，請檢查 FinMind Token 或網路。")
        return pd.DataFrame()

    stock_df = stock_df.head(MAX_STOCKS).reset_index(drop=True)

    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(stock_df)

    for i, row in stock_df.iterrows():
        stock_id = str(row["stock_id"])
        name = row.get("stock_name", "")

        status.write(f"掃描中：{i+1}/{total}｜{stock_id} {name}")

        df_price = get_price(stock_id)

        tech = calc_tech_score(df_price)

        if tech is None:
            progress.progress((i + 1) / total)
            continue

        if tech["收盤價"] < MIN_PRICE:
            progress.progress((i + 1) / total)
            continue

        if tech["成交量"] < MIN_VOLUME:
            progress.progress((i + 1) / total)
            continue

        inst = get_institution_score(stock_id)
        main_force = calc_main_force_score(tech, inst)

        total_score = round(
            tech["技術分"] * 0.45 +
            inst["法人分"] * 0.35 +
            main_force["主力分"] * 0.20,
            1
        )

        if total_score < MIN_SCORE:
            progress.progress((i + 1) / total)
            continue

        item = {
            "代號": stock_id,
            "名稱": name,
            "總分": total_score,
            "評價": final_grade(total_score),
            "主力分": main_force["主力分"],
            "法人分": inst["法人分"],
            "技術分": tech["技術分"],
            "收盤價": tech["收盤價"],
            "成交量": tech["成交量"],
            "外資": inst["外資"],
            "投信": inst["投信"],
            "自營商": inst["自營商"],
            "三大法人": inst["三大法人"],
            "法人買天": inst["法人買天"],
            "RSI": tech["RSI"],
            "MACD": tech["MACD"],
            "AI判斷": "",
            "主力原因": main_force["主力原因"],
            "法人原因": inst["法人原因"],
            "技術原因": tech["技術原因"]
        }

        item["AI判斷"] = ai_comment(item)

        results.append(item)

        progress.progress((i + 1) / total)

        if SCAN_DELAY > 0:
            time.sleep(SCAN_DELAY)

    status.write("掃描完成")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("總分", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", df.index + 1)

    return df

# =========================
# UI
# =========================

st.subheader("🏆 今日 AI 全股票選")

if st.button("🚀 開始全股掃描", use_container_width=True):
    result = scan_all()

    if result.empty:
        st.error("沒有掃到符合條件的股票。可以降低最低成交量、最低分數，或增加掃描檔數。")
    else:
        st.success("掃描完成")

        show_cols = [
            "排名", "代號", "名稱", "總分", "評價",
            "主力分", "法人分", "技術分",
            "收盤價", "三大法人", "外資", "投信", "自營商",
            "法人買天", "RSI", "AI判斷"
        ]

        st.subheader(f"🔥 今日票選前 {TOP_N} 名")
        st.dataframe(
            result.head(TOP_N)[show_cols],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🔥 五星強勢股")
        strong = result[result["總分"] >= 88]
        if strong.empty:
            st.info("今天沒有五星強勢股。")
        else:
            st.dataframe(
                strong[show_cols],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("⭐ N字第二波")
        n_df = result[result["技術原因"].str.contains("N字第二波", na=False)]
        if n_df.empty:
            st.info("今天沒有明顯 N 字第二波。")
        else:
            st.dataframe(
                n_df[show_cols],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("🚀 平台突破")
        breakout_df = result[result["技術原因"].str.contains("平台突破", na=False)]
        if breakout_df.empty:
            st.info("今天沒有明顯平台突破。")
        else:
            st.dataframe(
                breakout_df[show_cols],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("🐳 主力偷吃貨")
        stealth_df = result[
            (result["主力分"] >= 50) &
            (result["總分"] < 88) &
            (result["三大法人"] > 0)
        ]

        if stealth_df.empty:
            st.info("今天沒有明顯主力偷吃貨。")
        else:
            st.dataframe(
                stealth_df[show_cols],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("📋 完整細節")
        detail_cols = [
            "排名", "代號", "名稱", "總分", "評價",
            "主力分", "法人分", "技術分",
            "收盤價", "成交量",
            "外資", "投信", "自營商", "三大法人",
            "法人買天", "RSI", "MACD",
            "AI判斷", "主力原因", "法人原因", "技術原因"
        ]

        st.dataframe(
            result[detail_cols],
            use_container_width=True,
            hide_index=True
        )

        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載完整掃描結果 CSV",
            data=csv,
            file_name="future_stock_god_scan_10.csv",
            mime="text/csv"
        )
