import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import twstock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config"未來小股神 10.3 
st.title("🔥｜全池 AI 選股器")
st.caption("全上市櫃掃描｜技術分＋法人分＋主力分｜自動票選排行榜｜無 K 線頁")
st.warning("這是選股輔助，不是買賣保證。沒填 FinMind Token 也可跑技術面；有 Token 才會加法人分。")

TOKEN = st.sidebar.text_input("FinMind Token，可不填", type="password")
TOP_N = st.sidebar.slider("顯示前幾名", 10, 100, 30)
MAX_STOCKS = st.sidebar.slider("最多掃描幾檔", 100, 2000, 1800)
MAX_WORKERS = st.sidebar.slider("掃描速度", 2, 12, 6)
MIN_PRICE = st.sidebar.number_input("最低股價", value=10.0)
MIN_VOL = st.sidebar.number_input("最低成交量", value=1000)
MIN_SCORE = st.sidebar.slider("最低分數", 0, 100, 50)

@st.cache_data(ttl=86400)
def get_stock_list():
    rows = []
    for sid, info in twstock.codes.items():
        sid = str(sid)
        if not sid.isdigit() or len(sid) != 4:
            continue
        name = getattr(info, "name", "")
        market = getattr(info, "market", "")
        group = getattr(info, "group", "")
        if market not in ["上市", "上櫃"]:
            continue
        bad = ["ETF", "ETN", "權證", "牛", "熊", "指數", "特", "受益"]
        if any(x in name for x in bad):
            continue
        rows.append({"代號": sid, "名稱": name, "市場": market, "產業": group})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("代號").head(MAX_STOCKS).reset_index(drop=True)

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
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) >= 80:
                    return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_inst(stock_id, token):
    if not token:
        return {"外資": 0, "投信": 0, "自營商": 0, "三大法人": 0, "法人分": 0, "法人原因": "未填Token"}
    end = datetime.today().date()
    start = end - timedelta(days=30)
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": str(start),
        "end_date": str(end),
        "token": token,
    }
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
        df = pd.DataFrame(r.json().get("data", []))
    except Exception:
        df = pd.DataFrame()
    if df.empty or "buy" not in df.columns or "sell" not in df.columns:
        return {"外資": 0, "投信": 0, "自營商": 0, "三大法人": 0, "法人分": 0, "法人原因": "法人無資料"}
    df["net"] = df["buy"] - df["sell"]
    name_col = None
    for c in ["name", "institutional_investors", "institutional_investors_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return {"外資": 0, "投信": 0, "自營商": 0, "三大法人": 0, "法人分": 0, "法人原因": "法人欄位異常"}
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    names = latest[name_col].astype(str)
    foreign = latest[names.str.contains("Foreign|外資", case=False, regex=True)]["net"].sum()
    trust = latest[names.str.contains("Investment|投信", case=False, regex=True)]["net"].sum()
    dealer = latest[names.str.contains("Dealer|自營", case=False, regex=True)]["net"].sum()
    total = foreign + trust + dealer
    daily = df.groupby("date")["net"].sum().tail(5)
    buy_days = int((daily > 0).sum())
    score, reason = 0, []
    if foreign > 0:
        score += 15; reason.append("外資買")
    if trust > 0:
        score += 25; reason.append("投信買")
    if dealer > 0:
        score += 8; reason.append("自營買")
    if total > 3000:
        score += 25; reason.append("法人強買")
    elif total > 1000:
        score += 18; reason.append("法人明顯買")
    elif total > 0:
        score += 10; reason.append("法人偏買")
    if buy_days >= 5:
        score += 20; reason.append("法人連5買")
    elif buy_days >= 3:
        score += 10; reason.append("法人連3買")
    return {"外資": int(foreign), "投信": int(trust), "自營商": int(dealer), "三大法人": int(total), "法人分": min(score, 100), "法人原因": "、".join(reason) if reason else "法人普通"}

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    m = ema12 - ema26
    s = m.ewm(span=9, adjust=False).mean()
    h = m - s
    return m, s, h

def kd(df):
    low = df["Low"].rolling(9).min()
    high = df["High"].rolling(9).max()
    k = 100 * (df["Close"] - low) / (high - low).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)

def platform_breakout(close, vol):
    if len(close) < 35:
        return False
    high = close.iloc[-31:-1].max()
    low = close.iloc[-31:-1].min()
    if low <= 0:
        return False
    box = (high - low) / low
    return box <= 0.18 and close.iloc[-1] > high and vol.iloc[-1] > vol.iloc[-6:-1].mean() * 1.3

def n_pattern(close):
    if len(close) < 45:
        return False
    r = close.iloc[-45:]
    low1 = r.iloc[:15].min()
    high1 = r.iloc[10:30].max()
    low2 = r.iloc[25:40].min()
    now = r.iloc[-1]
    return high1 > low1 * 1.08 and low2 > low1 * 1.03 and now >= high1 * 0.97

def round_bottom(close):
    if len(close) < 70:
        return False
    a = close.iloc[-70:-45].mean()
    b = close.iloc[-45:-20].mean()
    c = close.iloc[-20:].mean()
    return b < a and c > b and close.iloc[-1] > close.iloc[-20:].mean()

def pullback_hold(close):
    if len(close) < 30:
        return False
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    return close.iloc[-5:].min() >= ma20.iloc[-1] * 0.98 and close.iloc[-1] > ma5.iloc[-1]

def tech_score(df):
    if df.empty or len(df) < 80:
        return None
    close = df["Close"]
    vol = df["Volume"]
    price = float(close.iloc[-1])
    volume = int(vol.iloc[-1])
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    rr = rsi(close)
    mm, ss, hh = macd(close)
    kk, dd = kd(df)
    score, reason = 0, []
    if price > ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
        score += 12; reason.append("短均多頭")
    if price > ma20.iloc[-1] > ma60.iloc[-1]:
        score += 10; reason.append("站上月季線")
    if ma20.iloc[-1] > ma20.iloc[-5]:
        score += 8; reason.append("月線上彎")
    if mm.iloc[-1] > ss.iloc[-1] and mm.iloc[-1] > 0:
        score += 15; reason.append("MACD主升")
    if hh.iloc[-1] > hh.iloc[-2]:
        score += 6; reason.append("MACD動能增強")
    if 45 <= rr.iloc[-1] <= 75:
        score += 8; reason.append("RSI健康")
    if kk.iloc[-1] > dd.iloc[-1] and kk.iloc[-1] < 85:
        score += 8; reason.append("KD偏多")
    if vol.iloc[-1] > vol.iloc[-6:-1].mean() * 1.4:
        score += 10; reason.append("量能放大")
    if price >= close.rolling(60).max().iloc[-1] * 0.97:
        score += 10; reason.append("接近60日高")
    if platform_breakout(close, vol):
        score += 15; reason.append("平台突破")
    if n_pattern(close):
        score += 12; reason.append("N字第二波")
    if round_bottom(close):
        score += 8; reason.append("圓弧底")
    if pullback_hold(close):
        score += 8; reason.append("回踩不破")
    return {"收盤價": round(price, 2), "成交量": volume, "技術分": min(score, 100), "RSI": round(float(rr.iloc[-1]), 1), "MACD": round(float(mm.iloc[-1]), 2), "技術原因": "、".join(reason) if reason else "技術普通"}

def main_score(tech, inst):
    score, reason = 0, []
    if inst["三大法人"] > 0:
        score += 15; reason.append("資金流入")
    if inst["投信"] > 0:
        score += 20; reason.append("投信加持")
    if inst["外資"] > 1000:
        score += 15; reason.append("外資明顯買")
    if "量能放大" in tech["技術原因"]:
        score += 15; reason.append("量能啟動")
    if "平台突破" in tech["技術原因"]:
        score += 15; reason.append("突破平台")
    if "N字第二波" in tech["技術原因"]:
        score += 15; reason.append("第二波型態")
    if "回踩不破" in tech["技術原因"]:
        score += 10; reason.append("回踩守住")
    return {"主力分": min(score, 100), "主力原因": "、".join(reason) if reason else "主力普通"}

def grade(score):
    if score >= 88:
        return "🔥 五星強勢"
    if score >= 78:
        return "⭐ 優先觀察"
    if score >= 68:
        return "👀 可追蹤"
    if score >= 60:
        return "普通偏多"
    return "普通"

def ai_text(row):
    r = []
    if row["主力分"] >= 50:
        r.append("主力動能")
    if row["法人分"] >= 50:
        r.append("法人偏多")
    if row["技術分"] >= 60:
        r.append("技術強")
    if row["投信"] > 0:
        r.append("投信買")
    if "N字第二波" in row["技術原因"]:
        r.append("N字")
    if "平台突破" in row["技術原因"]:
        r.append("突破")
    if "回踩不破" in row["技術原因"]:
        r.append("回踩不破")
    return "、".join(r) if r else "先觀察"

def scan_one(row):
    sid = row["代號"]
    name = row["名稱"]
    try:
        price_df = get_price(sid)
        tech = tech_score(price_df)
        if tech is None:
            return None
        if tech["收盤價"] < MIN_PRICE or tech["成交量"] < MIN_VOL:
            return None
        inst = get_inst(sid, TOKEN)
        main = main_score(tech, inst)
        total = round(tech["技術分"] * 0.50 + inst["法人分"] * 0.30 + main["主力分"] * 0.20, 1)
        if total < MIN_SCORE:
            return None
        item = {
            "代號": sid, "名稱": name, "總分": total, "評價": grade(total),
            "主力分": main["主力分"], "法人分": inst["法人分"], "技術分": tech["技術分"],
            "收盤價": tech["收盤價"], "成交量": tech["成交量"],
            "外資": inst["外資"], "投信": inst["投信"], "自營商": inst["自營商"], "三大法人": inst["三大法人"],
            "RSI": tech["RSI"], "MACD": tech["MACD"],
            "AI判斷": "", "主力原因": main["主力原因"], "法人原因": inst["法人原因"], "技術原因": tech["技術原因"],
        }
        item["AI判斷"] = ai_text(item)
        return item
    except Exception:
        return None

def scan_all():
    stocks = get_stock_list()
    if stocks.empty:
        st.error("抓不到股票清單。請確認 requirements.txt 有 twstock。")
        return pd.DataFrame()
    total = len(stocks)
    progress = st.progress(0)
    status = st.empty()
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scan_one, row) for _, row in stocks.iterrows()]
        for future in as_completed(futures):
            done += 1
            progress.progress(done / total)
            status.write(f"掃描中：{done}/{total}")
            res = future.result()
            if res:
                results.append(res)
    status.write("掃描完成")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("總分", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", df.index + 1)
    return df

st.subheader("🏆 今日 AI 全池票選")

with st.expander("使用提示", expanded=False):
    st.write("第一次跑建議：最低分數 0～50、最低成交量 1000、掃描速度 4～6。")
    st.write("沒填 FinMind Token 也可以跑，但法人欄位會是 0，主要看技術分與主力推估。")

if st.button("🚀 開始全池掃描", use_container_width=True):
    result = scan_all()
    if result.empty:
        st.error("沒有掃到符合條件的股票，請降低最低成交量或最低分數。")
    else:
        st.success("掃描完成")
        show_cols = ["排名", "代號", "名稱", "總分", "評價", "主力分", "法人分", "技術分", "收盤價", "三大法人", "外資", "投信", "自營商", "RSI", "AI判斷"]
        st.subheader(f"🔥 今日票選前 {TOP_N} 名")
        st.dataframe(result.head(TOP_N)[show_cols], use_container_width=True, hide_index=True)
        st.subheader("🔥 五星強勢")
        st.dataframe(result[result["總分"] >= 88][show_cols], use_container_width=True, hide_index=True)
        st.subheader("⭐ N字第二波")
        st.dataframe(result[result["技術原因"].str.contains("N字第二波", na=False)][show_cols], use_container_width=True, hide_index=True)
        st.subheader("🚀 平台突破")
        st.dataframe(result[result["技術原因"].str.contains("平台突破", na=False)][show_cols], use_container_width=True, hide_index=True)
        st.subheader("🐳 主力偷吃貨")
        st.dataframe(result[(result["主力分"] >= 40) & (result["總分"] < 88)][show_cols], use_container_width=True, hide_index=True)
        st.subheader("📋 完整細節")
        detail_cols = ["排名", "代號", "名稱", "總分", "評價", "主力分", "法人分", "技術分", "收盤價", "成交量", "外資", "投信", "自營商", "三大法人", "RSI", "MACD", "AI判斷", "主力原因", "法人原因", "技術原因"]
        st.dataframe(result[detail_cols], use_container_width=True, hide_index=True)
        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載完整結果 CSV", csv, "future_stock_god_10_3.csv", "text/csv")
