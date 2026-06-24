import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import twstock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="未來小股神 AI 選股系統", layout="wide")
st.title("未來小股神 AI 選股系統")
st.caption("全池選股｜AI Top 8｜技術面・籌碼面・主力方向｜僅供研究參考")

with st.sidebar:
    st.header("設定")
    token = st.text_input("FinMind Token（可不填）", type="password")
    top_n = st.slider("Top 幾檔", 8, 50, 8)
    max_workers = st.slider("掃描速度", 2, 12, 6)
    min_price = st.number_input("最低股價", value=10.0, min_value=0.0, step=1.0)
    min_volume = st.number_input("最低成交量", value=1000, min_value=0, step=500)
    min_score = st.slider("最低總分", 0, 100, 55)
    use_inst = st.checkbox("加入法人分數（需要 Token 較穩）", value=bool(token))

st.info("第一次全池掃描會比較久；之後資料會快取。法人抓不到時，系統會自動用技術面先排名，不會整個壞掉。")

@st.cache_data(ttl=86400)
def get_stock_list():
    rows = []
    bad_words = ["ETF", "ETN", "權證", "牛", "熊", "指數", "特", "受益"]
    for sid, info in twstock.codes.items():
        sid = str(sid)
        if not sid.isdigit() or len(sid) != 4:
            continue
        name = getattr(info, "name", "")
        market = getattr(info, "market", "")
        group = getattr(info, "group", "")
        if market not in ["上市", "上櫃"]:
            continue
        if any(w in str(name) for w in bad_words):
            continue
        rows.append({"代號": sid, "名稱": name, "市場": market, "產業": group})
    return pd.DataFrame(rows).sort_values("代號").reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_price(stock_id):
    for suffix in [".TW", ".TWO"]:
        try:
            df = yf.download(f"{stock_id}{suffix}", period="9mo", interval="1d", auto_adjust=False, progress=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                if len(df) >= 90:
                    return df
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_institution(stock_id, token_value):
    empty = {"外資":0, "投信":0, "自營商":0, "三大法人":0, "法人連買":0, "法人分":0, "法人原因":"法人無資料"}
    if not token_value:
        return {**empty, "法人原因":"未填 Token"}
    end = datetime.today().date()
    start = end - timedelta(days=35)
    params = {
        "dataset":"TaiwanStockInstitutionalInvestorsBuySell",
        "data_id":stock_id,
        "start_date":str(start),
        "end_date":str(end),
        "token":token_value,
    }
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
        df = pd.DataFrame(r.json().get("data", []))
    except Exception:
        return empty
    if df.empty or "buy" not in df.columns or "sell" not in df.columns:
        return empty
    name_col = next((c for c in ["name", "institutional_investors", "institutional_investors_name"] if c in df.columns), None)
    if not name_col:
        return empty
    df["net"] = df["buy"] - df["sell"]
    latest = df[df["date"] == df["date"].max()].copy()
    names = latest[name_col].astype(str)
    foreign = latest[names.str.contains("Foreign|外資", case=False, regex=True)]["net"].sum()
    trust = latest[names.str.contains("Investment|投信", case=False, regex=True)]["net"].sum()
    dealer = latest[names.str.contains("Dealer|自營", case=False, regex=True)]["net"].sum()
    total = foreign + trust + dealer
    daily = df.groupby("date")["net"].sum().tail(5)
    buy_days = int((daily > 0).sum())
    score, reasons = 0, []
    if foreign > 0:
        score += 12; reasons.append("外資買")
    if trust > 0:
        score += 18; reasons.append("投信買")
    if dealer > 0:
        score += 6; reasons.append("自營買")
    if total > 3000:
        score += 20; reasons.append("法人強買")
    elif total > 1000:
        score += 14; reasons.append("法人明顯買")
    elif total > 0:
        score += 8; reasons.append("法人偏買")
    if buy_days >= 5:
        score += 16; reasons.append("法人連5買")
    elif buy_days >= 3:
        score += 8; reasons.append("法人連3買")
    return {"外資":int(foreign), "投信":int(trust), "自營商":int(dealer), "三大法人":int(total), "法人連買":buy_days, "法人分":min(score, 60), "法人原因":"、".join(reasons) if reasons else "法人普通"}

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = -d.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    m = ema12 - ema26
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m-s

def kd(df):
    low = df["Low"].rolling(9).min()
    high = df["High"].rolling(9).max()
    k = 100*(df["Close"]-low)/(high-low).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)

def safe_slope(s, days=5):
    return len(s) > days and s.iloc[-1] > s.iloc[-days]

def platform_breakout(close, vol):
    if len(close) < 35: return False
    high = close.iloc[-31:-1].max(); low = close.iloc[-31:-1].min()
    if low <= 0: return False
    return ((high-low)/low <= 0.16) and (close.iloc[-1] > high) and (vol.iloc[-1] > vol.iloc[-6:-1].mean()*1.35)

def n_pattern(close):
    if len(close) < 50: return False
    r = close.iloc[-50:]
    low1 = r.iloc[:18].min(); high1 = r.iloc[12:32].max(); low2 = r.iloc[30:44].min(); now = r.iloc[-1]
    return high1 > low1*1.08 and low2 > low1*1.03 and now >= high1*0.97

def round_bottom(close):
    if len(close) < 75: return False
    a = close.iloc[-75:-50].mean(); b = close.iloc[-50:-25].mean(); c = close.iloc[-25:].mean()
    return b < a*0.98 and c > b*1.03 and close.iloc[-1] > close.iloc[-20:].mean()

def pullback_hold(close):
    if len(close) < 30: return False
    ma5 = close.rolling(5).mean(); ma20 = close.rolling(20).mean()
    return close.iloc[-5:].min() >= ma20.iloc[-1]*0.98 and close.iloc[-1] > ma5.iloc[-1]

def box_base(close):
    if len(close) < 30: return False
    hi = close.iloc[-25:].max(); lo = close.iloc[-25:].min()
    return lo > 0 and (hi-lo)/lo < 0.13

def tech_analyze(df):
    if df.empty or len(df) < 90:
        return None
    close, vol = df["Close"], df["Volume"]
    price, volume = float(close.iloc[-1]), int(vol.iloc[-1])
    ma5, ma10, ma20, ma60 = close.rolling(5).mean(), close.rolling(10).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    rr = rsi(close); mm, ss, hh = macd(close); kk, dd = kd(df)
    score, reasons, risks = 0, [], []
    if price > ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]: score += 8; reasons.append("短均多頭")
    if price > ma20.iloc[-1] > ma60.iloc[-1]: score += 8; reasons.append("站上月季線")
    if safe_slope(ma20, 5): score += 8; reasons.append("月線上彎")
    if safe_slope(ma60, 5): score += 5; reasons.append("季線上彎")
    if mm.iloc[-1] > ss.iloc[-1] and mm.iloc[-1] > 0: score += 12; reasons.append("MACD主升")
    if hh.iloc[-1] > hh.iloc[-2] > hh.iloc[-3]: score += 6; reasons.append("MACD柱增強")
    if rr.iloc[-1] > rr.iloc[-5] and 45 <= rr.iloc[-1] <= 72: score += 8; reasons.append("RSI趨勢向上")
    elif rr.iloc[-1] > 82: risks.append("RSI過熱")
    if kk.iloc[-1] > dd.iloc[-1] and kk.iloc[-1] < 85: score += 6; reasons.append("KD偏多")
    if vol.iloc[-1] > vol.iloc[-6:-1].mean()*1.5 and price > close.iloc[-2]: score += 10; reasons.append("量增價漲")
    if price >= close.rolling(60).max().iloc[-1]*0.97: score += 8; reasons.append("接近60日高")
    if platform_breakout(close, vol): score += 15; reasons.append("平台突破")
    if n_pattern(close): score += 14; reasons.append("N字第二波")
    if pullback_hold(close): score += 10; reasons.append("回踩不破")
    if round_bottom(close): score += 7; reasons.append("圓弧底")
    if box_base(close): score += 5; reasons.append("箱型整理")
    # 風險扣分
    if price < ma20.iloc[-1]: score -= 10; risks.append("跌破月線")
    if vol.iloc[-1] > vol.iloc[-6:-1].mean()*2.5 and close.iloc[-1] < close.iloc[-2]: score -= 8; risks.append("爆量收弱")
    if rr.iloc[-1] > 85: score -= 8
    return {"收盤價":round(price,2), "成交量":volume, "技術分":max(0, min(int(score), 100)), "RSI":round(float(rr.iloc[-1]),1), "MACD":round(float(mm.iloc[-1]),2), "技術原因":"、".join(reasons) if reasons else "技術普通", "風險":"、".join(risks) if risks else "無明顯風險"}

def main_force_score(tech, inst):
    score, reasons = 0, []
    if inst["三大法人"] > 0: score += 8; reasons.append("資金流入")
    if inst["投信"] > 0: score += 12; reasons.append("投信加持")
    if inst["外資"] > 1000: score += 8; reasons.append("外資明顯買")
    if "量增價漲" in tech["技術原因"]: score += 10; reasons.append("量價啟動")
    if "平台突破" in tech["技術原因"]: score += 12; reasons.append("突破平台")
    if "N字第二波" in tech["技術原因"]: score += 12; reasons.append("第二波型態")
    if "回踩不破" in tech["技術原因"]: score += 8; reasons.append("回踩守住")
    return min(score, 60), "、".join(reasons) if reasons else "主力普通"

def grade(score):
    if score >= 88: return "高信心"
    if score >= 78: return "優先研究"
    if score >= 68: return "可追蹤"
    if score >= 58: return "觀察"
    return "普通"

def ai_sentence(row):
    parts = []
    for key in ["平台突破", "N字第二波", "回踩不破", "MACD主升", "RSI趨勢向上", "量增價漲", "月線上彎"]:
        if key in row["技術原因"]: parts.append(key)
    if row["投信"] > 0: parts.append("投信買")
    if row["三大法人"] > 1000: parts.append("法人明顯買")
    if not parts: return "條件普通，先觀察，不追高。"
    risk = "；留意：" + row["風險"] if row["風險"] != "無明顯風險" else "。"
    return "、".join(parts[:6]) + "，符合你的策略條件，可列入研究" + risk

def scan_one(row, token_value, include_inst):
    sid, name = row["代號"], row["名稱"]
    try:
        df = get_price(sid)
        tech = tech_analyze(df)
        if tech is None: return None
        if tech["收盤價"] < min_price or tech["成交量"] < min_volume: return None
        inst = get_institution(sid, token_value) if include_inst else {"外資":0,"投信":0,"自營商":0,"三大法人":0,"法人連買":0,"法人分":0,"法人原因":"未啟用法人"}
        main_score, main_reason = main_force_score(tech, inst)
        if include_inst and token_value:
            total = round(tech["技術分"]*0.55 + inst["法人分"]*0.25 + main_score*0.20, 1)
        else:
            total = round(tech["技術分"]*0.80 + main_score*0.20, 1)
        if total < min_score: return None
        item = {
            "代號":sid, "名稱":name, "總分":total, "信心":grade(total),
            "技術分":tech["技術分"], "法人分":inst["法人分"], "主力分":main_score,
            "收盤價":tech["收盤價"], "成交量":tech["成交量"],
            "外資":inst["外資"], "投信":inst["投信"], "自營商":inst["自營商"], "三大法人":inst["三大法人"], "法人連買":inst["法人連買"],
            "RSI":tech["RSI"], "MACD":tech["MACD"], "技術原因":tech["技術原因"], "法人原因":inst["法人原因"], "主力原因":main_reason, "風險":tech["風險"]
        }
        item["AI一句話"] = ai_sentence(item)
        return item
    except Exception:
        return None

def scan_all():
    stocks = get_stock_list()
    if stocks.empty:
        st.error("抓不到股票清單，請確認 twstock 是否安裝成功。")
        return pd.DataFrame()
    total = len(stocks)
    progress = st.progress(0)
    status = st.empty()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(scan_one, row, token, use_inst) for _, row in stocks.iterrows()]
        for f in as_completed(futures):
            done += 1
            if done % 5 == 0 or done == total:
                progress.progress(done/total)
                status.write(f"掃描中：{done}/{total}")
            res = f.result()
            if res:
                results.append(res)
    status.write("掃描完成")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("總分", ascending=False).reset_index(drop=True)
    df.insert(0, "排名", df.index+1)
    return df

def show_table(title, df, cols):
    st.subheader(title)
    if df.empty:
        st.info("目前沒有符合條件的股票。")
    else:
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

base_cols = ["排名","代號","名稱","總分","信心","技術分","法人分","主力分","收盤價","RSI","AI一句話"]
detail_cols = ["排名","代號","名稱","總分","信心","技術分","法人分","主力分","收盤價","成交量","外資","投信","自營商","三大法人","法人連買","RSI","MACD","AI一句話","技術原因","法人原因","主力原因","風險"]

st.subheader("全池選股")
if st.button("開始掃描全池", use_container_width=True):
    result = scan_all()
    if result.empty:
        st.error("沒有掃到符合條件的股票。可降低最低分數或最低成交量。")
    else:
        st.success("掃描完成")
        show_table(f"AI 今日 Top {top_n}", result.head(top_n), base_cols)
        my_strategy = result[result["技術原因"].str.contains("平台突破|N字第二波|回踩不破|MACD主升", regex=True, na=False)]
        show_table("我的策略", my_strategy.head(30), base_cols)
        show_table("N字第二波", result[result["技術原因"].str.contains("N字第二波", na=False)].head(30), base_cols)
        show_table("平台突破", result[result["技術原因"].str.contains("平台突破", na=False)].head(30), base_cols)
        show_table("回踩不破", result[result["技術原因"].str.contains("回踩不破", na=False)].head(30), base_cols)
        show_table("完整排行榜", result, detail_cols)
        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載完整結果 CSV", csv, "future_stock_god_ai_scan.csv", "text/csv")

