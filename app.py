import io
import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="未來小股神 AI 操盤中心 V33 Core", layout="wide")

FAITH_CODE = "7828"

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_stock_pool():
    rows = []
    # TWSE listed
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        dfs = pd.read_html(url, encoding="big5")
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:].copy()
        col = "有價證券代號及名稱"
        df = df[df[col].astype(str).str.match(r"^\d{4}", na=False)]
        for _, r in df.iterrows():
            txt = str(r[col])
            code = txt.split()[0]
            name = txt.replace(code, "", 1).strip()
            if code.isdigit() and len(code) == 4:
                rows.append({"代號": code, "名稱": name, "市場": "上市", "suffix": ".TW"})
    except Exception:
        pass
    # TPEX otc
    try:
        url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
        dfs = pd.read_html(url, encoding="big5")
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:].copy()
        col = "有價證券代號及名稱"
        df = df[df[col].astype(str).str.match(r"^\d{4}", na=False)]
        for _, r in df.iterrows():
            txt = str(r[col])
            code = txt.split()[0]
            name = txt.replace(code, "", 1).strip()
            if code.isdigit() and len(code) == 4:
                rows.append({"代號": code, "名稱": name, "市場": "上櫃", "suffix": ".TWO"})
    except Exception:
        pass
    pool = pd.DataFrame(rows).drop_duplicates("代號") if rows else pd.DataFrame()
    if pool.empty:
        pool = pd.DataFrame([
            {"代號":"2330","名稱":"台積電","市場":"上市","suffix":".TW"},
            {"代號":"2303","名稱":"聯電","市場":"上市","suffix":".TW"},
            {"代號":"2409","名稱":"友達","市場":"上市","suffix":".TW"},
            {"代號":"6271","名稱":"同欣電","市場":"上市","suffix":".TW"},
            {"代號":"6191","名稱":"精成科","市場":"上市","suffix":".TW"},
            {"代號":"3567","名稱":"逸昌","市場":"上櫃","suffix":".TWO"},
            {"代號":"7828","名稱":"創新服務","市場":"未知","suffix":".TWO"},
        ])
    return pool

@st.cache_data(ttl=30*60, show_spinner=False)
def get_price_data(code, suffix_hint=None, period="8mo"):
    suffixes = []
    if suffix_hint: suffixes.append(suffix_hint)
    suffixes += [".TW", ".TWO"]
    seen = []
    for s in suffixes:
        if s in seen: continue
        seen.append(s)
        try:
            df = yf.download(f"{code}{s}", period=period, interval="1d", progress=False, auto_adjust=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if df is not None and not df.empty and "Close" in df.columns and df["Close"].dropna().shape[0] >= 35:
                return df.dropna(), s, None
        except Exception as e:
            last_err = str(e)
    return pd.DataFrame(), None, "資料不足或資料源暫時失敗"

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def kd(high, low, close, n=9):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d

def analyze_df(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"].fillna(0)
    last = c.iloc[-1]
    ma5, ma10, ma20, ma60 = c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    rr = rsi(c)
    dif, dea, hist = macd(c)
    k, d = kd(h, l, c)
    vol20 = v.rolling(20).mean()
    vol_ratio = float(v.iloc[-1] / vol20.iloc[-1]) if vol20.iloc[-1] and vol20.iloc[-1] > 0 else 0
    high60 = c.rolling(60).max().iloc[-1]
    near_high = (high60 - last) / high60 <= 0.03 if high60 and high60 > 0 else False
    platform_high = h.tail(25).max()
    platform_low = l.tail(25).min()
    range_pct = (platform_high - platform_low) / last if last else 1
    near_break = (platform_high - last) / platform_high <= 0.025 if platform_high else False
    platform = range_pct <= 0.18 and near_break
    macd_bull = dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > hist.iloc[-2]
    kd_gold = k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]
    ma_bull = ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]
    above_ma20 = last > ma20.iloc[-1]
    breakout = last >= platform_high * 0.995 and vol_ratio >= 1.2
    long_black = (df["Open"].iloc[-1] > df["Close"].iloc[-1]) and ((df["Open"].iloc[-1]-df["Close"].iloc[-1]) / last > 0.035) and vol_ratio > 1.8
    score = 0; plus=[]; minus=[]
    if ma_bull: score += 10; plus.append("均線多頭+10")
    if above_ma20: score += 5; plus.append("站上月線+5")
    if near_high: score += 10; plus.append("接近60日高+10")
    if macd_bull: score += 10; plus.append("MACD多頭+10")
    if kd_gold: score += 5; plus.append("KD黃金交叉+5")
    rsi_val = float(rr.iloc[-1]) if not np.isnan(rr.iloc[-1]) else 50
    if 50 <= rsi_val <= 70: score += 10; plus.append("RSI健康+10")
    if vol_ratio > 1.5: score += 10; plus.append("量比>1.5 +10")
    if breakout: score += 10; plus.append("接近/突破平台+10")
    if platform: score += 8; plus.append("平台整理+8")
    # Simple chip placeholder using volume-price behavior, not fake法人
    if last > c.iloc[-6] and v.tail(5).mean() > v.tail(20).mean()*0.9:
        score += 8; plus.append("短線資金偏強+8")
    if rsi_val > 80: score -= 10; minus.append("RSI過熱-10")
    if long_black: score -= 20; minus.append("爆量長黑-20")
    if last < ma5.iloc[-1]: score -= 15; minus.append("跌破5MA-15")
    if dif.iloc[-1] < dea.iloc[-1] and hist.iloc[-1] < hist.iloc[-2]: score -= 15; minus.append("MACD轉弱-15")
    if not near_break and (platform_high-last)/platform_high > 0.05: score -= 10; minus.append("離突破較遠-10")
    score = int(max(0, min(100, score)))
    if score >= 95: conf = 95
    elif score >= 90: conf = 90
    elif score >= 85: conf = 85
    elif score >= 80: conf = 80
    elif score >= 75: conf = 75
    else: conf = max(50, score)
    if rsi_val > 80 or long_black: launch = "高風險"
    elif breakout and vol_ratio > 1.2: launch = "已發動"
    elif score >= 90 and near_break: launch = "1～3天"
    elif score >= 80 and platform: launch = "2～5天"
    elif score >= 70: launch = "5～10天"
    else: launch = "觀察中"
    stars = "⭐" * max(1, min(5, int(np.ceil(score/20))))
    support1 = float(min(ma20.iloc[-1], l.tail(10).min())) if not np.isnan(ma20.iloc[-1]) else float(l.tail(10).min())
    pressure1 = float(platform_high)
    target1 = pressure1
    target2 = pressure1 + (pressure1 - support1) * 0.5
    target3 = pressure1 + (pressure1 - support1)
    if score >= 85:
        ai = "型態與動能偏多，留意量能是否續強。"
    elif score >= 75:
        ai = "整理中偏多，等待突破或回測確認。"
    elif score >= 60:
        ai = "訊號普通，先觀察不急追。"
    else:
        ai = "分數偏低，暫不列為優先。"
    return {
        "收盤": round(float(last),2), "爆發指數": score, "AI信心": f"{conf}%", "發動時間": launch, "評等": stars,
        "RSI": round(rsi_val,1), "量比": round(vol_ratio,2), "MACD": "多" if macd_bull else "弱/整理",
        "KD": "金叉" if kd_gold else "整理", "MA排列": "多頭" if ma_bull else "非多頭",
        "支撐": round(support1,2), "壓力": round(pressure1,2),
        "目標1": round(target1,2), "目標2": round(target2,2), "目標3": round(target3,2),
        "AI解讀": ai, "加分": "、".join(plus) if plus else "無明顯加分", "扣分": "、".join(minus) if minus else "無重大扣分"
    }

def analyze_stock(code, name="", suffix=None):
    df, used_suffix, err = get_price_data(code, suffix)
    if df.empty:
        return {"代號": code, "名稱": name, "狀態": err or "無資料", "爆發指數": 0, "AI信心": "-", "發動時間": "資料不足"}, df
    res = analyze_df(df)
    res.update({"代號": code, "名稱": name, "狀態": "OK", "市場代號": used_suffix})
    return res, df

@st.cache_data(ttl=20*60, show_spinner=True)
def scan_market(max_scan=250):
    pool = load_stock_pool()
    out = []
    # prioritize common liquid names but still use pool order; user can increase max_scan
    for _, row in pool.head(max_scan).iterrows():
        res, _ = analyze_stock(row["代號"], row["名稱"], row.get("suffix"))
        if res.get("狀態") == "OK": out.append(res)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    return df.sort_values(["爆發指數", "RSI"], ascending=[False, True]).reset_index(drop=True)

@st.cache_data(ttl=10*60, show_spinner=False)
def market_summary():
    data=[]
    for label, ticker in [("加權指數","^TWII"),("櫃買OTC","^TWOII")]:
        try:
            df = yf.download(ticker, period="3mo", progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            if df.empty: raise ValueError("no data")
            c=df["Close"]; ma20=c.rolling(20).mean()
            score = 70 + (10 if c.iloc[-1] > ma20.iloc[-1] else -10) + (10 if c.iloc[-1] > c.iloc[-5] else -5)
            data.append({"項目":label,"收盤":round(float(c.iloc[-1]),2),"AI分數":int(max(0,min(100,score))),"解讀":"偏多" if score>=75 else "震盪/保守"})
        except Exception:
            data.append({"項目":label,"收盤":"-","AI分數":"-","解讀":"資料源暫時失敗"})
    return pd.DataFrame(data)

def plot_k(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線"))
    for n in [5,10,20,60]:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(n).mean(), mode="lines", name=f"MA{n}"))
    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False)
    return fig

st.title("🚀 未來小股神 AI 操盤中心 V33 Core")
st.caption("真評分版：不全部100分、不全部99%、保留單股掃描＋全池掃描＋發動時間。")

pool = load_stock_pool()

with st.sidebar:
    st.header("設定")
    max_scan = st.slider("全池掃描檔數（越大越慢）", 50, int(min(len(pool), 2000)), 250, step=50)
    st.write(f"股票池：{len(pool)} 檔")

# Faith
st.subheader("❤️ 7828 信仰股")
faith_row = pool[pool["代號"] == FAITH_CODE]
faith_name = faith_row["名稱"].iloc[0] if not faith_row.empty else "創新服務"
faith_suffix = faith_row["suffix"].iloc[0] if not faith_row.empty else None
faith, faith_df = analyze_stock(FAITH_CODE, faith_name, faith_suffix)
cols = st.columns(5)
cols[0].metric("爆發", faith.get("爆發指數",0))
cols[1].metric("信心", faith.get("AI信心","-"))
cols[2].metric("發動", faith.get("發動時間","-"))
cols[3].metric("RSI", faith.get("RSI","-"))
cols[4].metric("狀態", faith.get("狀態","-"))
st.write(faith.get("AI解讀", faith.get("狀態", "")))

st.subheader("📊 AI 大盤分析")
st.dataframe(market_summary(), use_container_width=True, hide_index=True)

tab1, tab2, tab3 = st.tabs(["🔥 全池TOP20", "🔍 單股掃描", "📋 股票池"])

with tab1:
    if st.button("開始全池掃描 / 更新TOP20", type="primary"):
        st.cache_data.clear()
    with st.spinner("掃描中，檔數越多越久..."):
        top = scan_market(max_scan)
    if top.empty:
        st.warning("資料源暫時沒有回傳可分析股票，請稍後重試。")
    else:
        show_cols = ["代號","名稱","爆發指數","AI信心","發動時間","評等","收盤","RSI","量比","MA排列","MACD","KD","AI解讀","加分","扣分"]
        st.dataframe(top[show_cols].head(20), use_container_width=True, hide_index=True)
        csv = top.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載本次掃描CSV", csv, f"v33_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

with tab2:
    q = st.text_input("輸入股票代號或名稱", value="2330")
    if q:
        row = pool[(pool["代號"] == q) | (pool["名稱"].astype(str).str.contains(q, na=False))]
        if not row.empty:
            code = row.iloc[0]["代號"]; name = row.iloc[0]["名稱"]; suffix = row.iloc[0]["suffix"]
        else:
            code = ''.join([ch for ch in q if ch.isdigit()]) or q; name = q; suffix = None
        res, df = analyze_stock(code, name, suffix)
        st.markdown(f"### {res.get('代號')} {res.get('名稱','')}")
        c = st.columns(6)
        for col, key in zip(c, ["爆發指數","AI信心","發動時間","評等","收盤","狀態"]):
            col.metric(key, res.get(key,"-"))
        st.write("**AI一句話：**", res.get("AI解讀", res.get("狀態","")))
        if not df.empty:
            st.plotly_chart(plot_k(df.tail(120), f"{code} {name} K線"), use_container_width=True)
            detail = pd.DataFrame([res])
            st.dataframe(detail, use_container_width=True, hide_index=True)

with tab3:
    st.write(f"目前股票池共 {len(pool)} 檔（上市＋上櫃，資料源失敗時會用備援池）。")
    st.dataframe(pool[["代號","名稱","市場"]], use_container_width=True, hide_index=True)
