import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures, time, re

st.set_page_config(page_title="未來小股神 AI 掃描 V32.1 全池修正版", layout="wide")

# ========= 基本資料 =========
FALLBACK_POOL = [
    # 常用股＋ETF備援；真正全池會優先用 twstock 載入
    ("2330","台積電","上市"),("2317","鴻海","上市"),("2303","聯電","上市"),("2409","友達","上市"),
    ("1714","和桐","上市"),("6271","同欣電","上市"),("6191","精成科","上櫃"),("3557","嘉威","上櫃"),
    ("3037","欣興","上市"),("2382","廣達","上市"),("2313","華通","上市"),("2344","華邦電","上市"),
    ("2359","所羅門","上市"),("3060","銘異","上市"),("8923","時報","上櫃"),("6272","驊陞","上櫃"),
    ("5468","凱鈺","上櫃"),("6259","百徽","上櫃"),("5211","蒙恬","上櫃"),("1730","花仙子","上市"),
    ("4183","福永生技","上櫃"),("5469","瀚宇博","上市"),("8183","精星","上櫃"),("2643","捷迅","上櫃"),
    ("1817","凱撒衛","上市"),("2013","中鋼構","上市"),("1731","美吾華","上市"),("3479","安勤","上櫃"),
    ("2739","寒舍","上市"),("4554","橙的","上櫃"),("1240","茂生農經","上櫃"),("7828","創新服務","興櫃"),
]

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_twstock_pool(include_emerging=False, include_etf=False):
    rows = []
    try:
        import twstock
        for code, info in twstock.codes.items():
            if not re.fullmatch(r"\d{4}", str(code)):
                continue
            name = getattr(info, "name", "") or ""
            market = getattr(info, "market", "") or ""
            typ = getattr(info, "type", "") or ""
            # 保留上市/上櫃普通股；興櫃可勾；ETF可勾
            is_listed = ("上市" in market) or ("上櫃" in market)
            is_emerging = "興櫃" in market
            is_etf = ("ETF" in typ.upper()) or ("ETF" in name.upper())
            if is_etf and not include_etf:
                continue
            if is_emerging and not include_emerging:
                continue
            if is_listed or (include_emerging and is_emerging):
                rows.append({"代號":str(code), "名稱":name, "市場":market, "類型":typ})
    except Exception:
        rows = []
    if not rows:
        rows = [{"代號":c, "名稱":n, "市場":m, "類型":"備援"} for c,n,m in FALLBACK_POOL]
    df = pd.DataFrame(rows).drop_duplicates("代號").sort_values("代號").reset_index(drop=True)
    return df

def symbol_for(row):
    code = str(row["代號"])
    market = str(row.get("市場",""))
    if "上櫃" in market or "興櫃" in market:
        return f"{code}.TWO"
    return f"{code}.TW"

def calc_indicators(df):
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    for n in [5,10,20,60,120,240]:
        df[f"MA{n}"] = close.rolling(n).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["MACD"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["OSC"] = df["DIF"] - df["MACD"]
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close-low9)/(high9-low9)*100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["量比"] = vol / vol.rolling(20).mean()
    return df

def ai_score(last, prev=None):
    score = 50
    reasons=[]
    c=last.get("Close", np.nan)
    ma5,ma10,ma20 = last.get("MA5",np.nan),last.get("MA10",np.nan),last.get("MA20",np.nan)
    rsi=last.get("RSI",np.nan)
    osc=last.get("OSC",np.nan)
    volr=last.get("量比",np.nan)
    if np.isfinite(ma5) and np.isfinite(ma10) and np.isfinite(ma20) and c>ma5>ma10>ma20:
        score+=22; reasons.append("均線多頭")
    if np.isfinite(rsi) and 50 <= rsi <= 72:
        score+=12; reasons.append("RSI健康")
    elif np.isfinite(rsi) and rsi>80:
        score-=10; reasons.append("RSI過熱")
    if np.isfinite(osc) and osc>0:
        score+=12; reasons.append("MACD偏多")
    if np.isfinite(volr) and volr>=1.2:
        score+=8; reasons.append("量能放大")
    elif np.isfinite(volr) and volr<0.8:
        score+=3; reasons.append("量縮整理")
    score=int(max(0,min(100,score)))
    if score>=92: t="1～3天"
    elif score>=84: t="2～5天"
    elif score>=75: t="5～10天"
    else: t="觀察中"
    return score,t,"、".join(reasons) if reasons else "等待訊號"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_one(code, name, market):
    try:
        import yfinance as yf
        sym = f"{code}.TWO" if ("上櫃" in market or "興櫃" in market) else f"{code}.TW"
        df = yf.download(sym, period="9mo", interval="1d", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty or len(df)<60:
            # 備援換副檔名試一次
            alt = f"{code}.TW" if sym.endswith(".TWO") else f"{code}.TWO"
            df = yf.download(alt, period="9mo", interval="1d", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty or len(df)<60 or "Close" not in df:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = calc_indicators(df)
        last = df.iloc[-1]
        score, fire_time, reason = ai_score(last)
        close = float(last["Close"])
        return {
            "代號":code, "名稱":name, "市場":market, "現價":round(close,2),
            "爆發指數":score, "AI信心":f"{min(99, max(50, score-1))}%",
            "預估發動時間":fire_time, "RSI":round(float(last.get("RSI",np.nan)),1) if np.isfinite(last.get("RSI",np.nan)) else "-",
            "MACD":round(float(last.get("OSC",np.nan)),3) if np.isfinite(last.get("OSC",np.nan)) else "-",
            "量比":round(float(last.get("量比",np.nan)),2) if np.isfinite(last.get("量比",np.nan)) else "-",
            "AI解讀":reason
        }
    except Exception:
        return None

def scan_pool(pool_df, max_workers=12, max_scan=None):
    data = pool_df.to_dict("records")
    if max_scan:
        data = data[:max_scan]
    out=[]
    bar=st.progress(0)
    status=st.empty()
    total=len(data)
    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs=[ex.submit(fetch_one, r["代號"], r["名稱"], r["市場"]) for r in data]
        for fut in concurrent.futures.as_completed(futs):
            done+=1
            res=fut.result()
            if res: out.append(res)
            if done % 10 == 0 or done==total:
                bar.progress(done/total)
                status.write(f"AI掃描中：{done}/{total}，成功 {len(out)} 檔（抓不到會自動跳過）")
    return pd.DataFrame(out)

@st.cache_data(ttl=900, show_spinner=False)
def market_table():
    rows=[]
    for code,name in [("^TWII","加權指數"),("^TWOII","櫃買OTC")]:
        try:
            import yfinance as yf
            df=yf.download(code, period="6mo", interval="1d", progress=False, threads=False)
            if df is None or df.empty or len(df)<30:
                raise ValueError("no data")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns=df.columns.get_level_values(0)
            df=calc_indicators(df)
            last=df.iloc[-1]
            score,t,reason=ai_score(last)
            rows.append({"市場":name,"收盤":round(float(last["Close"]),2),"AI分數":score,"RSI":round(float(last["RSI"]),1),"MACD":round(float(last["OSC"]),3),"AI解讀":reason})
        except Exception:
            rows.append({"市場":name,"收盤":"-","AI分數":"-","RSI":"-","MACD":"-","AI解讀":"資料暫缺"})
    return pd.DataFrame(rows)

# ========= UI =========
st.title("🚀 未來小股神 AI 掃描 V32.1 單檔全池修正版")
st.caption("保留 V32.1 介面，只修正：股票池改成上市＋上櫃全市場，不再只有 910 檔。")

with st.sidebar:
    st.header("設定")
    include_emerging = st.checkbox("包含興櫃", value=False)
    include_etf = st.checkbox("包含 ETF", value=False)
    scan_limit = st.selectbox("掃描範圍", ["全池", "前300檔測試", "前100檔測試"], index=0)
    workers = st.slider("掃描速度", 4, 24, 12)

pool = load_twstock_pool(include_emerging, include_etf)
st.success(f"已載入股票池：{len(pool)} 檔（上市＋上櫃" + ("＋興櫃" if include_emerging else "") + ("＋ETF" if include_etf else "") + "）")
st.dataframe(pool.head(20), use_container_width=True, hide_index=True)

st.subheader("❤️ 7828 信仰股")
faith = fetch_one("7828", "創新服務", "興櫃")
if faith:
    st.dataframe(pd.DataFrame([faith]), use_container_width=True, hide_index=True)
else:
    st.warning("7828 暫時抓不到資料：多半是資料源尚未支援興櫃，主程式不會因此掛掉。")

st.subheader("📊 AI 大盤分析")
st.dataframe(market_table(), use_container_width=True, hide_index=True)

st.subheader("🔥 今日 AI TOP20（全市場）")
st.caption("首頁只顯示 TOP20，不會一次塞滿整頁。全池掃描可能需要幾分鐘。")
max_scan = None if scan_limit=="全池" else (300 if "300" in scan_limit else 100)
if st.button("開始全市場掃描", type="primary"):
    df=scan_pool(pool, max_workers=workers, max_scan=max_scan)
    if df.empty:
        st.error("目前資料源抓不到足夠資料，請稍後再試。")
    else:
        df=df.sort_values(["爆發指數"], ascending=False).reset_index(drop=True)
        st.session_state["scan_result"]=df

if "scan_result" in st.session_state:
    df=st.session_state["scan_result"]
    topn = st.slider("顯示前幾名", 10, min(100, len(df)), min(20, len(df)))
    st.dataframe(df.head(topn), use_container_width=True, hide_index=True)
    st.download_button("下載掃描結果 CSV", df.to_csv(index=False).encode("utf-8-sig"), "ai_scan_result.csv", "text/csv")

st.subheader("🔍 單股掃描")
q=st.text_input("輸入代號，例如 6271、1714、2409、7828", value="6271")
if st.button("分析單股"):
    row=pool[pool["代號"].astype(str)==q.strip()]
    if row.empty:
        name="自訂"; market="上市"
    else:
        name=row.iloc[0]["名稱"]; market=row.iloc[0]["市場"]
    res=fetch_one(q.strip(), name, market)
    if res:
        st.dataframe(pd.DataFrame([res]), use_container_width=True, hide_index=True)
    else:
        st.warning(f"{q} 暫時抓不到資料或資料不足。")
