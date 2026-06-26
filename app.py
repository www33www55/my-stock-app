import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

try:
    import twstock
except Exception:
    twstock = None

st.set_page_config(page_title='未來小股神 AI 操盤中心 V30 Ultimate', layout='wide', page_icon='🚀')

# -----------------------------
# 樣式
# -----------------------------
st.markdown('''
<style>
.block-container{padding-top:1.2rem;}
.big-title{font-size:34px;font-weight:900;margin-bottom:0px;}
.sub{color:#888;font-size:14px;}
.card{border:1px solid #333;border-radius:18px;padding:18px;margin:8px 0;background:rgba(255,255,255,0.03)}
.metric-title{font-size:14px;color:#999}.metric-value{font-size:28px;font-weight:800}
.good{color:#20c997}.warn{color:#ffc107}.bad{color:#ff6b6b}
.rank{font-size:22px;font-weight:800}
</style>
''', unsafe_allow_html=True)

FALLBACK_POOL = {
    '7828':'創新服務','2409':'友達','2303':'聯電','6271':'同欣電','6191':'精成科','3567':'逸昌',
    '5211':'蒙恬','1730':'花仙子','1714':'和桐','3037':'欣興','2313':'華通','2382':'廣達',
    '2330':'台積電','2359':'所羅門','2344':'華邦電','3060':'銘異','8923':'時報','5468':'凱鈺'
}
SECTOR_MAP = {
    '7828':'創新服務/資訊','2409':'面板','2303':'半導體','6271':'半導體','6191':'PCB','3567':'IC設計',
    '5211':'軟體','1730':'生活消費','1714':'化工','3037':'載板/PCB','2313':'PCB','2382':'AI伺服器',
    '2330':'半導體','2359':'機器人/自動化','2344':'記憶體','3060':'電子零組件','8923':'文化媒體','5468':'IC設計'
}

@st.cache_data(ttl=60*60*6)
def get_stock_pool(mode='全池'):
    if mode == '自訂示範池':
        return pd.DataFrame([{'code':c,'name':n,'sector':SECTOR_MAP.get(c,'其他')} for c,n in FALLBACK_POOL.items()])
    rows = []
    if twstock is not None:
        for code, info in twstock.codes.items():
            if len(code) == 4 and code.isdigit() and info.type == '股票':
                rows.append({'code':code,'name':info.name,'sector':getattr(info, 'group', '') or SECTOR_MAP.get(code,'其他')})
    if not rows:
        rows = [{'code':c,'name':n,'sector':SECTOR_MAP.get(c,'其他')} for c,n in FALLBACK_POOL.items()]
    return pd.DataFrame(rows).drop_duplicates('code').reset_index(drop=True)

@st.cache_data(ttl=60*30)
def fetch_price(code, period='6mo'):
    symbols = [f'{code}.TW', f'{code}.TWO']
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, auto_adjust=False, progress=False, threads=False)
            if df is not None and len(df) >= 35:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.dropna().copy()
                return df
        except Exception:
            pass
    return pd.DataFrame()

def rsi(series, n=14):
    delta = series.diff()
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

def kd(df, n=9):
    low = df['Low'].rolling(n).min()
    high = df['High'].rolling(n).max()
    rsv = (df['Close'] - low) / (high - low).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    return k, d



def bollinger(close, n=20, k=2):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return mid + k*std, mid, mid - k*std

def atr(df, n=14):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_v = tr.rolling(n).mean()
    plus_di = 100 * plus_dm.rolling(n).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(n).mean() / atr_v.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(n).mean()

def obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].replace(0, np.nan).cumsum()

def tech_summary(df):
    c = df['Close']; v = df['Volume']
    ma5, ma10, ma20, ma60, ma120, ma240 = [c.rolling(n).mean() for n in [5,10,20,60,120,240]]
    rv = rsi(c).iloc[-1]
    dif, dea, hist = macd(c)
    k, d = kd(df)
    bb_up, bb_mid, bb_low = bollinger(c)
    atr_v = atr(df).iloc[-1]
    adx_v = adx(df).iloc[-1]
    obv_v = obv(df).iloc[-1]
    vwap_v = vwap(df).iloc[-1]
    roc_v = (c.iloc[-1] / c.iloc[-11] - 1) * 100 if len(c) > 11 else np.nan
    close = c.iloc[-1]
    trend = '多頭排列' if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] else ('空頭排列' if ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1] else '整理')
    macd_state = '主升段' if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-1] > 0 else ('轉弱' if dif.iloc[-1] < dea.iloc[-1] else '醞釀')
    kd_state = '黃金交叉' if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] else ('死亡交叉' if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] else '延續')
    bb_state = '貼上緣偏強' if close > bb_up.iloc[-1]*0.985 else ('接近下緣' if close < bb_low.iloc[-1]*1.015 else '通道內')
    pressure1 = float(df['High'].tail(25).max()); support1 = float(df['Low'].tail(20).min())
    support2 = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else support1
    pressure2 = float(df['High'].tail(60).max()) if len(df)>=60 else pressure1
    score = 50
    if trend == '多頭排列': score += 18
    if ma20.iloc[-1] > ma20.iloc[-5]: score += 8
    if macd_state == '主升段': score += 14
    if 52 <= rv <= 72: score += 8
    if close > ma20.iloc[-1]: score += 8
    if adx_v >= 20: score += 4
    score = int(np.clip(score, 0, 100))
    return {
        '技術分數': score, '收盤價': round(float(close),2), 'MA5':round(float(ma5.iloc[-1]),2), 'MA10':round(float(ma10.iloc[-1]),2),
        'MA20':round(float(ma20.iloc[-1]),2), 'MA60':round(float(ma60.iloc[-1]),2) if not np.isnan(ma60.iloc[-1]) else np.nan,
        'MA120':round(float(ma120.iloc[-1]),2) if not np.isnan(ma120.iloc[-1]) else np.nan, 'MA240':round(float(ma240.iloc[-1]),2) if not np.isnan(ma240.iloc[-1]) else np.nan,
        '均線狀態':trend, 'RSI':round(float(rv),1), 'K':round(float(k.iloc[-1]),1), 'D':round(float(d.iloc[-1]),1), 'KD狀態':kd_state,
        'DIF':round(float(dif.iloc[-1]),3), 'DEA':round(float(dea.iloc[-1]),3), 'MACD柱':round(float(hist.iloc[-1]),3), 'MACD狀態':macd_state,
        '布林上緣':round(float(bb_up.iloc[-1]),2), '布林中線':round(float(bb_mid.iloc[-1]),2), '布林下緣':round(float(bb_low.iloc[-1]),2), '布林狀態':bb_state,
        'ATR':round(float(atr_v),2) if not np.isnan(atr_v) else np.nan, 'ADX':round(float(adx_v),1) if not np.isnan(adx_v) else np.nan,
        'OBV':round(float(obv_v),0) if not np.isnan(obv_v) else np.nan, 'VWAP':round(float(vwap_v),2) if not np.isnan(vwap_v) else np.nan, 'ROC10%':round(float(roc_v),2) if not np.isnan(roc_v) else np.nan,
        '第一支撐':round(min(support1, support2),2), '第二支撐':round(max(support1, support2),2), '第一壓力':round(pressure1,2), '第二壓力':round(pressure2,2),
        'AI技術一句話': f'技術面{trend}，MACD{macd_state}，KD{kd_state}，RSI {rv:.1f}，目前{bb_state}。'
    }

def detect_patterns(df):
    c = df['Close']; h = df['High']; l = df['Low']
    recent_high = h.tail(25).max(); recent_low = l.tail(25).min()
    width = (recent_high - recent_low) / max(recent_low, 1)
    platform = width < 0.18 and c.iloc[-1] > c.tail(25).mean()
    near_break = (recent_high - c.iloc[-1]) / max(c.iloc[-1], 1) * 100
    n_shape = len(c) > 50 and c.iloc[-1] > c.iloc[-20] and c.iloc[-20] > c.iloc[-35]
    round_bottom = c.tail(45).iloc[:15].mean() > c.tail(45).iloc[15:30].mean() and c.tail(45).iloc[-10:].mean() > c.tail(45).iloc[15:30].mean()
    pullback_hold = c.iloc[-1] >= c.rolling(20).mean().iloc[-1] and c.iloc[-1] < recent_high
    tags=[]
    if platform: tags.append('平台整理')
    if near_break <= 3: tags.append('快突破')
    if n_shape: tags.append('N字')
    if round_bottom: tags.append('圓弧底')
    if pullback_hold: tags.append('回踩不破')
    return tags, near_break

def launch_time(score, near_break, rsi_v, vol_ratio):
    if score >= 95 and near_break <= 1.2 and 55 <= rsi_v <= 76 and vol_ratio >= 1.1:
        return '🔥 今天～1天', 'Day 0'
    if score >= 90 and near_break <= 2.5:
        return '🔥 1～3天', 'Day 1'
    if score >= 84 and near_break <= 4:
        return '🚀 2～5天', 'Day 2'
    if score >= 76:
        return '⭐ 5～10天', 'Day 5'
    return '👀 觀察中', '-'

def analyze_stock(code, name='', sector='其他'):
    df = fetch_price(code)
    if df.empty:
        return None
    c = df['Close']; v = df['Volume']
    ma5, ma10, ma20, ma60 = c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    r = rsi(c).iloc[-1]
    dif, dea, hist = macd(c)
    k, d = kd(df)
    tech = tech_summary(df)
    tags, near_break = detect_patterns(df)
    vol_ratio = float(v.iloc[-1] / max(v.rolling(20).mean().iloc[-1], 1))
    close = float(c.iloc[-1])

    score = 0
    reasons = []
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]: score += 18; reasons.append('均線多頭排列')
    if ma20.iloc[-1] > ma20.iloc[-5]: score += 8; reasons.append('月線上彎')
    if close >= ma5.iloc[-1]: score += 8; reasons.append('站上5日線')
    if 52 <= r <= 72: score += 12; reasons.append('RSI健康')
    elif 72 < r <= 80: score += 6; reasons.append('RSI偏熱')
    elif r > 80: score -= 5; reasons.append('RSI過熱')
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-1] > 0: score += 15; reasons.append('MACD主升段')
    if hist.iloc[-1] > hist.iloc[-2]: score += 6; reasons.append('MACD柱體放大')
    if vol_ratio >= 1.3: score += 10; reasons.append('量能放大')
    elif 0.75 <= vol_ratio < 1.3: score += 5; reasons.append('量能溫和')
    if '平台整理' in tags: score += 10
    if '快突破' in tags: score += 12
    if 'N字' in tags: score += 8
    if '圓弧底' in tags: score += 8
    if '回踩不破' in tags: score += 8
    score = int(max(0, min(100, score)))
    t, countdown = launch_time(score, near_break, float(r), vol_ratio)

    risk = '低' if score >= 88 and r < 78 else ('中' if score >= 70 else '高')
    if r > 82 or close / max(ma20.iloc[-1],1) > 1.18: risk = '高'
    star = '★★★★★' if score >= 90 else '★★★★☆' if score >= 82 else '★★★☆☆' if score >= 70 else '★★☆☆☆'
    stop = min(float(ma20.iloc[-1]), float(df['Low'].tail(20).min()))
    box_h = float(df['High'].tail(25).max() - df['Low'].tail(25).min())
    target1 = close + box_h * 0.6
    target2 = close + box_h * 1.0
    target3 = close + box_h * 1.618
    chip = int(np.clip(score*0.75 + vol_ratio*8 + (5 if dif.iloc[-1] > dea.iloc[-1] else -5), 0, 100))
    legal = int(np.clip(score*0.6 + (10 if vol_ratio > 1 else 0), 0, 100))
    best_buy = '回踩5日線' if close > ma5.iloc[-1] else '站回5日線'
    if near_break <= 2: best_buy = '突破平台高點'
    if risk == '高': best_buy = '不追，等回測'
    ai = f"{','.join(tags[:3]) or '觀察中'}；爆發{score}分，預估{t}，風險{risk}。"

    return {
        '股票代號':code,'股票名稱':name or FALLBACK_POOL.get(code,''),'產業':sector or SECTOR_MAP.get(code,'其他'),
        '收盤價':round(close,2),'爆發指數':score,'AI信心值':score,'星級':star,'預估發動時間':t,'發動倒數':countdown,
        '距離突破%':round(float(near_break),2),'RSI':round(float(r),1),'K':round(float(k.iloc[-1]),1),'D':round(float(d.iloc[-1]),1),
        '技術分數':tech['技術分數'],'均線狀態':tech['均線狀態'],'MACD狀態':tech['MACD狀態'],'KD狀態':tech['KD狀態'],'第一支撐':tech['第一支撐'],'第一壓力':tech['第一壓力'],
        '量比':round(vol_ratio,2),'MA5':round(float(ma5.iloc[-1]),2),'MA10':round(float(ma10.iloc[-1]),2),'MA20':round(float(ma20.iloc[-1]),2),
        '主力建倉率':chip,'三大法人共振':legal,'停損':round(stop,2),'第一目標':round(target1,2),'第二目標':round(target2,2),'第三目標':round(target3,2),
        '最佳買點':best_buy,'風險':risk,'型態':','.join(tags),'發動原因':'、'.join(reasons[:6]),'AI一句話':ai,
        '_df':df
    }

@st.cache_data(ttl=60*20, show_spinner=False)
def scan_pool(codes, names, sectors, limit):
    out=[]
    for code,name,sector in zip(codes[:limit], names[:limit], sectors[:limit]):
        res = analyze_stock(str(code), str(name), str(sector))
        if res: 
            res.pop('_df', None)
            out.append(res)
    if not out: return pd.DataFrame()
    return pd.DataFrame(out).sort_values(['爆發指數','距離突破%'], ascending=[False, True]).reset_index(drop=True)

def market_analysis():
    items = {'加權指數':'^TWII','櫃買OTC':'^TWOII'}
    rows=[]
    for name,symbol in items.items():
        try:
            df=yf.download(symbol, period='6mo', progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns=[c[0] for c in df.columns]
            c=df['Close']; ma20=c.rolling(20).mean(); ma60=c.rolling(60).mean()
            rv = rsi(c).iloc[-1]
            score=50
            if c.iloc[-1]>ma20.iloc[-1]: score+=15
            if ma20.iloc[-1]>ma60.iloc[-1]: score+=15
            if 45<=rv<=75: score+=10
            if c.iloc[-1]>c.iloc[-5]: score+=10
            rows.append({'市場':name,'收盤':round(float(c.iloc[-1]),2),'AI分數':min(score,100),'RSI':round(float(rv),1),'狀態':'偏多' if score>=75 else '震盪' if score>=55 else '偏弱'})
        except Exception:
            rows.append({'市場':name,'收盤':np.nan,'AI分數':60,'RSI':np.nan,'狀態':'資料不足'})
    return pd.DataFrame(rows)

def plot_k(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K'))
    for n in [5,10,20,60]:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(n).mean(), mode='lines', name=f'MA{n}'))
    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False)
    return fig

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title('🚀 V30 Ultimate')
page = st.sidebar.radio('功能', ['首頁戰情中心','全池AI掃描','技術分析中心','個股搜尋','三大法人AI中心','歷史回測','設定說明'])
pool_mode = st.sidebar.selectbox('股票池', ['全池','自訂示範池'])
pool = get_stock_pool(pool_mode)
scan_limit = st.sidebar.slider('本次掃描檔數（全池很慢，先用300～800）', 50, min(2000, len(pool)), min(300, len(pool)), 50)
show_n = st.sidebar.selectbox('首頁顯示', [20,50,100,'全部'], index=0)

st.markdown('<div class="big-title">🔥 未來小股神 AI 操盤中心 V30 Ultimate</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">全池掃描｜股票名稱｜發動日｜大盤分析｜三大法人｜歷史回測｜❤️7828 信仰模式</div>', unsafe_allow_html=True)

# -----------------------------
# Pages
# -----------------------------
if page == '首頁戰情中心':
    st.subheader('❤️ 7828 信仰股')
    faith = analyze_stock('7828','創新服務','創新服務/資訊')
    if faith:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric('爆發指數', faith['爆發指數'])
        c2.metric('預估發動', faith['預估發動時間'])
        c3.metric('主力建倉率', f"{faith['主力建倉率']}%")
        c4.metric('最佳買點', faith['最佳買點'])
        c5.metric('風險', faith['風險'])
        st.info(faith['AI一句話'])
    else:
        st.warning('7828 今日抓不到資料，可能是資料源暫時缺漏。')

    st.subheader('📊 AI 大盤分析')
    mkt = market_analysis()
    st.dataframe(mkt, use_container_width=True, hide_index=True)
    avg = pd.to_numeric(mkt['AI分數'], errors='coerce').mean()
    st.success('大盤AI一句話：' + ('大盤偏多，可優先找主升段與突破股。' if avg>=75 else '大盤震盪，分批布局、避免追高。' if avg>=55 else '大盤偏弱，降低部位、等待確認。'))

    st.subheader('🔥 今日 AI TOP 榜')
    with st.spinner('AI 掃描中，第一次會比較久...'):
        topdf = scan_pool(pool['code'].tolist(), pool['name'].tolist(), pool['sector'].tolist(), scan_limit)
    if topdf.empty:
        st.error('目前沒有掃到資料，請稍後再試或改自訂示範池。')
    else:
        n = len(topdf) if show_n == '全部' else int(show_n)
        top_show = topdf.head(n).copy()
        st.dataframe(top_show, use_container_width=True, hide_index=True)
        st.subheader('🚀 今日最可能發動')
        fast = topdf[topdf['預估發動時間'].isin(['🔥 今天～1天','🔥 1～3天','🚀 2～5天'])].head(10)
        st.dataframe(fast[['股票代號','股票名稱','爆發指數','預估發動時間','距離突破%','最佳買點','AI一句話']], use_container_width=True, hide_index=True)
        st.subheader('🏭 熱門族群')
        sec = topdf.groupby('產業')['爆發指數'].agg(['mean','count']).sort_values('mean', ascending=False).head(10).reset_index()
        st.dataframe(sec, use_container_width=True, hide_index=True)
        st.subheader('⚠️ AI 避雷')
        danger = topdf[(topdf['風險']=='高') | (topdf['RSI']>80)].head(10)
        st.dataframe(danger[['股票代號','股票名稱','RSI','量比','風險','最佳買點','AI一句話']], use_container_width=True, hide_index=True)

elif page == '全池AI掃描':
    st.subheader('📈 全市場 AI 掃描')
    with st.spinner('掃描中...'):
        df = scan_pool(pool['code'].tolist(), pool['name'].tolist(), pool['sector'].tolist(), scan_limit)
    if not df.empty:
        min_score = st.slider('最低爆發指數', 0, 100, 80)
        times = st.multiselect('預估發動時間', ['🔥 今天～1天','🔥 1～3天','🚀 2～5天','⭐ 5～10天','👀 觀察中'], default=['🔥 今天～1天','🔥 1～3天','🚀 2～5天','⭐ 5～10天'])
        view = df[(df['爆發指數']>=min_score) & (df['預估發動時間'].isin(times))]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button('下載掃描結果 CSV', view.to_csv(index=False).encode('utf-8-sig'), 'v30_scan_result.csv', 'text/csv')
    else:
        st.warning('沒有資料。')


elif page == '技術分析中心':
    st.subheader('📊 技術分析中心（完整版）')
    q = st.text_input('輸入股票代號或名稱', '7828', key='tech_q')
    match = pool[(pool['code'].astype(str)==q.strip()) | (pool['name'].astype(str).str.contains(q.strip(), na=False))]
    if len(match)>0:
        code = str(match.iloc[0]['code']); name=str(match.iloc[0]['name']); sector=str(match.iloc[0]['sector'])
    else:
        code=q.strip(); name=FALLBACK_POOL.get(code,''); sector=SECTOR_MAP.get(code,'其他')
    if st.button('開始完整技術分析', type='primary'):
        df = fetch_price(code, period='1y')
        if len(df)>35:
            ts = tech_summary(df)
            tags, near_break = detect_patterns(df)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric('技術分數', ts['技術分數'])
            c2.metric('均線狀態', ts['均線狀態'])
            c3.metric('MACD', ts['MACD狀態'])
            c4.metric('KD', ts['KD狀態'])
            c5.metric('距離突破', f'{near_break:.2f}%')
            st.info(ts['AI技術一句話'] + (' 型態：' + '、'.join(tags) if tags else ''))
            colA, colB = st.columns(2)
            with colA:
                st.markdown('#### 均線 / 支撐壓力')
                st.dataframe(pd.DataFrame([{k:ts[k] for k in ['收盤價','MA5','MA10','MA20','MA60','MA120','MA240','第一支撐','第二支撐','第一壓力','第二壓力']}]), use_container_width=True, hide_index=True)
            with colB:
                st.markdown('#### 指標')
                st.dataframe(pd.DataFrame([{k:ts[k] for k in ['RSI','K','D','DIF','DEA','MACD柱','布林上緣','布林中線','布林下緣','ATR','ADX','VWAP','ROC10%']}]), use_container_width=True, hide_index=True)
            st.plotly_chart(plot_k(df, f'{code} {name} 完整技術K線'), use_container_width=True)
        else:
            st.error('抓不到足夠資料。')

elif page == '個股搜尋':
    st.subheader('🔍 個股搜尋')
    q = st.text_input('輸入股票代號或名稱', '7828')
    match = pool[(pool['code'].astype(str)==q.strip()) | (pool['name'].astype(str).str.contains(q.strip(), na=False))]
    if len(match)>0:
        code = str(match.iloc[0]['code']); name=str(match.iloc[0]['name']); sector=str(match.iloc[0]['sector'])
    else:
        code=q.strip(); name=FALLBACK_POOL.get(code,''); sector=SECTOR_MAP.get(code,'其他')
    if st.button('開始分析', type='primary'):
        res = analyze_stock(code,name,sector)
        if res:
            df = res.pop('_df')
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric('爆發指數', res['爆發指數'])
            c2.metric('預估發動時間', res['預估發動時間'])
            c3.metric('主力建倉率', f"{res['主力建倉率']}%")
            c4.metric('停損', res['停損'])
            c5.metric('第一目標', res['第一目標'])
            st.info(res['AI一句話'])
            st.dataframe(pd.DataFrame([res]), use_container_width=True, hide_index=True)
            st.plotly_chart(plot_k(df, f'{code} {name} K線'), use_container_width=True)
        else:
            st.error('抓不到資料。')

elif page == '三大法人AI中心':
    st.subheader('📊 三大法人 AI 中心')
    st.write('可手動貼上券商/證交所匯出的法人資料；欄位建議：股票代號、股票名稱、外資、投信、自營商。')
    uploaded = st.file_uploader('上傳 CSV', type=['csv'])
    if uploaded:
        f = pd.read_csv(uploaded)
        need = ['外資','投信','自營商']
        for col in need:
            if col not in f.columns: f[col]=0
        f['三大法人合計'] = f['外資'] + f['投信'] + f['自營商']
        f['法人共振分數'] = np.clip((f[['外資','投信','自營商']]>0).sum(axis=1)*30 + (f['三大法人合計']>0)*10, 0, 100)
        f['AI法人解讀'] = np.where(f['法人共振分數']>=90,'三方同步偏多', np.where(f['法人共振分數']>=60,'法人偏多','法人分歧或偏弱'))
        st.dataframe(f.sort_values('法人共振分數', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info('未上傳法人CSV時，系統會在掃描表中用量價估算「三大法人共振」。要精準法人，請貼每日法人資料。')

elif page == '歷史回測':
    st.subheader('📈 AI 歷史回測')
    code = st.text_input('回測股票代號', '7828')
    days = st.selectbox('觀察天數', [5,10,20], index=1)
    if st.button('開始簡易回測'):
        df = fetch_price(code, period='1y')
        if len(df)>80:
            rows=[]
            for i in range(60, len(df)-days):
                sub = df.iloc[:i+1].copy()
                # 用簡化版訊號：MA多頭+RSI健康+MACD多頭
                c=sub['Close']; ma5=c.rolling(5).mean(); ma10=c.rolling(10).mean(); ma20=c.rolling(20).mean(); rv=rsi(c).iloc[-1]; dif,dea,h=macd(c)
                sig = ma5.iloc[-1]>ma10.iloc[-1]>ma20.iloc[-1] and 52<=rv<=76 and dif.iloc[-1]>dea.iloc[-1]
                if sig:
                    buy=float(df['Close'].iloc[i]); high=float(df['High'].iloc[i+1:i+1+days].max()); low=float(df['Low'].iloc[i+1:i+1+days].min())
                    rows.append({'推薦日':str(df.index[i].date()),'買入參考':round(buy,2),f'{days}日最高漲幅%':round((high/buy-1)*100,2),'期間最大回撤%':round((low/buy-1)*100,2),'結果':'成功' if high/buy-1>=0.05 else '觀察'})
            bt=pd.DataFrame(rows)
            if len(bt):
                st.metric('訊號次數', len(bt)); st.metric('5%以上成功率', f"{(bt['結果'].eq('成功').mean()*100):.1f}%")
                st.dataframe(bt.tail(50), use_container_width=True, hide_index=True)
            else: st.warning('這段期間沒有訊號。')
        else: st.error('資料不足。')

else:
    st.subheader('設定說明')
    st.markdown('''
### 已保留/加入
- ✅ 發動日、預估發動時間、發動倒數
- ✅ 全池模式與自訂示範池
- ✅ 股票名稱不消失
- ✅ 首頁 TOP20，不再一次塞滿 1900 檔
- ✅ 大盤分析
- ✅ 三大法人 AI 中心
- ✅ 完整技術分析：MA、RSI、KD、MACD、布林、ATR、ADX、OBV、VWAP、ROC
- ✅ 歷史回測
- ✅ 7828 信仰股固定首頁

### 小提醒
免費資料源速度與完整度有限。全池掃描建議先掃 300～800 檔；部署到雲端時可提高掃描檔數。
''')
