import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="全能技術分析健檢儀表板", layout="wide")
st.title("🔍 全能技術分析健檢儀表板 (KD背離面板直視版)")

# --- 1. 網頁正中央輸入框 ---
st.markdown("### 📥 請在下方輸入你想健檢的股票代號")
col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    ticker = st.text_input("股票代號 (上市加 .TW；上櫃加 .TWO；美股直接打代號)", "2330.TW")
with col_input2:
    period = st.selectbox("觀看範圍", ["6mo", "1y", "2y"], index=1)

st.markdown("---") 

@st.cache_data
def load_data(symbol, p):
    try:
        df = yf.download(symbol, period=p)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

df = load_data(ticker, period)

if df is not None and not df.empty and len(df) > 10:
    # --- 2. 核心計算：技術指標 ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['BIAS20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']
    df['MACD_Super_Good'] = df['DIF'] > 0  
    df['MACD_Bullish'] = df['DIF'] > df['MACD_Signal']
    
    # KD
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    k_list, d_list = [], []
    current_k, current_d = 50.0, 50.0
    for r in rsv.fillna(50):
        current_k = (2/3) * current_k + (1/3) * r
        current_d = (2/3) * current_d + (1/3) * current_k
        k_list.append(current_k)
        d_list.append(current_d)
    df['K'] = k_list
    df['D'] = d_list
    df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    df['KD_Super_Good'] = df['K'] < 35  

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    df['RSI5'] = 100 - (100 / (1 + rs))

    # --- 3. 自動偵測：KD 背離邏輯 ---
    df['Is_Peak'] = (df['K'] > df['K'].shift(1)) & (df['K'] > df['K'].shift(-1)) & (df['K'] > 60)
    df['Is_Trough'] = (df['K'] < df['K'].shift(1)) & (df['K'] < df['K'].shift(-1)) & (df['K'] < 40)
    
    kd_bear_div = False 
    kd_bull_div = False 
    
    peaks = df[df['Is_Peak']]
    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        if p2['Close'] > p1['Close'] and p2['K'] < p1['K']:
            kd_bear_div = True
            
    troughs = df[df['Is_Trough']]
    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        if t2['Close'] < t1['Close'] and t2['K'] > t1['K']:
            kd_bull_div = True

    df['HIGH_WIN_SIGNAL'] = df['MACD_Bullish'] & df['KD_Cross']
    df['SUPER_WIN_SIGNAL'] = df['MACD_Super_Good'] & df['KD_Cross'] & df['KD_Super_Good']

    # --- 4. 今日最新狀態評分大健檢 ---
    latest = df.iloc[-1]
    score = 0
    is_overheated = (latest['RSI5'] > 78) or (latest['BIAS20'] > 12) or kd_bear_div
    
    st.subheader("📋 今日技術指標大健檢表")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if latest['MA5'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
            st.success("🟢 均線：多頭排列")
            score += 20
        elif latest['Close'] > latest['MA20']:
            st.warning("🟡 均線：月線盤整")
            score += 10
        else:
            st.error("🔴 均線：空頭跌勢")
            
    with col2:
        if latest['MACD_Bullish']:
            st.success("🟢 MACD：多頭波段")
            score += 20
        else:
            st.error("🔴 MACD：空頭修正")
            
    with col3:
        if latest['K'] > latest['D']:
            st.success("🟢 KD：黃金交叉")
            score += 20
        else:
            st.error("🔴 KD：死亡交叉")
            
    with col4:
        if latest['Close'] > latest['MA60']:
            st.success("🟢 季線：生命線之上")
            score += 20
        else:
            st.error("🔴 季線：跌破生命線")
            
    with col5:
        # 把乖離 % 數值完美補上文字顯示！
        if latest['BIAS20'] > 12:
            st.error(f"🚨 乖離：過高 ({latest['BIAS20']:.1f}%)")
        elif latest['BIAS20'] < -10:
            st.success(f"🟢 乖離：跌深 ({latest['BIAS20']:.1f}%)")
            score += 20
        else:
            st.success(f"🟢 乖離：安全 ({latest['BIAS20']:.1f}%)")
            score += 20

    with col6:
        if kd_bear_div:
            st.error("🚨 背離：高檔背離(危險)")
        elif kd_bull_div:
            st.success("🔥 背離：低檔背離(好買點)")
            score += 20
        else:
            st.success("🟢 背離：目前無背離")

    # 顯示綜合評分與即時警報雷達
    if kd_bear_div:
        score = min(score, 20)
        st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
        st.markdown(f"### 🚦 今日即時訊號：🚨🚨 **特大警報！偵測到「KD 高檔背離」！主力拉高出貨中，隨時暴跌！** 🚨🚨")
    elif is_overheated:
        score = min(score, 40)  
        st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
        st.markdown(f"### 🚦 今日即時訊號：⚠️⚠️ **警告！目前股價或 RSI ({latest['RSI5']:.1f}) 過熱，處於高檔追高危險區！** ⚠️⚠️")
    else:
        st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
        if kd_bull_div and latest['KD_Cross']:
            st.markdown("### 🚦 今日即時訊號：🌟🚀 **黃金訊號：偵測到「KD 低檔背離」！這是主力默默吃貨的絕對底部！** 🚀🌟")
        elif latest['SUPER_WIN_SIGNAL']:
            st.markdown("### 🚦 今日即時訊號：🌟🌟 **神級超漂亮買點！安全低檔起漲點！** 🌟🌟")
        elif latest['HIGH_WIN_SIGNAL']:
            st.markdown("### 🚦 今日即時訊號：🔥 **雙重確認！今天觸發普通高勝率買進訊號。**")
        else:
            st.markdown("### 🚦 今日即時訊號：🍏 **多頭趨勢良好，目前安全無背離，持股可續抱。**")

    # --- 5. 繪製專業圖表 ---
    st.subheader("📊 完整技術圖表")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.04, 
                        row_width=[0.15, 0.15, 0.15, 0.55])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='5MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='magenta', width=1.5), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=2), name='60MA'), row=1, col=1)
    
    normal_signals = df[df['HIGH_WIN_SIGNAL'] & ~df['SUPER_WIN_SIGNAL']]
    fig.add_trace(go.Scatter(x=normal_signals.index, y=normal_signals['Low'] * 0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='gold'), name='🔥高勝率買點'), row=1, col=1)
    super_signals = df[df['SUPER_WIN_SIGNAL']]
    fig.add_trace(go.Scatter(x=super_signals.index, y=super_signals['Low'] * 0.96, mode='markers', marker=dict(symbol='star', size=15, color='cyan'), name='🌟神級超漂亮買點'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='red', width=1.5), name='K值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='blue', width=1.5), name='D值'), row=2, col=1)
    
    colors = ['red' if val >= 0 else 'green' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD柱狀圖'), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI5'], line=dict(color='orange', width=1.5), name='RSI 5日'), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=750, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("請在上方輸入正確的股票代號（例如台股 2330.TW 或美股 AAPL）來開始進行全能技術分析。")


