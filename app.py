import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="全能操盤儀表板", layout="wide")
st.title("🚀 全能操盤儀表板 (極致優化版)")

# --- 輸入框 ---
ticker = st.text_input("輸入股票代號 (如 2330.TW / AAPL)", "2330.TW")
period = st.selectbox("時間範圍", ["6mo", "1y", "2y"], index=1)

@st.cache_data
def load_data(symbol, p):
    try:
        df = yf.download(symbol, period=p)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

df = load_data(ticker, period)

if not df.empty and len(df) > 10:
    # --- 指標計算 ---
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BIAS20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
    
    # KD & 背離判定 (簡化版)
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    
    # 背離偵測
    df['Bear_Div'] = (df['Close'] > df['Close'].shift(1)) & (df['K'] < df['K'].shift(1)) & (df['K'] > 70)
    df['Bull_Div'] = (df['Close'] < df['Close'].shift(1)) & (df['K'] > df['K'].shift(1)) & (df['K'] < 30)

    # --- 視覺排版 ---
    latest = df.iloc[-1]
    
    # 燈號區
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("乖離率", f"{latest['BIAS20']:.1f}%")
    with c2: st.metric("K值", f"{latest['K']:.1f}")
    with c3: st.metric("D值", f"{latest['D']:.1f}")
    with c4: st.write("背離狀態", "🚨 高檔!" if latest['Bear_Div'] else "🔥 低檔!" if latest['Bull_Div'] else "✅ 正常")

    # --- 繪圖 ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
    
    # K線 + 成交量 (改為共用區塊更清晰)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    
    # KD
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='blue')), row=2, col=1)
    
    # 背離標示 (直接畫在圖上)
    fig.add_trace(go.Scatter(x=df[df['Bear_Div']].index, y=df[df['Bear_Div']]['Close'], mode='markers', name='高檔背離', marker=dict(size=10, color='red', symbol='x')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Bull_Div']].index, y=df[df['Bull_Div']]['Close'], mode='markers', name='低檔背離', marker=dict(size=10, color='green', symbol='circle')), row=1, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
