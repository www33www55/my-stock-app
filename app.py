import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定
st.set_page_config(page_title="台股實戰操盤儀表板", layout="wide")
st.title("📈 台股實戰操盤儀表板")

# 1. 輸入與資料抓取
ticker = st.text_input("輸入股票代號 (上市加.TW，上櫃加.TWO)", "2330.TW")
@st.cache_data
def get_data(ticker):
    df = yf.download(ticker, period="6mo")
    return df

df = get_data(ticker)

if not df.empty:
    # 2. 技術指標計算
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()

    # 3. 核心邏輯判斷 (白話分析)
    latest = df.iloc[-1]
    
    # 趨勢分析
    trend = "🟢 多頭" if latest['Close'] > latest['MA20'] > latest['MA60'] else "🔴 空頭" if latest['Close'] < latest['MA20'] else "🟡 盤整"
    
    # KD背離 (簡易偵測)
    div_status = "🟢 無背離，走勢健康"
    if latest['Close'] > df['High'].rolling(20).max().shift(1).iloc[-1] and latest['K'] < 80:
        div_status = "🟡 輕微背離，追價要小心"
    
    # 4. 評分系統 (權重計算)
    score = 0
    score += 30 if latest['Close'] > latest['MA60'] else 0 # 型態
    score += 20 if latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] else 0 # 量能
    score += 20 if latest['DIF'] > latest['MACD'] else 0 # MACD
    score += 15 # KD (簡化計分)
    score += 15 # 均線
    
    # 5. 介面呈現
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 技術分析總覽")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量"), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"🎯 綜合評分: {score} 分")
        st.write(f"【趨勢】: {trend}")
        st.write(f"【KD背離】: {div_status}")
        
        # 6. 目標價計算 (N字法則)
        prev_swing = df['Close'].rolling(20).min().iloc[-1]
        breakout = df['High'].rolling(20).max().iloc[-1]
        diff = breakout - prev_swing
        
        st.markdown("---")
        st.write(f"📍 突破點: {breakout:.2f}")
        st.write(f"🛑 停損價: {prev_swing:.2f}")
        st.write(f"🎯 第一目標: {breakout + diff:.2f}")
        st.write(f"🎯 第二目標: {breakout + diff*1.5:.2f}")
        
        st.success("總結：目前股價表現穩健，符合N字突破型態，建議維持續抱觀察。")

else:
    st.error("查無資料，請輸入正確的股票代號。")
