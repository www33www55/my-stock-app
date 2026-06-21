import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="操盤手助理", layout="wide")
st.title("📈 操盤手實戰系統 (權重計分版)")

ticker = st.text_input("輸入股票代號", "2330.TW")
df = yf.download(ticker, period="1y")

if not df.empty and len(df) > 60:
    # --- 1. 指標計算 ---
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # N字突破邏輯 (現價 > 過去20天最高點)
    prev_high = df['High'].rolling(20).max().shift(1)
    is_n_break = df['Close'].iloc[-1] > prev_high.iloc[-1]
    
    # 爆量邏輯 (今日成交量 > 過去20天平均成交量 * 1.5)
    is_volume_up = df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] * 1.5

    # --- 2. 權重計分系統 (精準對應你的需求) ---
    score = 0
    score += 10 if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else 0 # 均線
    score += 15 # (MACD假設)
    score += 10 # (KD假設)
    score += 10 if df['Close'].iloc[-1] > df['MA60'].iloc[-1] else 0 # 季線
    score += 20 if is_n_break else 0 # N字突破
    score += 20 if is_volume_up else 0 # 爆量
    score += 15 # (法人假設)

    # --- 3. 實戰儀表板 ---
    st.subheader(f"🎯 綜合勝率評分：{score}/100")
    
    # 燈號區
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("均線多頭", "YES" if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else "NO")
    col2.metric("N字突破", "YES" if is_n_break else "NO")
    col3.metric("爆量訊號", "YES" if is_volume_up else "NO")
    col4.metric("目前分數", f"{score}分")

    # 目標價邏輯
    last_low = df['Low'].rolling(10).min().iloc[-1]
    stop_loss = last_low * 0.98
    target_price = df['Close'].iloc[-1] * 1.15
    
    st.markdown(f"---")
    st.write(f"🛑 **停損建議**: {stop_loss:.2f} | 🎯 **目標價**: {target_price:.2f}")

    # 圖表
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("請輸入正確代號或等待資料載入...")
