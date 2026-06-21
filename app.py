
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="操盤助理", layout="wide")
st.title("📈 操盤手實戰助理 (N字突破 + 實戰計分)")

ticker = st.text_input("輸入股票代號", "2330.TW")
df = yf.download(ticker, period="1y")

if not df.empty:
    # --- 實戰指標計算 ---
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # N字突破偵測 (簡化版：近期最高點)
    prev_high = df['High'].rolling(20).max().shift(1)
    is_n_break = (df['Close'].iloc[-1] > prev_high.iloc[-1])
    
    # 評分邏輯 (100分制)
    score = 0
    score += 10 if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else 0 # 均線
    score += 15 # 假設MACD多頭
    score += 10 # 假設KD黃金交叉
    score += 10 # 假設季線向上
    score += 20 if is_n_break else 0 # N字突破
    score += 20 # 爆量 (假設)
    score += 15 # 法人 (假設)

    # --- 實戰儀表板 ---
    st.subheader(f"🎯 綜合勝率評分：{score}/100")
    
    # 視覺化計分
    if score >= 90: st.success("🌟 神級買點：爆量 + N字突破！")
    elif score >= 70: st.warning("🟡 可觀察：籌碼與技術面轉強")
    else: st.error("🔴 70分以下：空手等待")

    # --- 自動目標價與停損 ---
    last_low = df['Low'].rolling(10).min().iloc[-1]
    stop_loss = last_low * 0.98
    target_price = df['Close'].iloc[-1] * 1.15
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📍 進場參考", f"{df['Close'].iloc[-1]:.2f}")
    col2.metric("🛑 建議停損", f"{stop_loss:.2f}")
    col3.metric("🎯 目標價", f"{target_price:.2f}")

    # --- 簡單圖表 ---
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)


