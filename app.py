
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="操盤手系統", layout="wide")
st.title("📈 操盤手實戰系統 (修正版)")

ticker = st.text_input("輸入股票代號", "2330.TW")
df = yf.download(ticker, period="1y")

# --- 防呆機制：確保資料存在且結構單純 ---
if not df.empty and len(df) > 60:
    # 如果是多重索引，只取第一層
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 計算均線
    df['MA20'] = df['Close'].rolling(20).mean()
    
    # 【關鍵修正】：強制轉換型態，確保比較時是純數字
    curr_close = float(df['Close'].iloc[-1])
    curr_ma20 = float(df['MA20'].iloc[-1])
    
    # 評分邏輯 (使用轉換後的 float)
    score = 0
    if curr_close > curr_ma20: 
        score += 20
        trend_text = "🟢 多頭"
    else:
        trend_text = "🔴 空頭"

    # 顯示
    st.subheader(f"🎯 綜合勝率評分：{score}/100")
    st.write(f"【趨勢】: {trend_text}")

    # 繪圖
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("正在讀取資料或代號錯誤，請稍候...")
