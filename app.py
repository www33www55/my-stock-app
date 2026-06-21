import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="操盤手系統", layout="wide")
st.title("📈 操盤手實戰系統 (除錯重構版)")

ticker = st.text_input("輸入股票代號 (例如: 2330.TW)", "2330.TW")
df = yf.download(ticker, period="1y")

# --- 安全檢查：確保資料載入成功 ---
if not df.empty and len(df) > 60:
    # 確保 columns 是單層結構
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. 計算均線
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 2. 獲取當前數據
    curr_close = float(df['Close'].iloc[-1])
    curr_ma20 = float(df['MA20'].iloc[-1])
    curr_ma60 = float(df['MA60'].iloc[-1])
    
    # 3. N字突破判定 (現價 > 過去20天最高)
    prev_high = float(df['High'].rolling(window=20).max().shift(1).iloc[-1])
    is_n_break = curr_close > prev_high
    
    # 4. 爆量判定
    curr_vol = float(df['Volume'].iloc[-1])
    avg_vol = float(df['Volume'].rolling(window=20).mean().iloc[-1])
    is_volume_up = curr_vol > (avg_vol * 1.5)

    # 5. 計分系統 (權重設定)
    score = 0
    if curr_close > curr_ma20: score += 10    # 均線
    score += 15                              # MACD (保留權重)
    score += 10                              # KD (保留權重)
    if curr_close > curr_ma60: score += 10    # 季線
    if is_n_break: score += 20               # N字突破
    if is_volume_up: score += 20             # 爆量
    score += 15                              # 法人 (保留權重)

    # --- 顯示區 ---
    st.subheader(f"🎯 綜合勝率評分：{score}/100")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("均線多頭", "YES" if curr_close > curr_ma20 else "NO")
    c2.metric("N字突破", "YES" if is_n_break else "NO")
    c3.metric("爆量訊號", "YES" if is_volume_up else "NO")
    c4.metric("目前總分", f"{score}分")

    # 目標價邏輯
    last_low = float(df['Low'].rolling(window=10).min().iloc[-1])
    st.markdown(f"---")
    st.write(f"🛑 **停損建議**: {last_low * 0.98:.2f} | 🎯 **目標價**: {curr_close * 1.15:.2f}")

    # 圖表
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("目前抓不到資料，請檢查代號是否正確，或稍候再試。")
