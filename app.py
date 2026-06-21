
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="全能技術分析健檢儀表板", layout="wide")
st.title("🔍 全能技術分析健檢儀表板")

# --- 把輸入框直接搬到網頁正中央 ---
st.markdown("### 📥 請在下方輸入你想健檢的股票代號")
col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    ticker = st.text_input("股票代號 (台股請加 .TW,上櫃.TWO，例如: 2330.TW；美股直接輸入，例如: NVDA)", "2330.TW")
with col_input2:
    period = st.selectbox("觀看範圍", ["6mo", "1y", "2y"], index=1)

st.markdown("---") # 分隔線

@st.cache_data
def load_data(symbol, p):
    df = yf.download(symbol, period=p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    df = load_data(ticker, period)
    
    if not df.empty:
        # --- 1. 計算所有技術指標 ---
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']
        
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

        # --- 2. 今日最新狀態評分大健檢 ---
        latest = df.iloc[-1]
        
        st.subheader("📋 今日四大技術指標綜合健檢")
        col1, col2, col3, col4 = st.columns(4)
        
        score = 0
        
        with col1:
            if latest['MA5'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
                st.success("🟢 均線排列：多頭排列")
                score += 25
            elif latest['Close'] > latest['MA20']:
                st.warning("🟡 均線排列：站上月線盤整")
                score += 15
            else:
                st.error("🔴 均線排列：空頭排列防守")
                
        with col2:
            if latest['DIF'] > latest['MACD_Signal']:
                st.success("🟢 MACD：波段多頭波段")
                score += 25
            else:
                st.error("🔴 MACD：空頭修正波段")
                
        with col3:
            if latest['K'] > latest['D']:
                if latest['K'] > 80:
                    st.warning("🟡 KD：高檔鈍化強勢")
                else:
                    st.success("🟢 KD：黃金交叉多頭")
                score += 25
            else:
                st.error("🔴 KD：死亡交叉弱勢")
                
        with col4:
            if latest['Close'] > latest['MA60']:
                st.success("🟢 季線守護：站穩生命線之上")
                score += 25
            else:
                st.error("🔴 季線守護：跌破生命線")

        # 顯示綜合評分
        st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
        if score >= 75:
            st.markdown("💡 **診斷結論：趨勢極度強勢，順風局！符合高勝率進場/續抱條件。**")
        elif score >= 50:
            st.markdown("💡 **診斷結論：多空拉鋸盤整中，建議分批加碼或觀望。**")
        else:
            st.markdown("💡 **診斷結論：空頭趨勢明顯，目前風險極高，不建議入場。**")

        # --- 3. 繪製專業綜合圖表 ---
        st.subheader("📊 完整技術圖表")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            row_width=[0.2, 0.2, 0.6])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='5MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='magenta', width=1.5), name='20MA(月線)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=2), name='60MA(季線)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='red', width=1.5), name='K值'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='blue', width=1.5), name='D值'), row=2, col=1)
        
        colors = ['red' if val >= 0 else 'green' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD柱狀圖'), row=3, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("找不到該股票資料，請檢查代號是否輸入正確。")
except Exception as e:
    st.error(f"發生錯誤: {e}")
