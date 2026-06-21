import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="高勝率雙指標選股機", layout="wide")
st.title("🚦 KD + MACD 雙重多頭高勝率選股機")

# 側邊欄輸入
st.sidebar.header("設定參數")
ticker = st.sidebar.text_input("輸入股票代號 (例如: 2330.TW 或 AAPL)", "2330.TW")
period = st.sidebar.selectbox("資料範圍", ["1y", "2y", "5y"], index=0)

@st.cache_data
def load_data(symbol, p):
    df = yf.download(symbol, period=p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    df = load_data(ticker, period)
    
    if not df.empty:
        # --- 1. 計算 MACD 多頭趨勢 ---
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_Bullish'] = df['DIF'] > df['MACD_Signal']  # 多頭趨勢定義：DIF > Signal

        # --- 2. 計算 KD 黃金交叉 ---
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
        
        # KD 黃金交叉：今天 K>D 且 昨天 K<=D
        df['KD_Cross'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))

        # --- 3. 核心：高勝率買進訊號 (MACD是多頭波段 + 今天剛好KD黃金交叉) ---
        df['HIGH_WIN_SIGNAL'] = df['MACD_Bullish'] & df['KD_Cross']

        # --- 4. 判斷今天最新狀態 ---
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        st.subheader("📢 今日即時雷達")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if latest['MACD_Bullish']:
                st.success("📈 MACD 趨勢：波段多頭安全區")
            else:
                st.error("📉 MACD 趨勢：空頭修正防守區")
                
        with col2:
            if latest['K'] > latest['D']:
                st.warning(f"⚡ KD 狀態：K值({latest['K']:.1f}) > D值({latest['D']:.1f})")
            else:
                st.info(f"❄️ KD 狀態：K值({latest['K']:.1f}) < D值({latest['D']:.1f})")
                
        with col3:
            if latest['HIGH_WIN_SIGNAL']:
                st.markdown("### 🚦 訊號：🔥 **雙重確認！強烈進場訊號**")
            elif latest['MACD_Bullish'] and not latest['KD_Cross']:
                st.markdown("### 🚦 訊號：🍏 **多頭持股續抱，等待低吸**")
            else:
                st.markdown("### 🚦 訊號：❌ **目前不符合高勝率買點**")

        # --- 5. 繪製精簡 K 線圖 ---
        st.subheader("📊 歷史高勝率訊號點驗證")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"))
        
        # 標示出歷史上雙重符合的亮點
        signals = df[df['HIGH_WIN_SIGNAL']]
        fig.add_trace(go.Scatter(x=signals.index, y=signals['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='🔥高勝率買點'))
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. 歷史紀錄表格 ---
        st.subheader("📋 歷史進場點清單 (對照上方綠色三角形)")
        show_df = signals[['Close', 'K', 'D', 'DIF', 'MACD_Signal']].copy()
        show_df.index = show_df.index.strftime('%Y-%m-%d')
        st.dataframe(show_df, use_container_width=True)

    else:
        st.error("找不到該股票資料，請檢查代號是否輸入正確。")
except Exception as e:
    st.error(f"發生錯誤: {e}")
