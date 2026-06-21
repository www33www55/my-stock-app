
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="全能技術分析健檢儀表板", layout="wide")
st.title("🔍 全能技術分析健檢儀表板 (終極安全高勝率版)")

# --- 1. 網頁正中央輸入框 (手機友善，免拉抽屜) ---
st.markdown("### 📥 請在下方輸入你想健檢的股票代號")
col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    ticker = st.text_input("股票代號 (上市加 .TW；上櫃加 .TWO；美股直接打代號，例如: NVDA)", "2330.TW")
with col_input2:
    period = st.selectbox("觀看範圍", ["6mo", "1y", "2y"], index=1)

st.markdown("---") 

@st.cache_data
def load_data(symbol, p):
    df = yf.download(symbol, period=p)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    df = load_data(ticker, period)
    
    if not df.empty:
        # --- 2. 核心計算：五大技術指標 ---
        # (1) 均線與月線乖離率
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['BIAS20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        
        # (2) MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['MACD_Signal'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['DIF'] - df['MACD_Signal']
        df['MACD_Super_Good'] = df['DIF'] > 0  # 漂亮定義：0軸之上波段多頭
        df['MACD_Bullish'] = df['DIF'] > df['MACD_Signal']
        
        # (3) KD (9, 3, 3)
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
        df['KD_Super_Good'] = df['K'] < 35  # 漂亮定義：35以下低檔起漲點

        # (4) RSI (5日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
        rs = gain / loss
        df['RSI5'] = 100 - (100 / (1 + rs))
        
        gain10 = (delta.where(delta > 0, 0)).rolling(window=10).mean()
        loss10 = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
        rs10 = gain10 / loss10
        df['RSI10'] = 100 - (100 / (1 + rs10))

        # --- 3. 交叉買點訊號定義 ---
        df['HIGH_WIN_SIGNAL'] = df['MACD_Bullish'] & df['KD_Cross']
        df['SUPER_WIN_SIGNAL'] = df['MACD_Super_Good'] & df['KD_Cross'] & df['KD_Super_Good']

        # --- 4. 健檢邏輯與防追高扣分機制 ---
        latest = df.iloc[-1]
        score = 0
        
        # 預防追高警報觸發門檻 (RSI超過78 或 偏離月線超過12%)
        is_overheated = (latest['RSI5'] > 78) or (latest['BIAS20'] > 12)
        
        st.subheader("📋 今日技術指標大健檢表")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if latest['MA5'] > latest['MA20'] and latest['MA20'] > latest['MA60']:
                st.success("🟢 均線：多頭排列 (20分)")
                score += 20
            elif latest['Close'] > latest['MA20']:
                st.warning("🟡 均線：月線盤整 (10分)")
                score += 10
            else:
                st.error("🔴 均線：空頭跌勢 (0分)")
                
        with col2:
            if latest['MACD_Bullish']:
                st.success("🟢 MACD：多頭波段 (20分)")
                score += 20
            else:
                st.error("🔴 MACD：空頭修正 (0分)")
                
        with col3:
            if latest['K'] > latest['D']:
                if latest['K'] > 80:
                    st.warning("🟡 KD：高檔鈍化強勢 (20分)")
                elif latest['K'] < 35:
                    st.success("🔥 KD：低檔超完美 (20分)")
                else:
                    st.success("🟢 KD：黃金交叉 (20分)")
                score += 20
            else:
                st.error("🔴 KD：死亡交叉 (0分)")
                
        with col4:
            if latest['Close'] > latest['MA60']:
                st.success("🟢 季線：生命線之上 (20分)")
                score += 20
            else:
                st.error("🔴 季線：跌破生命線 (0分)")
                
        with col5:
            if latest['BIAS20'] > 12:
                st.error(f"🚨 乖離：離月線過遠({latest['BIAS20']:.1f}%)")
            elif latest['BIAS20'] < -10:
                st.success(f"🟢 乖離：跌深負乖離 (20分)")
                score += 20
            else:
                st.success(f"🟢 乖離：安全範圍 (20分)")
                score += 20

        # 顯示綜合評分與即時警報雷達
        if is_overheated:
            score = min(score, 40)  # 強制壓低分數，高檔過熱不給高分
            st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
            st.markdown(f"### 🚦 今日即時訊號：⚠️⚠️ **警告！目前 RSI ({latest['RSI5']:.1f}) 或月線乖離率過高，處於極度危險高檔區，請絕對不要盲目追高！** ⚠️⚠️")
        else:
            st.markdown(f"### 🎯 技術面綜合多頭評分：` {score} / 100 ` 分")
            if latest['SUPER_WIN_SIGNAL']:
                st.markdown("### 🚦 今日即時訊號：🌟🌟 **神級超漂亮買點！(MACD大於0 + KD低檔黃金交叉) 抓到夢幻低檔起漲點！** 🌟🌟")
            elif latest['HIGH_WIN_SIGNAL']:
                st.markdown("### 🚦 今日即時訊號：🔥 **雙重確認！今天觸發普通高勝率買進訊號。**")
            elif latest['DIF'] > 0:
                st.markdown("### 🚦 今日即時訊號：🍏 **多頭趨勢良好，目前在等KD拉回低檔，請續抱耐性等待。**")
            else:
                st.markdown("### 🚦 今日即時訊號：❌ **目前處於弱勢調整期，不符合任何買點。**")

        # --- 5. 繪製專業多層綜合圖表 (K線、均線、KD、MACD、RSI) ---
        st.subheader("📊 完整技術圖表")
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.04, 
                            row_width=[0.15, 0.15, 0.15, 0.55])
        
        # Row 1: K線 + 均線 + 訊號星星
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='orange', width=1), name='5MA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='magenta', width=1.5), name='20MA(月線)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=2), name='60MA(季線)'), row=1, col=1)
        
        # 標示普通買點與神級買點
        normal_signals = df[df['HIGH_WIN_SIGNAL'] & ~df['SUPER_WIN_SIGNAL']]
        fig.add_trace(go.Scatter(x=normal_signals.index, y=normal_signals['Low'] * 0.97, mode='markers', marker=dict(symbol='triangle-up', size=12, color='gold'), name='🔥高勝率買點'), row=1, col=1)
        super_signals = df[df['SUPER_WIN_SIGNAL']]
        fig.add_trace(go.Scatter(x=super_signals.index, y=super_signals['Low'] * 0.96, mode='markers', marker=dict(symbol='star', size=15, color='cyan'), name='🌟神級超漂亮買點'), row=1, col=1)
        
        # Row 2: KD 指標
        fig.add_trace(go.Scatter(x=df.index, y=df['K'], line=dict(color='red', width=1.5), name='K值'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['D'], line=dict(color='blue', width=1.5), name='D值'), row=2, col=1)
        
        # Row 3: MACD 柱狀圖
        colors = ['red' if val >= 0 else 'green' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD柱狀圖'), row=3, col=1)

        # Row 4: RSI 5日
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI5'], line=dict(color='orange', width=1.5), name='RSI 5日'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI10'], line=dict(color='gray', width=1), name='RSI 10日'), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=750, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. 歷史紀錄表格 ---
        st.subheader("📋 歷史【神級超漂亮買點】清單 (對照上方藍色星星)")
        if not super_signals.empty:
            show_df = super_signals[['Close', 'K', 'D', 'RSI5', 'BIAS20']].copy()
            show_df.index = show_df.index.strftime('%Y-%m-%d')
            st.dataframe(show_df, use_container_width=True)
        else:
            st.info("在目前的觀看範圍內，這檔股票尚未出現符合「MACD大於0 + KD低於35黃金交叉」的夢幻神級買點。")

    else:
        st.error("找不到該股票資料，請檢查代號是否輸入正確。")
except Exception as e:
    st.error(f"發生錯誤: {e}")

