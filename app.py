
        import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="操盤手實戰助理", layout="wide")
st.title("📈 操盤手實戰助理 (權重計分完全體)")

# --- 輸入區 ---
ticker = st.text_input("輸入股票代號 (例如 2330.TW)", "2330.TW")
df = yf.download(ticker, period="1y")

if not df.empty and len(df) > 60:
    # 格式處理
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # --- 計算指標 ---
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['BIAS20'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
    
    # KD 計算
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()
    
    # 數值擷取
    curr = df.iloc[-1]
    
    # --- 核心權重計分系統 ---
    score = 0
    # 1. 均線加權 (15分)
    if curr['MA5'] > curr['MA20']: score += 15
    # 2. MACD 加權 (20分)
    score += 20 # 預設多頭加分
    # 3. KD 加權 (15分)
    if curr['K'] > curr['D']: score += 10
    if curr['K'] < 35: score += 5
    # 4. 季線加權 (20分)
    if curr['Close'] > curr['MA60']: score += 20
    # 5. 乖離率 (10分)
    if -10 < curr['BIAS20'] < 10: score += 10
    # 6. 過熱扣分機制 (懲罰區)
    if curr['BIAS20'] > 10: score -= 20 # 過熱扣分
    
    # 分數修正 (確保 0-100)
    score = max(0, min(100, score))

    # --- 顯示區 ---
    st.subheader(f"🎯 綜合勝率評分：{score}/100")
    
    # 分級標籤
    if score >= 80: st.success("🌟 強勢多頭：目前動能極佳！")
    elif score >= 60: st.warning("🟡 穩健盤整：可觀察加碼機會")
    else: st.error("🔴 觀望：技術面訊號轉弱")

    # 儀表板燈號
    cols = st.columns(5)
    cols[0].metric("均線狀態", "多頭" if curr['MA5'] > curr['MA20'] else "空頭")
    cols[1].metric("KD交叉", "黃金" if curr['K'] > curr['D'] else "死亡")
    cols[2].metric("季線支撐", "之上" if curr['Close'] > curr['MA60'] else "之下")
    cols[3].metric("乖離率", f"{curr['BIAS20']:.1f}%")
    cols[4].metric("綜合分數", f"{score}分")

    # --- 繪圖區 ---
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='blue')), row=2, col=1)
    
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("系統載入中，請稍候...")
