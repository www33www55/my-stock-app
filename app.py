import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 網頁基本設定
st.set_page_config(layout="wide", page_title="未來小股神｜勝率計算器", page_icon="📈")
st.title("🔮 未來小股神｜MACD + KD 高勝率策略回測 App")
st.markdown("本工具融合了 **MACD 零軸上金叉 + KD 金叉 + 月線多頭 + 量增** 四大強勢指標，幫你一鍵驗證歷史勝率！")

# ==================== 側邊欄參數設定 ====================
st.sidebar.header("⚙️ 策略參數設定")
stock = st.sidebar.text_input("輸入股票代號", "2330.TW", help="台股請加 .TW (如 2330.TW)，美股直接輸入代號 (如 AAPL)")
period = st.sidebar.selectbox("回測時間長度", ["1y", "2y", "3y", "5y", "6mo"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 出場條件")
target_profit = st.sidebar.number_input("停利目標 (%)", value=5.0, step=0.5)
stop_loss = st.sidebar.number_input("停損限制 (%)", value=3.0, step=0.5)
hold_days = st.sidebar.slider("最大持有觀察天數", min_value=3, max_value=60, value=10)

# ==================== 主要計算邏輯 ====================
if st.sidebar.button("🚀 開始回測計算", use_container_width=True):
    with st.spinner("正在下載即時數據並計算中..."):
        df = yf.download(stock, period=period)

    if df.empty:
        st.error("❌ 抓不到資料，請確認股票代號是否輸入正確（例如台股需補上 .TW）。")
        st.stop()
        
    # 打平 yfinance 的多層 Index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.dropna()

    # 確保資料為一維 float 格式
    close_ser = df["Close"].squeeze().astype(float)
    high_ser = df["High"].squeeze().astype(float)
    low_ser = df["Low"].squeeze().astype(float)
    vol_ser = df["Volume"].squeeze().astype(float)

    # ===== 技術指標計算 =====
    # MACD
    df["EMA12"] = close_ser.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close_ser.ewm(span=26, adjust=False).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["DIF"].ewm(span=9, adjust=False).mean()
    
    # KD
    low_9 = low_ser.rolling(window=9).min()
    high_9 = high_ser.rolling(window=9).max()
    df["RSV"] = (close_ser - low_9) / (high_9 - low_9) * 100
    df["RSV"] = df["RSV"].fillna(0)
    df["K"] = df["RSV"].ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()

    # 均線與量
    df["MA20"] = close_ser.rolling(20).mean()
    df["VOL20"] = vol_ser.rolling(20).mean()

    # ===== 訊號判定 =====
    df["MACD_GOLD"] = (df["DIF"].shift(1) < df["MACD_SIGNAL"].shift(1)) & (df["DIF"] > df["MACD_SIGNAL"])
    df["MACD_ZERO_GOLD"] = df["MACD_GOLD"] & (df["DIF"] > 0)
    df["KD_GOLD"] = (df["K"].shift(1) < df["D"].shift(1)) & (df["K"] > df["D"])
    df["ABOVE_MA20"] = close_ser > df["MA20"]
    df["VOLUME_UP"] = vol_ser > df["VOL20"]

    # 綜合買進訊號
    df["BUY_SIGNAL"] = df["MACD_ZERO_GOLD"] & df["KD_GOLD"] & df["ABOVE_MA20"] & df["VOLUME_UP"]

    # ===== 逐筆回測 =====
    results = []
    for i in range(len(df) - int(hold_days)):
        if df["BUY_SIGNAL"].iloc[i]:
            buy_price = float(close_ser.iloc[i])
            buy_date = df.index[i]
            future = df.iloc[i+1 : i+1+int(hold_days)]

            final_status = "未達標(到期)"
            p_profit, p_loss = 0.0, 0.0
            
            # 逐日偵測看先撞到哪一個
            for idx, row in future.iterrows():
                day_high = float(row["High"])
                day_low = float(row["Low"])
                
                p_profit = (day_high - buy_price) / buy_price * 100
                p_loss = (day_low - buy_price) / buy_price * 100
                
                if p_loss <= -stop_loss:
                    final_status = "失敗"
                    break
                if p_profit >= target_profit:
                    final_status = "成功"
                    break

            # 記錄這段期間的最高與最低
            future_high = float(future["High"].max())
            future_low = float(future["Low"].min())
            max_profit = (future_high - buy_price) / buy_price * 100
            max_loss = (future_low - buy_price) / buy_price * 100

            results.append({
                "日期": buy_date.date() if hasattr(buy_date, 'date') else buy_date,
                "買進價": round(buy_price, 2),
                "區間最高漲幅%": round(max_profit, 2),
                "區間最大跌幅%": round(max_loss, 2),
                "回測結果": final_status
            })

    result_df = pd.DataFrame(results)

    # ==================== 畫面呈現 ====================
    if result_df.empty:
        st.warning("⚠️ 這段期間沒有出現符合條件的買進訊號。提示：此策略極為嚴格，建議調整左側『回測時間長度』為 3y 或 5y 試試看！")
    else:
        total = len(result_df)
        win = len(result_df[result_df["回測結果"] == "成功"])
        loss = len(result_df[result_df["回測結果"] == "失敗"])
        win_rate = win / total * 100
        
        avg_profit = result_df["區間最高漲幅%"].mean()
        avg_loss = result_df["區間最大跌幅%"].mean()

        # 1. 數據大看板
        st.subheader("📊 策略回測績效")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("訊號總次數", f"{total} 次")
        c2.metric("策略勝率", f"{win_rate:.2f}%")
        c3.metric("平均最高回報", f"{avg_profit:.2f}%")
        c4.metric("平均最大風險", f"{avg_loss:.2f}%")

        # 2. 互動式 K 線圖表與買點標記
        st.subheader("📈 歷史買點與 K 線視覺化")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_width=[0.3, 0.7])
        
        # 主 K 線圖
        fig.add_trace(gr.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="K線", label_to_display="Close"
        ), row=1, col=1)
        
        # 加上 20MA
        fig.add_trace(gr.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20MA', line=dict(color='orange', width=1.5)), row=1, col=1)
        
        # 標註買進訊號
        signal_dates = df[df["BUY_SIGNAL"]].index
        signal_prices = df[df["BUY_SIGNAL"]]["Close"]
        fig.add_trace(gr.Scatter(
            x=signal_dates, y=signal_prices, mode='markers', name='★買進訊號',
            marker=dict(color='magenta', size=12, symbol='triangle-up', line=dict(color='black', width=1))
        ), row=1, col=1)
        
        # 副圖：MACD
        fig.add_trace(gr.Scatter(x=df.index, y=df['DIF'], mode='lines', name='DIF', line=dict(color='blue', width=1)), row=2, col=1)
        fig.add_trace(gr.Scatter(x=df.index, y=df['MACD_SIGNAL'], mode='lines', name='MACD_Signal', line=dict(color='red', width=1)), row=2, col=1)
        
        fig.update_layout(height=550, xaxis_rangeslider_visible=False, title_text=f"{stock} 訊號點位對照圖")
        st.plotly_chart(fig, use_container_width=True)

        # 3. 詳細表格
        st.subheader("📋 詳細交易紀錄")
        
        def color_result(val):
            if val == "成功": return "background-color: #d4edda; color: #155724;"
            elif val == "失敗": return "background-color: #f8d7da; color: #721c24;"
            return "background-color: #fff3cd; color: #856404;"
            
        st.dataframe(result_df.style.applymap(color_result, subset=["回測結果"]), use_container_width=True)

else:
    st.info("💡 請在左側設定好您想回測的股票代號與停損利條件，並點擊『開始回測計算』！")
