from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_engine import DEFAULT_POOL, fetch_stock, analyze_df, scan_pool, fetch_market, stock_name

st.set_page_config(page_title='未來小股神 AI 操盤中心 V32', layout='wide', page_icon='🚀')

st.markdown('''
<style>
.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1100px;}
.big-title{font-size:46px; font-weight:900; line-height:1.15;}
.card{border:1px solid rgba(255,255,255,.15); border-radius:18px; padding:18px; margin:10px 0; background:rgba(255,255,255,.04)}
.good{color:#ff4b4b;font-weight:800}.ok{color:#f5c542;font-weight:800}.muted{opacity:.72}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="big-title">🚀 未來小股神 AI<br/>操盤中心 V32 Complete</div>', unsafe_allow_html=True)
st.caption('重點：完整版能動、不 empty、單檔抓不到不會整個掛掉；資料源失效時自動 Demo 備援。')

# Sidebar
with st.sidebar:
    st.header('⚙️ 功能選單')
    page = st.radio('選擇功能', ['首頁戰情中心','單股 AI 掃描','全池 AI 掃描','大盤分析'], index=0)
    st.divider()
    st.write('❤️ 信仰股：7828 創新服務')
    st.write('資料來源：Yahoo → Demo備援')

def metric_card(title, value, sub=''):
    st.markdown(f'<div class="card"><div class="muted">{title}</div><div style="font-size:30px;font-weight:900">{value}</div><div>{sub}</div></div>', unsafe_allow_html=True)

def show_stock_card(code: str):
    df, src = fetch_stock(code, allow_demo=True)
    ana = analyze_df(df, code)
    name = stock_name(code)
    if not ana.get('ok'):
        st.warning(f'{code} {name} 暫時抓不到資料：資料不足')
        return
    st.markdown(f'### ❤️ {code} {name or ""} 信仰股')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('現價', f"{ana['close']:.2f}")
    c2.metric('爆發指數', ana['score'])
    c3.metric('預估發動', ana['launch'])
    c4.metric('資料源', src)
    st.success(ana['summary'])
    st.write(f"停損：**{ana['stop']}**｜目標：**{ana['target1']} / {ana['target2']} / {ana['target3']}**｜壓力：**{ana['pressure1']}**")

def plot_k(df: pd.DataFrame, title='K線'):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
    for n in [5,10,20]:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(n).mean(), mode='lines', name=f'MA{n}'))
    fig.update_layout(height=420, title=title, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

if page == '首頁戰情中心':
    show_stock_card('7828')
    st.markdown('## 📊 AI 大盤分析')
    market = fetch_market()
    st.dataframe(market, use_container_width=True, hide_index=True)

    st.markdown('## 🔥 今日 AI TOP20')
    if st.button('快速產生 TOP20', type='primary'):
        with st.spinner('AI 掃描中，抓不到的股票會自動用備援資料，不會中斷...'):
            top = scan_pool(DEFAULT_POOL, limit=20)
        st.session_state['top20'] = top
    if 'top20' not in st.session_state:
        st.info('按「快速產生 TOP20」開始。')
    else:
        st.dataframe(st.session_state['top20'], use_container_width=True, hide_index=True)

elif page == '單股 AI 掃描':
    st.markdown('## 🔍 單股 AI 掃描')
    code = st.text_input('輸入股票代號', value='7828')
    if st.button('開始分析', type='primary'):
        df, src = fetch_stock(code, allow_demo=True)
        ana = analyze_df(df, code)
        name = stock_name(code)
        if not ana.get('ok'):
            st.error('資料不足')
        else:
            st.subheader(f'{code} {name}｜資料源：{src}')
            c1,c2,c3,c4 = st.columns(4)
            c1.metric('現價', f"{ana['close']:.2f}")
            c2.metric('爆發指數', ana['score'])
            c3.metric('預估發動時間', ana['launch'])
            c4.metric('RSI', f"{ana['rsi']:.1f}")
            st.success(ana['summary'])
            st.markdown('### 📈 K線與均線')
            plot_k(df, f'{code} {name}')
            st.markdown('### 📊 技術分析')
            tech = pd.DataFrame([{
                'MA5': round(ana['ma5'],2), 'MA10': round(ana['ma10'],2), 'MA20': round(ana['ma20'],2),
                'RSI': round(ana['rsi'],1), 'K': round(ana['k'],1), 'D': round(ana['d'],1),
                'MACD狀態': ana['macd_state'], '量比': round(ana['vol_ratio'],2), '距離突破%': round(ana['dist_break'],2)
            }])
            st.dataframe(tech, use_container_width=True, hide_index=True)
            st.markdown('### 🎯 操作參考')
            st.write(f"支撐：**{ana['support1']}**｜壓力：**{ana['pressure1']}**｜停損：**{ana['stop']}**")
            st.write(f"目標1：**{ana['target1']}**｜目標2：**{ana['target2']}**｜目標3：**{ana['target3']}**")

elif page == '全池 AI 掃描':
    st.markdown('## 🌍 全池 AI 掃描')
    pool_text = st.text_area('股票池，可自行增減，用逗號分隔', value=', '.join(DEFAULT_POOL), height=120)
    limit = st.slider('顯示前幾名', 10, 100, 20)
    if st.button('開始掃描', type='primary'):
        codes = [x.strip() for x in pool_text.replace('\n', ',').split(',') if x.strip()]
        with st.spinner('掃描中...'):
            res = scan_pool(codes, limit=limit)
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button('下載 CSV', res.to_csv(index=False).encode('utf-8-sig'), 'ai_top.csv', 'text/csv')

elif page == '大盤分析':
    st.markdown('## 📊 AI 大盤分析')
    st.dataframe(fetch_market(), use_container_width=True, hide_index=True)
    st.info('如果 Yahoo 暫時抓不到，系統會用 Demo 備援，畫面仍能正常運作。')

st.divider()
st.caption('V32 Complete：先讓資料穩、畫面不空、功能能跑；之後可再接 TWSE/TPEX/FinMind 正式資料源。')
