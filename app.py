import io, time, math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title='未來小股神 AI 操盤中心 V34', layout='wide')

FALLBACK = pd.DataFrame([
    ('1101','台泥','上市'),('1102','亞泥','上市'),('1210','大成','上市'),('1301','台塑','上市'),('1303','南亞','上市'),
    ('1414','東和','上市'),('1515','力山','上市'),('1714','和桐','上市'),('1717','長興','上市'),('1718','中纖','上市'),
    ('1730','花仙子','上市'),('1731','美吾華','上市'),('1817','凱撒衛','上市'),('2013','中鋼構','上市'),('2303','聯電','上市'),
    ('2313','華通','上市'),('2330','台積電','上市'),('2344','華邦電','上市'),('2359','所羅門','上市'),('2382','廣達','上市'),
    ('2409','友達','上市'),('2527','宏璟','上市'),('2739','寒舍','上市'),('3037','欣興','上市'),('3060','銘異','上市'),
    ('3228','金麗科','上櫃'),('3479','安勤','上櫃'),('3557','逸昌','上櫃'),('3567','逸昌','上櫃'),('3588','通嘉','上市'),
    ('4183','福永生技','上櫃'),('4554','橙的','上櫃'),('5211','蒙恬','上櫃'),('5432','新門','上櫃'),('5468','凱鈺','上櫃'),
    ('5469','瀚宇博','上市'),('6129','普誠','上櫃'),('6139','亞翔','上市'),('6144','得利影','上櫃'),('6182','合晶','上櫃'),
    ('6191','精成科','上市'),('6246','臺龍','上櫃'),('6259','百徽','上櫃'),('6270','倍微','上櫃'),('6271','同欣電','上市'),
    ('6272','驊陞','上市'),('6582','申豐','上市'),('7828','創新服務','上櫃'),('8183','精星','上市'),('8284','三竹','上櫃'),
    ('8443','阿瘦','上市'),('8923','時報','上櫃')], columns=['代號','名稱','市場'])

@st.cache_data(ttl=60*60*12)
def load_twse():
    url='https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL'
    try:
        data=requests.get(url, timeout=12).json()
        rows=[]
        for x in data:
            code=str(x.get('Code','')).strip()
            name=str(x.get('Name','')).strip()
            if code.isdigit() and len(code)==4 and name:
                rows.append((code,name,'上市'))
        return pd.DataFrame(rows, columns=['代號','名稱','市場']).drop_duplicates('代號')
    except Exception:
        return pd.DataFrame(columns=['代號','名稱','市場'])

@st.cache_data(ttl=60*60*12)
def load_tpex():
    urls=[
        'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes',
        'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'
    ]
    for url in urls:
        try:
            data=requests.get(url, timeout=12).json()
            rows=[]
            for x in data:
                code=str(x.get('SecuritiesCompanyCode') or x.get('CompanyCode') or x.get('Code') or '').strip()
                name=str(x.get('CompanyName') or x.get('SecuritiesCompanyName') or x.get('Name') or '').strip()
                if code.isdigit() and len(code)==4 and name:
                    rows.append((code,name,'上櫃'))
            df=pd.DataFrame(rows, columns=['代號','名稱','市場']).drop_duplicates('代號')
            if len(df)>100: return df
        except Exception:
            pass
    return pd.DataFrame(columns=['代號','名稱','市場'])

@st.cache_data(ttl=60*60*12)
def stock_pool():
    twse=load_twse(); tpex=load_tpex()
    df=pd.concat([twse,tpex,FALLBACK], ignore_index=True).drop_duplicates('代號', keep='first')
    df=df[df['代號'].astype(str).str.fullmatch(r'\d{4}')].sort_values('代號').reset_index(drop=True)
    return df

@st.cache_data(ttl=60*10)
def get_price(code, market, period='8mo'):
    suffix='.TW' if market=='上市' else '.TWO'
    for sym in [f'{code}{suffix}', f'{code}.TW', f'{code}.TWO']:
        try:
            df=yf.download(sym, period=period, interval='1d', progress=False, auto_adjust=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns=df.columns.get_level_values(0)
            if df is not None and not df.empty and 'Close' in df and df['Close'].dropna().shape[0] > 35:
                return df.dropna(), sym
        except Exception:
            pass
    return pd.DataFrame(), ''

def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))

def indicators(df):
    c=df['Close']; h=df['High']; l=df['Low']; v=df['Volume']
    out={}
    for n in [5,10,20,60]: out[f'MA{n}']=float(c.rolling(n).mean().iloc[-1])
    out['close']=float(c.iloc[-1]); out['high20']=float(h.rolling(20).max().iloc[-1]); out['high60']=float(h.rolling(60).max().iloc[-1])
    out['low20']=float(l.rolling(20).min().iloc[-1]); out['RSI']=float(rsi(c).iloc[-1])
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean(); dif=ema12-ema26; macd=dif.ewm(span=9).mean()
    out['DIF']=float(dif.iloc[-1]); out['MACD']=float(macd.iloc[-1]); out['MACD_hist']=float((dif-macd).iloc[-1]); out['MACD_prev']=float((dif-macd).iloc[-2])
    low9=l.rolling(9).min(); high9=h.rolling(9).max(); k=100*(c-low9)/(high9-low9+1e-9); d=k.rolling(3).mean()
    out['K']=float(k.iloc[-1]); out['D']=float(d.iloc[-1]); out['vol_ratio']=float(v.iloc[-1]/(v.rolling(20).mean().iloc[-1]+1e-9))
    out['dist_high_pct']=float((out['high20']-out['close'])/(out['high20']+1e-9)*100)
    out['range20_pct']=float((out['high20']-out['low20'])/(out['close']+1e-9)*100)
    return out

def score_stock(code, name, market):
    df, sym=get_price(code, market)
    if df.empty:
        return {'代號':code,'名稱':name,'市場':market,'狀態':'資料不足','現價':np.nan,'爆發指數':0,'AI信心':'50%','發動時間':'觀察中','評等':'⭐','RSI':np.nan,'MACD':np.nan,'量比':np.nan,'AI解讀':'抓不到資料'}
    x=indicators(df); score=0; reasons=[]; risks=[]
    c=x['close']
    if x['MA5']>x['MA10']>x['MA20']: score+=20; reasons.append('均線多頭')
    elif c>x['MA20']: score+=8; reasons.append('站上月線')
    if x['RSI']>=50 and x['RSI']<=72: score+=12; reasons.append('RSI健康')
    elif x['RSI']<50: score+=4
    if x['DIF']>x['MACD']: score+=13; reasons.append('MACD偏多')
    if x['MACD_hist']>x['MACD_prev']: score+=7; reasons.append('MACD動能增')
    if x['K']>x['D']: score+=5; reasons.append('KD偏多')
    if x['vol_ratio']>1.3: score+=10; reasons.append('量能放大')
    if x['dist_high_pct']>=0 and x['dist_high_pct']<=3: score+=13; reasons.append('接近突破')
    elif x['dist_high_pct']<=6: score+=6; reasons.append('接近壓力')
    if x['range20_pct']<12 and c>x['MA20']: score+=10; reasons.append('平台整理')
    # 風險扣分
    if x['RSI']>82: score-=12; risks.append('RSI過熱')
    if c<x['MA20']: score-=10; risks.append('跌破月線')
    if c<x['MA5']: score-=6; risks.append('跌破5MA')
    if x['vol_ratio']>4 and df['Close'].iloc[-1] < df['Open'].iloc[-1]: score-=18; risks.append('爆量長黑')
    score=int(max(0,min(100,score)))
    conf=max(50,min(96, int(score*0.85+18)))
    if score>=90: launch='已發動'
    elif score>=80: launch='1~3天'
    elif score>=68: launch='2~5天'
    elif score>=55: launch='5~10天'
    else: launch='觀察中'
    if x['RSI']>82: launch='高風險'
    stars='⭐'*max(1,min(5,math.ceil(score/20)))
    text='、'.join(reasons[:4]) if reasons else '條件未成熟'
    if risks: text += '；風險：' + '、'.join(risks[:2])
    return {'代號':code,'名稱':name,'市場':market,'現價':round(c,2),'爆發指數':score,'AI信心':f'{conf}%','發動時間':launch,'評等':stars,'RSI':round(x['RSI'],1),'MACD':round(x['MACD_hist'],3),'量比':round(x['vol_ratio'],2),'AI解讀':text,'狀態':'OK'}

@st.cache_data(ttl=60*5)
def market_eval():
    rows=[]
    for label,sym in [('加權指數','^TWII'),('櫃買OTC','^TWOII')]:
        try:
            df=yf.download(sym, period='6mo', interval='1d', progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if df.empty or len(df)<30: raise ValueError('no data')
            x=indicators(df); s=50
            if x['close']>x['MA20']: s+=10
            if x['MA5']>x['MA10']: s+=8
            if x['DIF']>x['MACD']: s+=8
            if 45<=x['RSI']<=70: s+=6
            s=max(0,min(100,s)); mood='偏多震盪' if s>=70 else ('中性整理' if s>=55 else '偏弱保守')
            rows.append({'市場':label,'收盤':round(x['close'],2),'AI分數':s,'RSI':round(x['RSI'],1),'MACD':round(x['MACD_hist'],3),'AI解讀':mood})
        except Exception:
            rows.append({'市場':label,'收盤':'-','AI分數':'-','RSI':'-','MACD':'-','AI解讀':'資料暫無'})
    return pd.DataFrame(rows)

def scan(pool, max_n=0):
    if max_n and max_n>0: pool=pool.head(max_n)
    results=[]; total=len(pool); prog=st.progress(0); txt=st.empty()
    for i,row in pool.iterrows():
        txt.write(f'掃描中：{i+1}/{total}  {row.代號} {row.名稱}')
        try: results.append(score_stock(str(row.代號), str(row.名稱), str(row.市場)))
        except Exception as e:
            results.append({'代號':row.代號,'名稱':row.名稱,'市場':row.市場,'狀態':str(e),'爆發指數':0,'AI信心':'50%','發動時間':'觀察中','評等':'⭐'})
        prog.progress(min(1.0,(i+1)/max(total,1)))
    txt.empty(); prog.empty()
    df=pd.DataFrame(results)
    return df.sort_values(['爆發指數'], ascending=False).reset_index(drop=True)

st.title('🚀 未來小股神 AI 操盤中心 V34 Ultimate')
pool=stock_pool()
st.success(f'已載入股票池：{len(pool)} 檔（上市＋上櫃；若官方資料源失敗會自動備援）')

st.subheader('📊 AI 大盤技術評估')
st.dataframe(market_eval(), use_container_width=True, hide_index=True)

tabs=st.tabs(['🔥 全池TOP20','🔍 單股掃描','📋 股票池'])
with tabs[0]:
    st.subheader('🔥 今日 AI TOP20（全市場）')
    max_n=st.number_input('最多掃描檔數（0=全池；手機/免費雲建議先試 100）', min_value=0, max_value=3000, value=0, step=50)
    show_n=st.slider('顯示前幾名', 5, 100, 20)
    if st.button('開始全池掃描 / 更新TOP20', type='primary'):
        st.info('全池會逐檔下載，免費 Streamlit Cloud 可能需要較久。')
        res=scan(pool, int(max_n))
        st.session_state['last_scan']=res
    if 'last_scan' in st.session_state:
        res=st.session_state['last_scan']
        ok=res[res['狀態'].eq('OK')] if '狀態' in res else res
        st.write(f'掃描完成：有效 {len(ok)} 檔 / 股票池 {len(res)} 檔')
        cols=['代號','名稱','市場','現價','爆發指數','AI信心','發動時間','評等','RSI','MACD','量比','AI解讀']
        st.dataframe(res[cols].head(show_n), use_container_width=True, hide_index=True)
        st.download_button('下載本次掃描CSV', res.to_csv(index=False).encode('utf-8-sig'), 'v34_scan.csv')
with tabs[1]:
    st.subheader('🔍 單股 AI 掃描')
    q=st.text_input('輸入代號或名稱', value='7828')
    if st.button('分析單股'):
        match=pool[(pool['代號'].astype(str)==q.strip()) | (pool['名稱'].astype(str).str.contains(q.strip(), na=False))]
        if match.empty:
            st.error('股票池找不到這檔，請確認代號或名稱。')
        else:
            r=match.iloc[0]
            ans=score_stock(str(r.代號), str(r.名稱), str(r.市場))
            st.metric(f"{ans['代號']} {ans['名稱']} 爆發", ans['爆發指數'])
            c1,c2,c3,c4=st.columns(4)
            c1.metric('AI信心', ans['AI信心']); c2.metric('發動', ans['發動時間']); c3.metric('RSI', ans.get('RSI','-')); c4.metric('量比', ans.get('量比','-'))
            st.write('AI解讀：', ans.get('AI解讀',''))
            df,sym=get_price(str(r.代號), str(r.市場))
            if not df.empty:
                fig=go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
with tabs[2]:
    st.subheader('📋 股票池檢查')
    st.write(f'總檔數：{len(pool)}')
    st.dataframe(pool, use_container_width=True, hide_index=True)
