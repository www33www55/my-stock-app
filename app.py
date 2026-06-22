 import streamlit as st

st.set_page_config(page_title="未來小股神", layout="wide")

st.title("🚀 未來小股神")
st.write("如果你看到這個畫面，代表 Streamlit 已經正常運作！")

stock = st.text_input("輸入股票代號", "2303.TW")

if st.button("分析"):
    st.success(f"成功分析 {stock}")
