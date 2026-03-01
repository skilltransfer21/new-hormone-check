import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 画面の初期設定 ---
st.set_page_config(page_title="ホルモン変動チェック", layout="centered")

# --- モデルの読み込み ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('hormone_risk_model.pkl')
    except FileNotFoundError:
        st.error("モデルファイルが見つかりません。先に train_and_save.py を実行してください。")
        return None

model = load_model()

# --- UI構築 ---
st.title("🌿 毎日の女性ホルモン変動チェック (Mockup)")
st.write("本日の身体の数値を入力して、ホルモンバランスのハイリスク状態を判定します。")
st.markdown("---")

# ユーザー入力フォーム
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("年齢", min_value=18, max_value=100, value=40)
    height = st.number_input("身長 (cm)", min_value=100.0, max_value=220.0, value=160.0)
    weight = st.number_input("今日の体重 (kg)", min_value=30.0, max_value=150.0, value=55.0)
    
    # BMIは自動計算
    bmi = weight / ((height / 100) ** 2)
    st.info(f"計算されたBMI: **{bmi:.1f}**")

with col2:
    waist = st.number_input("ウエスト周囲径 (cm)", min_value=50.0, max_value=150.0, value=80.0)
    menstruation = st.selectbox("月経の有無", options=[1.0, 2.0], format_func=lambda x: "あり" if x == 1.0 else "なし")
    
    # 閉経年齢は月経なしの場合に入力させる
    menopause_age = np.nan
    if menstruation == 2.0:
        menopause_age = st.number_input("閉経年齢", min_value=30.0, max_value=70.0, value=50.0)

st.markdown("### 生活習慣")
col3, col4 = st.columns(2)
with col3:
    smoke = st.selectbox("喫煙状況", options=[1.0, 2.0, 3.0], format_func=lambda x: {1.0:"毎日吸う", 2.0:"時々吸う", 3.0:"吸わない"}[x])
with col4:
    alcohol = st.slider("飲酒頻度カテゴリ (小:高頻度 〜 大:低頻度)", min_value=1.0, max_value=10.0, value=5.0)

# --- 判定処理 ---
st.markdown("---")
if st.button("状態を判定する", type="primary"):
    if model is not None:
        # 入力データをDataFrameにまとめる (学習時と同じカラム名・順番にする)
        input_data = pd.DataFrame([[
            age, bmi, weight, height, waist, 
            menstruation, menopause_age, smoke, alcohol
        ]], columns=[
            'RIDAGEYR', 'BMXBMI', 'BMXWT', 'BMXHT', 'BMXWAIST', 
            'RHQ031', 'RHQ060', 'SMQ040', 'ALQ121'
        ])
        
        # 欠損値（NaN）の処理（モックアップ用簡易対応）
        input_data.fillna(-1, inplace=True)
        
        # 予測の実行
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # ハイリスクになる確率
        
        st.markdown("### 📊 判定結果")
        if prediction == 1:
            st.error(f"**【ハイリスク状態】** (確率: {probability*100:.1f}%)")
            st.write("ホルモンバランスが大きく変動している、またはリスクレベルが高い状態と判定されました。休息をとり、必要に応じて専門医への相談を検討してください。")
        else:
            st.success(f"**【正常範囲内】** (ハイリスク確率: {probability*100:.1f}%)")
            st.write("現在のところ、大きな変動リスクは検出されていません。このままの生活習慣を維持しましょう。")