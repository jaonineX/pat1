import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(page_title="การพยากรณ์ความเสี่ยงด้านสุขภาพ", page_icon="❤️")

# --- ส่วนหัวเรื่อง ---
st.title("🩺 การพยากรณ์ความเสี่ยงด้านสุขภาพด้วย Decision Tree")
st.markdown("---")
st.markdown("แอปพลิเคชันนี้ใช้โมเดล *Decision Tree* ในการวิเคราะห์ข้อมูลและทำนายระดับความเสี่ยงด้านสุขภาพ")

# ใช้ st.cache_resource เพื่อแคชโมเดลที่ผ่านการฝึกฝนแล้ว
@st.cache_resource
def train_model(X_train, y_train):
    with st.spinner('กำลังฝึกโมเดล Decision Tree...'):
        ModelDtree = DecisionTreeClassifier()
        dtree = ModelDtree.fit(X_train, y_train)
    st.success("✨ ฝึกโมเดลสำเร็จ!")
    return dtree

# --- ส่วนการโหลดข้อมูลจากผู้ใช้ ---
st.subheader("📁 อัปโหลดไฟล์ข้อมูล")
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.info("✅ โหลดข้อมูลจากไฟล์ที่อัปโหลดเรียบร้อยแล้ว")
        st.write("ตัวอย่าง 10 แถวแรกของชุดข้อมูล:")
        st.dataframe(df.head(10))

        # --- การเตรียมข้อมูล ---
        target_column = 'Risk_Level_Num'
        features = [
            'Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale',
            'Systolic_BP', 'Heart_Rate', 'Temperature',
            'Consciousness', 'On_Oxygen'
        ]
        
        missing_cols = [col for col in features + [target_column] if col not in df.columns]
        if missing_cols:
            st.error(f"❌ *เกิดข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็น:* {', '.join(missing_cols)}")
        else:
            X = df[features]
            y = df[target_column]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

            # --- การฝึกโมเดล ---
            dtree = train_model(x_train, y_train)

            # --- ส่วนรับข้อมูลจากผู้ใช้ ---
            st.subheader("✍️ ป้อนข้อมูลเพื่อพยากรณ์")
            st.markdown("---")

            col1, col2 = st.columns(2)
            
            feature_labels = {
                'Respiratory_Rate': 'อัตราการหายใจ',
                'Oxygen_Saturation': 'ความอิ่มตัวของออกซิเจน',
                'O2_Scale': 'ระดับ O2 (1-2)',
                'Systolic_BP': 'ความดันโลหิตซิสโตลิก',
                'Heart_Rate': 'อัตราการเต้นของหัวใจ',
                'Temperature': 'อุณหภูมิ',
                'Consciousness': 'ระดับความรู้สึกตัว',
                'On_Oxygen': 'ได้รับออกซิเจน'
            }

            user_input = {}
            with col1:
                user_input['Respiratory_Rate'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Respiratory_Rate"]}', min_value=0, value=20, step=1)
                user_input['Oxygen_Saturation'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Oxygen_Saturation"]}', min_value=0, value=95, step=1)
                user_input['O2_Scale'] = st.selectbox(f'ป้อนค่าสำหรับ: {feature_labels["O2_Scale"]}', options=[1, 2], index=0)
                user_input['Systolic_BP'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Systolic_BP"]}', min_value=0, value=120, step=1)
            
            with col2:
                user_input['Heart_Rate'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Heart_Rate"]}', min_value=0, value=80, step=1)
                user_input['Temperature'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Temperature"]}', min_value=0.0, value=37.0, step=0.1)
                selected_consciousness = st.radio(f'ป้อนค่าสำหรับ: {feature_labels["Consciousness"]}', options=['ตื่นตัว (A)', 'บกพร่อง (P)'])
                user_input['Consciousness'] = 1 if selected_consciousness == 'บกพร่อง (P)' else 0
                selected_on_oxygen = st.radio(f'ป้อนค่าสำหรับ: {feature_labels["On_Oxygen"]}', options=['ไม่ได้รับ', 'ได้รับ'])
                user_input['On_Oxygen'] = 1 if selected_on_oxygen == 'ได้รับ' else 0

            if st.button("พยากรณ์ผล", type="primary"):
                x_input = [[user_input[feature] for feature in features]]
                y_predict2 = dtree.predict(x_input)
                
                st.write("---")
                st.subheader("### 💡 ผลการพยากรณ์:")
                if y_predict2[0] == 2:
                    st.error("⚠️ *มีความเสี่ยงสูง*")
                elif y_predict2[0] == 1:
                    st.warning("🟠 *มีความเสี่ยงต่ำ*")
                else:
                    st.success("🟢 *ไม่มีความเสี่ยง*")

            # --- ประสิทธิภาพของโมเดลและการแสดงภาพ ---
            st.markdown("---")
            st.subheader("📈 ประสิทธิภาพของโมเดล")
            y_predict = dtree.predict(x_test)
            score = accuracy_score(y_test, y_predict)
            
            st.metric(label="ความแม่นยำของโมเดล (Accuracy Score)", value=f"{int(score * 100)} %")

            st.subheader("🌳 แผนผัง Decision Tree")
            fig, ax = plt.subplots(figsize=(20, 15))
            tree.plot_tree(dtree, feature_names=features, class_names=['No Risk', 'Low Risk', 'High Risk'], ax=ax, filled=True, rounded=True, fontsize=10)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ *เกิดข้อผิดพลาดในการประมวลผลไฟล์*: {e}")
else:
    st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้น")