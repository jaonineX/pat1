import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# --- Set Page Config ---
st.set_page_config(
    page_title="การพยากรณ์ความเสี่ยงด้านสุขภาพด้วย Naive Bayes",
    page_icon="❤️",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        color: #333;
    }
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1c7y2n2 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-16ajdlt {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        color: #1890ff;
    }
    h1 {
        color: #e83e8c;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2, h3, h4 {
        color: #004085;
    }
    .stButton>button {
        background-color: #1890ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #40a9ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("🩺 การพยากรณ์ความเสี่ยงด้านสุขภาพด้วย Naive Bayes")
st.markdown("---")
st.markdown("แอปพลิเคชันนี้ใช้โมเดล *Naive Bayes* ในการวิเคราะห์ข้อมูลและทำนายระดับความเสี่ยงด้านสุขภาพ")

# --- Load Data Section ---
st.subheader("📁 อัปโหลดไฟล์ข้อมูล")
uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.info("✅ โหลดข้อมูลจากไฟล์ที่อัปโหลดเรียบร้อยแล้ว")
        st.write("ตัวอย่าง 10 แถวแรกของชุดข้อมูล:")
        st.dataframe(df.head(10))

        # --- Data Cleaning and Preparation ---
        # แปลงคอลัมน์ 'Consciousness' จาก 'A' และ 'P' เป็นตัวเลข 0 และ 1
        df['Consciousness'] = df['Consciousness'].map({'A': 0, 'P': 1})
        
        # ลบแถวที่มีค่าว่าง (NaN) ในคอลัมน์เป้าหมาย
        target_column = 'Risk_Level_Num'
        df.dropna(subset=[target_column], inplace=True)
        df[target_column] = df[target_column].astype(int)

        # --- Model Training ---
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner('กำลังฝึกโมเดล Naive Bayes...'):
                clf = GaussianNB()
                clf.fit(X_train, y_train)
            st.success("✨ ฝึกโมเดลสำเร็จ!")

            # --- Prediction Section ---
            st.subheader("🔮 กรุณาป้อนข้อมูลเพื่อพยากรณ์")
            st.info("กรุณาป้อนข้อมูลของคุณในช่องด้านล่างเพื่อรับการทำนาย")
            
            # สร้าง dictionary สำหรับแปลงชื่อคอลัมน์เป็นภาษาไทย
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
            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    label_text = feature_labels.get(feature, feature)
                    if feature == 'Consciousness':
                        selected_consciousness = st.radio(
                            f'ป้อนค่าสำหรับ: {label_text}', 
                            options=['ตื่นตัว (A)', 'บกพร่อง (P)'],
                            key=f"input_{feature}"
                        )
                        user_input[feature] = 1 if selected_consciousness == 'บกพร่อง (P)' else 0
                    elif feature == 'On_Oxygen':
                        selected_on_oxygen = st.radio(
                            f'ป้อนค่าสำหรับ: {label_text}', 
                            options=['ไม่ได้รับ', 'ได้รับ'],
                            key=f"input_{feature}"
                        )
                        user_input[feature] = 1 if selected_on_oxygen == 'ได้รับ' else 0
                    else:
                        user_input[feature] = st.number_input(
                            f'กรุณาป้อนค่าสำหรับ: *{label_text}*', 
                            value=0, 
                            step=1,
                            key=f"input_{feature}"
                        )

            # Prediction button and result display
            st.markdown("---")
            if st.button("🌟 พยากรณ์", type="primary"):
                x_input = np.array([[user_input[feature] for feature in features]])
                y_predict = clf.predict(x_input)
                st.subheader("✅ ผลการพยากรณ์:")
                
                if y_predict[0] == 2:
                    st.error("⚠️ *มีความเสี่ยงสูง*")
                elif y_predict[0] == 1:
                    st.warning("🟠 *มีความเสี่ยงต่ำ*")
                else:
                    st.success("🟢 *ไม่มีความเสี่ยง*")

            # --- Model Performance ---
            st.markdown("---")
            st.subheader("📈 ประสิทธิภาพของโมเดล")
            y_pred_test = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred_test)
            st.metric(label="ความแม่นยำของโมเดล (Accuracy Score)", value=f"{int(score * 100)} %")

    except Exception as e:
        st.error(f"❌ *เกิดข้อผิดพลาดในการประมวลผลไฟล์*: {e}")
else:
    st.info("⬆️ กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้น")