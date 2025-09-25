from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="การทำนายความเสี่ยงสุขภาพ",
    page_icon="❤️",
    layout="wide"
)

# --- Header Section ---
st.title("❤️ การทำนายความเสี่ยงสุขภาพด้วย K-Nearest Neighbor")
st.markdown("""
แอปพลิเคชันนี้ใช้โมเดล *K-Nearest Neighbor (KNN)* เพื่อทำนายความเสี่ยงด้านสุขภาพจากข้อมูลของคุณ
""")
st.markdown("---")

# --- Data Information Section ---
st.subheader("📊 ข้อมูลที่ใช้ในการฝึกโมเดล")
st.info("โปรดตรวจสอบข้อมูลเพื่อทำความเข้าใจลักษณะของชุดข้อมูล")
try:
    dt = pd.read_csv("Health_Risk_Dataset_Encoded.csv")
    st.write("แสดงข้อมูล 10 แถวแรก:")
    st.dataframe(dt.head(10))
except FileNotFoundError:
    st.error("❌ *ไม่พบไฟล์ 'Health_Risk_Dataset_Encoded.csv'* กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับโค้ด")
    st.stop()

# Clean data by dropping any rows with missing values
dt.dropna(inplace=True)

# --- Visualization Section ---
st.markdown("---")
st.subheader("📈 การสำรวจข้อมูล (Data Exploration)")
st.info("ใช้กราฟด้านล่างเพื่อดูการกระจายของข้อมูลแต่ละฟีเจอร์")

# Select only numerical features and the target for visualization
numerical_features = dt.select_dtypes(include=np.number).columns.tolist()
# Exclude Patient_ID and the target variable from the list of selectable features
features_for_plot = [col for col in numerical_features if col not in ['Patient_ID', 'Risk_Level_Num']]

feature_to_plot = st.selectbox(
    "เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล", 
    features_for_plot,
    help="เลือกฟีเจอร์เพื่อดูว่าค่าของฟีเจอร์นั้นๆ มีการกระจายตัวอย่างไรเมื่อเทียบกับผลลัพธ์ (Risk_Level_Num)"
)

# Boxplot
st.write(f"#### 🎯 Boxplot: แสดงการกระจายของฟีเจอร์ '{feature_to_plot}'")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=dt, x='Risk_Level_Num', y=feature_to_plot, ax=ax)
ax.set_title(f'Boxplot ของ {feature_to_plot} เทียบกับ Risk Level', fontsize=16)
ax.set_xlabel('Risk Level (0: ต่ำ, 1: ปานกลาง, 2: สูง)', fontsize=12)
ax.set_ylabel(feature_to_plot, fontsize=12)
st.pyplot(fig)

# --- Prediction Section ---
st.markdown("---")
st.header("🔮 ทำนายผลความเสี่ยงสุขภาพ")
st.info("กรุณาป้อนข้อมูลของคุณในช่องด้านล่างเพื่อรับการทำนาย")

# Dictionary for mapping English column names to Thai labels
feature_labels = {
    'Respiratory_Rate': 'อัตราการหายใจ (ครั้ง/นาที)', 
    'Oxygen_Saturation': 'ความอิ่มตัวของออกซิเจน (%)', 
    'O2_Scale': 'ระดับออกซิเจน',
    'Systolic_BP': 'ความดันโลหิตตัวบน (mmHg)', 
    'Heart_Rate': 'อัตราการเต้นของหัวใจ (ครั้ง/นาที)',
    'Temperature': 'อุณหภูมิร่างกาย (เซลเซียส)',
    'Consciousness': 'ระดับการรับรู้', 
    'On_Oxygen': 'ใช้เครื่องช่วยหายใจ (ออกซิเจน)',
}

# Define the features for prediction, excluding Patient_ID and the target
input_features = ['Respiratory_Rate', 'Oxygen_Saturation', 'O2_Scale', 'Systolic_BP', 'Heart_Rate', 'Temperature', 'Consciousness', 'On_Oxygen']

# Create input fields using columns for better layout
user_input = {}
cols = st.columns(2)

for i, feature in enumerate(input_features):
    with cols[i % 2]:
        label_text = feature_labels.get(feature, feature)
        if feature == 'Consciousness':
            consciousness_options = {'A': 'A', 'P': 'P'}
            selected_consciousness = st.selectbox(
                f'กรุณาเลือกระดับ: *{label_text}*',
                options=list(consciousness_options.keys()),
                key=f"input_{feature}"
            )
            user_input[feature] = 1 if selected_consciousness == 'P' else 0
        elif feature == 'On_Oxygen':
            oxygen_options = {'ใช่': 1, 'ไม่ใช่': 0}
            selected_oxygen = st.selectbox(
                f'กรุณาเลือก: *{label_text}*',
                options=list(oxygen_options.keys()),
                key=f"input_{feature}"
            )
            user_input[feature] = oxygen_options[selected_oxygen]
        else:
            value = st.number_input(
                f'กรุณาป้อนค่าสำหรับ: *{label_text}*', 
                key=f"input_{feature}",
                step=1
            )
            user_input[feature] = value

# Prediction button and result display
st.markdown("---")
if st.button("🌟 ทำนายผล", type="primary"):
    # Define features and target for training
    X = dt.drop(columns=['Patient_ID', 'Risk_Level_Num'])
    y = dt['Risk_Level_Num']

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)
    
    # Ensure the input array has the same feature order as the training data
    x_input = np.array([[user_input[feature] for feature in X.columns]])

    prediction = Knn_model.predict(x_input)
    st.subheader("✅ ผลการทำนาย:")
    
    risk_level_map = {0: 'ความเสี่ยงต่ำ', 1: 'ความเสี่ยงปานกลาง', 2: 'ความเสี่ยงสูง'}
    predicted_risk = risk_level_map.get(prediction[0], 'ไม่สามารถระบุได้')

    if prediction[0] == 2:
        st.error(f'⚠️ *คุณมีความเสี่ยงในระดับ: {predicted_risk}*')
        st.markdown("ขอแนะนำให้ปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อยืนยันผลและรับคำแนะนำที่ถูกต้อง")
    elif prediction[0] == 1:
        st.warning(f'🟡 *คุณมีความเสี่ยงในระดับ: {predicted_risk}*')
        st.markdown("ควรเฝ้าระวังและดูแลสุขภาพอย่างใกล้ชิด")
    else:
        st.success(f'🟢 *คุณมีความเสี่ยงในระดับ: {predicted_risk}*')
        st.markdown("อย่างไรก็ตาม การดูแลสุขภาพอย่างสม่ำเสมอเป็นสิ่งสำคัญ")