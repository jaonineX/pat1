import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

st.title("ระบบพยากรณ์โรคหัวใจด้วยโมเดล KNN")

# แทนที่การโหลดรูปภาพจาก local file ด้วย URL
# 
st.image("https://images.unsplash.com/photo-1627916694602-0a187b92f41f?q=80&w=1740&auto=format&fit=crop", caption="A healthy heart", use_column_width=True)

# โหลดข้อมูล
try:
    df = pd.read_csv("data/Health_Risk_Dataset_Encoded.csv")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Health_Risk_Dataset_Encoded.csv' โปรดตรวจสอบ path")
    st.stop()

# ลบแถวที่มีค่าว่างออกเพื่อป้องกันข้อผิดพลาด
df.dropna(inplace=True)

# กำหนดคอลัมน์เป้าหมาย
# จากไฟล์ที่ให้มา คอลัมน์ที่เกี่ยวข้องกับความเสี่ยงคือ 'Risk_Level_Num'
target_col = 'Risk_Level_Num'

# แยก Features และ Target
X = df.drop(columns=[target_col, 'Patient_ID'])
y = df[target_col]

# แปลงข้อมูล categorical เป็น numeric
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# แบ่งข้อมูลสำหรับ Train และ Test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.subheader("พยากรณ์ความเสี่ยงโรคหัวใจ")

# ตั้งค่าพารามิเตอร์สำหรับโมเดล KNN
n_neighbors = st.slider('จำนวนเพื่อนบ้าน (K)', 1, 10, 3)

# สร้างและเทรนโมเดล KNN
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_model.fit(x_train, y_train)

# พยากรณ์ผลลัพธ์จากข้อมูลทดสอบ
y_pred = knn_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"ความแม่นยำของโมเดล: {accuracy * 100:.2f} %")

st.subheader("พยากรณ์จากข้อมูลใหม่")
st.write("กรุณาป้อนค่าในแต่ละคอลัมน์เพื่อพยากรณ์ความเสี่ยง")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"ค่าของ {col}", value=0.0)

if st.button('พยากรณ์'):
    # แปลงข้อมูลที่ป้อนให้เป็น DataFrame
    input_df = pd.DataFrame([input_data])
    
    # พยากรณ์ผลลัพธ์
    prediction = knn_model.predict(input_df)
    
    # แสดงผลลัพธ์
    st.write(f"ผลการพยากรณ์ความเสี่ยง: {prediction[0]}")
    st.info("ระดับความเสี่ยง: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk")

st.subheader("กราฟแสดงข้อมูลจริงและข้อมูลพยากรณ์")

# สร้าง scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

# ใช้เพียง 2 features แรกสำหรับ visualization
feature1 = X.columns[0]
feature2 = X.columns[1]

ax.scatter(x_test[feature1], x_test[feature2], c=y_test, cmap='viridis', label='ข้อมูลจริง')
ax.scatter(x_test[feature1], x_test[feature2], c=y_pred, cmap='plasma', marker='x', s=100, label='ข้อมูลที่พยากรณ์')

ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_title("การเปรียบเทียบข้อมูลจริงและข้อมูลพยากรณ์")
ax.legend()
st.pyplot(fig)
