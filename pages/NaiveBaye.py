import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# ส่วนหน้า UI
st.title("การจำแนกความเสี่ยงด้านสุขภาพด้วย Naive Bayes")
st.markdown("กรุณาป้อนข้อมูลเพื่อพยากรณ์ระดับความเสี่ยง")

# โหลดข้อมูล
try:
    df = pd.read_csv("data/Health_Risk_Dataset_Encoded.csv")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Health_Risk_Dataset_Encoded.csv' โปรดตรวจสอบ path ของไฟล์")
    st.stop()

# กำจัดแถวที่มีค่าว่าง
df.dropna(inplace=True)

# กำหนด Features (X) และ Target (y)
X = df.drop(columns=['Risk_Level_Num', 'Patient_ID'])
y = df['Risk_Level_Num']

# แบ่งข้อมูลเป็นชุดฝึก (Training set) และชุดทดสอบ (Testing set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# แสดงความแม่นยำของโมเดล
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"ความแม่นยำของโมเดล: {accuracy * 100:.2f}%")

# UI สำหรับรับข้อมูลจากผู้ใช้
st.subheader("ป้อนข้อมูลผู้ป่วยเพื่อพยากรณ์ความเสี่ยง")

# สร้าง dictionary สำหรับเก็บค่าที่ผู้ใช้ป้อน
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"ค่าของ {col}", value=0.0)

# ปุ่มพยากรณ์
if st.button("พยากรณ์"):
    # แปลงข้อมูลที่ผู้ใช้ป้อนให้เป็น DataFrame
    x_input = pd.DataFrame([input_data])
    
    # พยากรณ์ผลลัพธ์
    y_predict = clf.predict(x_input)
    
    # แสดงผลการพยากรณ์
    st.subheader("ผลการพยากรณ์:")
    if y_predict[0] == 0:
        st.success("ระดับความเสี่ยง: **Low Risk (ความเสี่ยงต่ำ)**")
    elif y_predict[0] == 1:
        st.warning("ระดับความเสี่ยง: **Moderate Risk (ความเสี่ยงปานกลาง)**")
    else:
        st.error("ระดับความเสี่ยง: **High Risk (ความเสี่ยงสูง)**")
