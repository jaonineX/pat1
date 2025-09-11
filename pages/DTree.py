import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

st.title("Naive Bayes Classification for Health Risk")

# โหลดข้อมูล
try:
    df = pd.read_csv("data/Health_Risk_Dataset_Encoded.csv")
    st.write("แสดงข้อมูลตัวอย่าง:")
    st.write(df.head(10))
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Health_Risk_Dataset_Encoded.csv' โปรดตรวจสอบ path")
    st.stop()

# ลบแถวที่มีค่าว่างออกเพื่อป้องกันข้อผิดพลาด
df.dropna(inplace=True)

# กำหนด target column
target_col = "Risk_Level_Num"

# แยก Features และ Target
# Patient_ID ถูกตัดออกเพราะเป็นเพียงรหัสผู้ป่วย
X = df.drop(columns=[target_col, "Patient_ID"])
y = df[target_col]

# แบ่งข้อมูลสำหรับ Train และ Test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างและเทรนโมเดล Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)

st.subheader("พยากรณ์ความเสี่ยงจากข้อมูลใหม่")

# UI สำหรับป้อนข้อมูลใหม่
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"ค่าของ {col}", value=0.0)

if st.button("พยากรณ์"):
    # แปลงข้อมูลที่ป้อนให้เป็น DataFrame
    input_df = pd.DataFrame([input_data])
    
    # พยากรณ์ผลลัพธ์
    prediction = model.predict(input_df)
    
    # แสดงผลลัพธ์
    st.write(f"ผลการพยากรณ์: {prediction[0]}")
    st.info("ระดับความเสี่ยง: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk")

# Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"ความแม่นยำของโมเดล: {accuracy * 100:.2f} %")
