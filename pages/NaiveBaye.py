import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# ส่วนหน้า UI
st.title("การจำแนกประเภทดอกไอริสด้วย Naive Bayes")
st.markdown("กรุณาป้อนข้อมูลเพื่อพยากรณ์ชนิดของดอกไอริส")

# โหลดข้อมูล
try:
    df = pd.read_csv("data/iris.csv")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'iris.csv' โปรดตรวจสอบ path ของไฟล์")
    st.stop()

# กำหนด Features (X) และ Target (y)
X = df.drop('variety', axis=1)
y = df['variety']

# แบ่งข้อมูลเป็นชุดฝึก (Training set) และชุดทดสอบ (Testing set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes (Gaussian Naive Bayes เหมาะสำหรับข้อมูลต่อเนื่อง)
clf = GaussianNB()
clf.fit(X_train, y_train)

# แสดงความแม่นยำของโมเดล
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"ความแม่นยำของโมเดล: {accuracy * 100:.2f}%")

# UI สำหรับรับข้อมูลจากผู้ใช้
st.subheader("ป้อนขนาดของกลีบดอกและใบเลี้ยง")
spL = st.number_input('ความยาวกลีบเลี้ยง (cm)', value=5.1)
spW = st.number_input('ความกว้างกลีบเลี้ยง (cm)', value=3.5)
ptL = st.number_input('ความยาวกลีบดอก (cm)', value=1.4)
ptW = st.number_input('ความกว้างกลีบดอก (cm)', value=0.2)

# ปุ่มพยากรณ์
if st.button("พยากรณ์"):
    # แปลงข้อมูลจากผู้ใช้ให้อยู่ในรูปแบบที่โมเดลรับได้
    x_input = [[spL, spW, ptL, ptW]]
    
    # พยากรณ์ผลลัพธ์
    y_predict = clf.predict(x_input)
    
    # แสดงผลการพยากรณ์
    st.subheader("ผลการพยากรณ์:")
    st.success(f"ชนิดของดอกไอริสที่พยากรณ์ได้คือ: **{y_predict[0]}**")