from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรคหัวใจด้วยเทคนิค K-Nearest Neighbor')
#st.image("./img/kairung.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("")
   st.image("./img/heart1.jpg")

with col2:
   st.header("")
   st.image("./img/heart2.jpg")


html_7 = """
<div style="background-color:#33beff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

A1 = st.number_input("กรุณาเลือกข้อมูล A1")
A2 = st.number_input("กรุณาเลือกข้อมูล A2")
A3 = st.number_input("กรุณาเลือกข้อมูล A3")
A4 = st.number_input("กรุณาเลือกข้อมูล A4")
A5 = st.number_input("กรุณาเลือกข้อมูล A5")
A6 = st.number_input("กรุณาเลือกข้อมูล A6")
A7 = st.number_input("กรุณาเลือกข้อมูล A7")
A8 = st.number_input("กรุณาเลือกข้อมูล A8")
A9 = st.number_input("กรุณาเลือกข้อมูล A9")
A10 = st.number_input("กรุณาเลือกข้อมูล A10")
A11 = st.number_input("กรุณาเลือกข้อมูล A11")

if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris-3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 1:
    st.image("./img/heart1.jpg")
   else:
    st.image("./img/heart2.jpg")
else:
    st.write("ไม่ทำนาย")