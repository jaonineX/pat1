from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from sklearn import datasets
#from sklearn import metrics

import pandas as pd
import streamlit as st

# โหลดข้อมูล Iris
df = pd.read_csv("./data/iris.csv")
X = df.drop('variety',axis=1)
y = df['variety']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes (ในกรณีของ Iris จะใช้ Gaussian Naive Bayes)
clf = GaussianNB()
clf.fit(X_train, y_train)

# ทดสอบโมเดล
#y_pred = clf.predict(X_test)

st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")
spW=st.number_input('Insert sepalwidth')
spL=st.number_input('Insert sepallength')
ptW=st.number_input('Insert petalwidth')
ptL=st.number_input('Insert petallength')

if st.button("พยากรณ์"):
    x_input=[[spW,spL,ptW,ptL]] # ใส่ข้อมูลสำหรับการจำแนกข้อมูล
    y_predict2=clf.predict(x_input)
    st.write(y_predict2)
    st.button("ไม่พยากรณ์")
else:
    st.button("ไม่พยากรณ์")