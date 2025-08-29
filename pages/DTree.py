import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# โหลดข้อมูล
st.header("Decision Tree for classification0000")

# ใช้ไฟล์ที่อัปโหลดมาแทน iris.csv
df = pd.read_csv("./data/Health_Risk_Dataset_Encoded.csv")
st.write("แสดงข้อมูลตัวอย่าง:")
st.write(df.head(10))

# เลือก target column (เช่น Risk_Level)
target_col = st.selectbox("เลือก Target Column", df.columns)

# กำหนด feature และ target
X = df.drop(target_col, axis=1)
y = df[target_col]

# แบ่งข้อมูล train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# Train model
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

