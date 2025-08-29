import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# โหลดข้อมูล
st.header("Decision Tree for classification")

df = pd.read_csv("Health_Risk_Dataset_Encoded.csv")
st.write("แสดงข้อมูลตัวอย่าง:")
st.write(df.head(10))

# กำหนด target column
target_col = "Risk_Level"

# แยก Features และ Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# แปลงทุก feature ที่ไม่ใช่ตัวเลขให้กลายเป็นตัวเลข
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# แปลง target y ถ้าไม่ใช่ numeric
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y.astype(str))
else:
    le_target = None  # ถ้าเป็น numeric อยู่แล้ว

# แบ่งข้อมูล train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# Train model
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

# UI สำหรับป้อนข้อมูลใหม่
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Insert {col}", value=0.0)

if st.button("พยากรณ์"):
    x_input = [list(input_data.values())]
    y_predict2 = dtree.predict(x_input)
    if le_target:  # แปลงกลับเป็น label
        y_predict2 = le_target.inverse_transform(y_predict2)
    st.write("ผลการพยากรณ์:", y_predict2)

# Accuracy
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์ {(score*100):.2f} %')

# วาด Decision Tree
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(
    dtree, 
    feature_names=X.columns, 
    class_names=[str(c) for c in (le_target.classes_ if le_target else pd.Series(y).unique())], 
    filled=True, 
    ax=ax
)
st.pyplot(fig)
