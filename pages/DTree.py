import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Header
# =========================
st.header("Decision Tree for Health Risk Classification")

# =========================
# โหลดข้อมูล
# =========================
df = pd.read_csv("./data/Health_Risk_Dataset_Encoded.csv")
st.write("แสดงข้อมูลตัวอย่าง:")
st.write(df.head(10))

# =========================
# กำหนด Target column
# =========================
target_col = "Risk_Level_Num"

# ลบคอลัมน์ที่ไม่จำเป็น (ถ้ามี)
drop_cols = [target_col, "Patient_ID", "Risk_Level"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df[target_col]

# =========================
# Train/Test Split
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=200
)

# =========================
# Train model
# =========================
ModelDtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree = ModelDtree.fit(x_train, y_train)

# =========================
# UI สำหรับป้อนข้อมูลใหม่
# =========================
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")
input_data = {}
for col in X.columns:
    # ถ้าเป็น int ให้ default เป็น int
    if df[col].dtype == "int64":
        input_data[col] = st.number_input(f"Insert {col}", value=0)
    else:
        input_data[col] = st.number_input(f"Insert {col}", value=0.0)

if st.button("พยากรณ์"):
    x_input = [list(input_data.values())]
    y_predict2 = dtree.predict(x_input)
    class_names = {0: "Low", 1: "Medium", 2: "High"}
    st.write("ผลการพยากรณ์:", class_names[y_predict2[0]])

# =========================
# Accuracy
# =========================
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์ {(score*100):.2f} %')

# =========================
# วาด Decision Tree
# =========================
fig, ax = plt.subplots(figsize=(14, 8))
tree.plot_tree(
    dtree,
    feature_names=X.columns,
    class_names=["Low", "Medium", "High"],
    filled=True,
    fontsize=10,
    ax=ax
)
st.pyplot(fig)
