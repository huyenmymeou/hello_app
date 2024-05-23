import streamlit as st
st.write("Hello world")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Giao diện Streamlit
st.title("Iris Flower Classifier")
st.write("This app uses a Random Forest classifier to predict the species of Iris flower based on its sepal and petal measurements.")

# Các thông số đầu vào
st.sidebar.header("Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider('Petal length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider('Petal width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Hiển thị các thông số đầu vào
st.subheader('Input Parameters')
st.write(input_df)

# Dự đoán
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Hiển thị kết quả dự đoán
st.subheader('Prediction')
st.write(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)
