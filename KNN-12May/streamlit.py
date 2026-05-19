import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsRegressor

st.title("KNN Regressor - Iris Dataset")

# Load and prepare data
@st.cache_data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

df, target_names = load_data()

x = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']

# Train model (using Regressor as requested)
knn_reg = KNeighborsRegressor(n_neighbors=5, metric='minkowski', p=2)
knn_reg.fit(x, y)

st.sidebar.header("Input Features")
# Sliders for input features
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(x.iloc[:, 0].min()), float(x.iloc[:, 0].max()), float(x.iloc[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(x.iloc[:, 1].min()), float(x.iloc[:, 1].max()), float(x.iloc[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(x.iloc[:, 2].min()), float(x.iloc[:, 2].max()), float(x.iloc[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(x.iloc[:, 3].min()), float(x.iloc[:, 3].max()), float(x.iloc[:, 3].mean()))

if st.button("Predict"):
    prediction = knn_reg.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f"### Predicted Target Value: {prediction[0]:.2f}")
