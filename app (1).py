import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open('iris_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🌸 Iris Flower Classifier")

labels = ["Setosa", "Versicolor", "Virginica"]

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.5, 0.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    
    try:
        proba = model.predict_proba(features)
        confidence = np.max(proba)
        st.success(f"Prediction: {labels[prediction]} ({confidence:.2f} confidence)")
    except:
        st.success(f"Prediction: {labels[prediction]}")
