import streamlit as st
import pickle
import numpy as np

# 1. Load model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Classifier")
st.write("Enter the features below:")

# 2. Sliders use karke inputs lena
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.5, value=0.2, step=0.1)

# 3. Predict button aur output logic
if st.button("Predict"):
    # Features ko 2D array me convert karna
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction lena
    prediction = model.predict(features)

    # Result show karna
    st.write(f"The predicted species is: {prediction[0]}")
