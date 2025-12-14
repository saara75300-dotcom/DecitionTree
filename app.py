!pip install streamlit -q
import streamlit as st
import pickle
import numpy as np
import pandas as pd 
try:
    with open('dtree_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error("Error: 'dtree_classifier.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()
st.title('Decision Tree Classifier Prediction App')
st.write('Enter the feature values to get a prediction for Class 0 or Class 1.')
feature_1 = st.slider('Feature 1', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
feature_2 = st.slider('Feature 2', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
feature_3 = st.slider('Feature 3', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
feature_4 = st.slider('Feature 4', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
if st.button('Predict'):
    # Prepare input for the model
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.success('The model predicts: Class 0')
    else:
        st.success('The model predicts: Class 1')
