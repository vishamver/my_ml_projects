import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

# Load the Iris dataset from GitHub
df = pd.read_csv("https://raw.githubusercontent.com/vishamver/my_ml_projects/main/iris.csv")

# Feature and target selection
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)

# RandomForest Classifier
clf1 = RandomForestClassifier()
clf1.fit(X_train, y_train)

# Streamlit App
st.title('Classifying Iris Flowers')
st.markdown('Toy model to classify iris flowers into Setosa, Versicolor, Virginica')

# Sliders for user input
st.header('Plant Features')
col1, col2 = st.columns(2)

with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 3.0)
    sepal_w = st.slider('Sepal width (cm)', 0.5, 4.4, 1.0)
    
with col2:
    st.text('Petal characteristics')
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 4.0)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

# Prediction button
if st.button('Predict'):
    # Prepare the input for prediction
    input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    # Make a prediction
    prediction = clf1.predict(input_data)[0]
    
    # Display the result
    st.success(f"The predicted Iris species is: {prediction}")
