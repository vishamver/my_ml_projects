import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# Load the dataset
df = pd.read_csv("E:/creditcard.csv")

# Define features and target
X = df.drop(columns=['Class'])  # Features are all columns except 'Class'
y = df['Class']  # Target is the 'Class' column

# Check the shapes of X and y
print(X.shape)
print(y.shape)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of the split datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize the model
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(clf, 'credit_fraud_model.pkl')

# Load the trained model in Streamlit app
clf = joblib.load('credit_fraud_model.pkl')

# Streamlit UI for Credit Card Fraud Detection
st.title('Credit Card Fraud Detection')
st.markdown('Detect whether a transaction is fraudulent or not.')

st.header('Transaction Features')

# Creating two columns for better layout
col1, col2 = st.columns(2)

# Inputs in column 1
with col1:
    time = st.slider('Time', 0.0, 172792.0, 0.0)
    v1 = st.slider('V1', -56.4, 2.45, 0.0)
    v2 = st.slider('V2', -72.71, 22.57, 0.0)
    v3 = st.slider('V3', -48.33, 9.38, 0.0)
    v4 = st.slider('V4', -5.683, 16.88, 0.0)
    v5 = st.slider('V5', -113.74, 34.8, 0.0)
    v6 = st.slider('V6', -26.16, 73.3, 0.0)
    v7 = st.slider('V7', -43.56, 121.0, 0.0)
    v8 = st.slider('V8', -73.3, 20.01, 0.0)

# Inputs in column 2
with col2:
    v9 = st.slider('V9', -15.59, 33.7, 0.0)
    v10 = st.slider('V10', -50.3, 17.84, 0.0)
    v11 = st.slider('V11', -43.9, 21.03, 0.0)
    v12 = st.slider('V12', -56.40, 2.45, 0.0)
    v13 = st.slider('V13', -48.33, 9.38, 0.0)
    v14 = st.slider('V14', -5.683, 16.88, 0.0)
    v15 = st.slider('V15', -113.74, 34.8, 0.0)
    v16 = st.slider('V16', -73.22, 20.02, 0.0)
    v17 = st.slider('V17', -25.16, 73.25, 0.0)
    v18 = st.slider('V18', -9.497, 33.0, 0.0)
    v19 = st.slider('V19', -43.57, 20.34, 0.0)
    v20 = st.slider('V20', -15.50, 10.52, 0.0)
    amount = st.slider('Amount', 0.0, 25691.16, 0.0)

# Prepare the input features as an array
features = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, amount]])

# Prediction button
if st.button('Predict'):
    prediction = clf.predict(features)
    if prediction[0] == 1:
        st.write('**Prediction**: This transaction is fraudulent!')
    else:
        st.write('**Prediction**: This transaction is not fraudulent.')
