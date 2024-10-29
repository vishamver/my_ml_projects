import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import numpy as np

# Set up the S3 client
session = boto3.Session(profile_name='default')
s3 = session.resource('s3')

# Define the bucket and file path
bucket_name = 'vishamber'
file_path = 'credit_card_fraud_dataset.csv'

# Load the data
obj = s3.Object(bucket_name, file_path)
response = obj.get()
data = pd.read_csv(response['Body'])

# Display data information
print(data.head())
print(data.info())
print(data.isnull().sum())

# Data preprocessing
data = data.dropna()
data = data.drop(columns=['TransactionID', 'TransactionDate'])
data = pd.get_dummies(data, columns=['TransactionType', 'Location'], drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['IsFraud'])
y = data['IsFraud']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Test model accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(clf, 'credit_card_fraud_model.pkl')

# Streamlit UI
st.title('Credit Card Fraud Detection')
st.markdown('Predict whether a transaction is fraudulent.')

st.header('Transaction Features')

amount = st.slider('Amount', 0.0, 25691.16, 0.0)
merchant_id = st.number_input('Merchant ID', min_value=0, max_value=1000, step=1)
transaction_type_refund = st.selectbox('Transaction Type', ['Purchase', 'Refund'])
transaction_type_encoded = 1 if transaction_type_refund == 'Refund' else 0

location = st.selectbox('Location', [
    'San Antonio', 'Dallas', 'New York', 'Philadelphia', 'Phoenix', 
    'San Diego', 'San Jose', 'Houston', 'Los Angeles'
])

location_encoded = [
    1 if location == 'Dallas' else 0, 
    1 if location == 'Houston' else 0, 
    1 if location == 'Los Angeles' else 0,
    1 if location == 'New York' else 0,
    1 if location == 'Philadelphia' else 0,
    1 if location == 'Phoenix' else 0,
    1 if location == 'San Antonio' else 0,
    1 if location == 'San Diego' else 0,
    1 if location == 'San Jose' else 0
]

features = np.array([[amount, merchant_id, transaction_type_encoded] + location_encoded])

if st.button('Predict'):
    prediction = clf.predict(features)
    if prediction[0] == 1:
        st.write('**Prediction**: This transaction is fraudulent!')
    else:
        st.write('**Prediction**: This transaction is not fraudulent.')
