import boto3
import pandas as pd

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

print(data.head())  # Display the first few rows of the dataset
data
# Check basic info
print(data.info())
# Check for missing values
print(data.isnull().sum())
# View class distribution (0 = not fraud, 1 = fraud)
print(data['Amount'].value_counts())  # Replace 'Class' with the correct column name if it's different
data = data.dropna()  # Simplest way, or consider imputing missing values based on analysis
from sklearn.model_selection import train_test_split
import pandas as pd

# Drop columns that are not needed
data = data.drop(columns=['TransactionID', 'TransactionDate'])

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['TransactionType', 'Location'], drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['IsFraud'])  # Feature columns
y = data['IsFraud']                 # Target column

# Split the dataset (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the split data
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
import joblib
joblib.dump(clf, 'credit_card_fraud_model.pkl')
import streamlit as st
import predction as predcit
import numpy as np

# Streamlit UI for Credit Card Fraud Detection
st.title('Credit Card Fraud Detection')
st.markdown('Predict whether a transaction is fraudulent.')

st.header('Transaction Features')

# Amount and MerchantID
amount = st.slider('Amount', 0.0, 25691.16, 0.0)
merchant_id = st.number_input('Merchant ID', min_value=0, max_value=1000, step=1)

# Transaction Type (Refund or Purchase)
transaction_type_refund = st.selectbox('Transaction Type', ['Purchase', 'Refund'])
transaction_type_encoded = 1 if transaction_type_refund == 'Refund' else 0

# Location (One-hot encoded)
location = st.selectbox('Location', [
    'San Antonio', 'Dallas', 'New York', 'Philadelphia', 'Phoenix', 
    'San Diego', 'San Jose', 'Houston', 'Los Angeles'
])

# Prepare one-hot encoded locations
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

# Combine all inputs into a feature array
features = np.array([[amount, merchant_id, transaction_type_encoded] + location_encoded])

# Prediction button
if st.button('Predict'):
    prediction = clf.predict(features)
    if prediction[0] == 1:
        st.write('**Prediction**: This transaction is fraudulent!')
    else:
        st.write('**Prediction**: This transaction is not fraudulent.')