# Libraries
from pathlib import Path
import streamlit as st
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the model
model = tf.keras.models.load_model(BASE_DIR / "miscellaneous" / "ANN.h5")

# Load encoders and scalers
with open(BASE_DIR / "miscellaneous" / "label_encode.pkl", 'rb') as file:
    label_encode = pickle.load(file)

with open(BASE_DIR / "miscellaneous" / "onehot_encode.pkl", 'rb') as file:
    onehot_encode = pickle.load(file)

with open(BASE_DIR / "miscellaneous" / "scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


# Streamlit Code:
st.title('Customer Churn Prediction')

# Input
geography = st.selectbox('Geography', onehot_encode.categories_[0])
gender = st.selectbox('Gender', label_encode.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Activate Member', [0,1])


input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encode.transform([gender])[0]],
    'Age' : [age],
    'Tenure': [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encode = onehot_encode.transform([[geography]])
geo_encode_df = pd.DataFrame(geo_encode, columns = onehot_encode.get_feature_names_out(['Geography']))

# Concat the categorical feature to the input data after it is encoded correctly
input_data = pd.concat([input_data.reset_index(drop = True), geo_encode_df], axis = 1)

# Scale the data as per our training data
input_scaled = scaler.transform(input_data)

# Predict the churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability : {prediction_prob:.2f}')

if prediction >= 0.5:
    st.write("Customer likely to Churn.")
else:
    st.write("Customer not likely to Churn.")

