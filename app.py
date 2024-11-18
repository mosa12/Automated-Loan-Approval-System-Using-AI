import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Model or scaler file not found. Please upload 'model.pkl' and 'scaler.pkl'.")
    st.stop()

# App Title and Description
st.title('Automated Loan Approval System Using AI')
st.subheader('By TEAM-7')
st.markdown("### Predict your loan approval status by filling in the details below:")

# User Inputs with number input and slider behavior
no_of_dep = st.number_input('Number of Dependents', min_value=0, max_value=5, step=1, help="Enter the number of dependents (0-5).")
grad = st.selectbox('Education Level', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self-Employed', ['Yes', 'No'])
annual_income = st.number_input('Annual Income (in ₹)', min_value=0, max_value=10000000, step=10000, help="Enter your annual income.")
loan_amount = st.number_input('Loan Amount (in ₹)', min_value=0, max_value=10000000, step=10000, help="Enter the loan amount.")
loan_dur = st.number_input('Loan Duration (Years)', min_value=0, max_value=30, step=1, help="Enter the loan duration in years.")
cibil = st.number_input('CIBIL Score', min_value=300, max_value=900, step=10, help="Enter your CIBIL score (typically between 300 and 900).")
assets = st.number_input('Net Assets Value (in ₹)', min_value=0, max_value=10000000, step=10000, help="Enter the value of your net assets.")

# Encode categorical variables
grad_encoded = 0 if grad == 'Graduated' else 1
self_emp_encoded = 1 if self_emp == 'Yes' else 0

# Prediction Logic
if st.button("Predict"):
    try:
        # Create DataFrame for prediction
        pred_data = pd.DataFrame([[no_of_dep, grad_encoded, self_emp_encoded, annual_income, loan_amount, loan_dur, cibil, assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        
        # Scale the data
        pred_data_scaled = scaler.transform(pred_data)
        
        # Make prediction
        prediction = model.predict(pred_data_scaled)
        
        # Display the result
        if prediction[0] == 1:
            st.success('🎉 Loan is Approved!')
        else:
            st.error('❌ Loan is Rejected. Try adjusting your inputs or reviewing your financial status.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
