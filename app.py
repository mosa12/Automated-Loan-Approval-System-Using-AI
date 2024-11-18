import streamlit as st
import pandas as pd
import pickle as pk

# Load model and scaler with improved error handling
def load_model_and_scaler():
    try:
        model = pk.load(open('model.pkl', 'rb'))
        scaler = pk.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError as e:
        st.error("Model or scaler file not found. Please upload 'model.pkl' and 'scaler.pkl'.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model and scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# App Title and Description
st.title('Automated Loan Approval System Using AI')
st.subheader('By TEAM-7')
st.markdown("### Predict loan approval status for a single person or batch of people:")

# Original Input Form for single prediction
st.markdown("### Predict for a single person:")

# User Inputs with number input and slider behavior
name = st.text_input("Enter your Name:")
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

# Validation: Check if loan amount is greater than annual income
if loan_amount > annual_income:
    st.warning("Warning: The loan amount is greater than your annual income. This may affect approval.")

# Prediction Logic for single input
if st.button("Predict for Single Person"):
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
            st.success('Congratulation 🎉Your Loan is Approved!')
        else:
            st.error('Sorry Your ❌ Loan is Rejected. Try adjusting your inputs or reviewing your financial status.')
    except ValueError as e:
        st.error(f"ValueError: {e}. Please check your input values.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")


# File Upload for batch prediction (Large Number of People)
st.markdown("### Predict for a large number of people (Batch Prediction):")

# Upload Excel or CSV file
uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xls", "xlsx", "csv"])

if uploaded_file is not None:
    try:
        # Check file extension to read it correctly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display the uploaded data to the user for verification
        st.write("### Data Preview:")
        st.write(df.head())

        # Check if required columns are present in the uploaded file
        required_columns = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
        if all(col in df.columns for col in required_columns):
            # Encode categorical variables
            df['education'] = df['education'].apply(lambda x: 0 if x == 'Graduated' else 1)
            df['self_employed'] = df['self_employed'].apply(lambda x: 1 if x == 'Yes' else 0)
            
            # Scale the data
            df_scaled = scaler.transform(df[required_columns])
            
            # Make predictions for all rows
            predictions = model.predict(df_scaled)
            
            # Add predictions to the dataframe
            df['Loan_Approval'] = predictions
            df['Loan_Approval'] = df['Loan_Approval'].apply(lambda x: 'Approved' if x == 1 else 'Rejected')
            
            # Show the result in the app
            st.write("### Loan Approval Predictions for Batch:")
            st.write(df)

            # Option to download the result as an Excel file
            output_file = 'loan_approval_predictions.xlsx'
            df.to_excel(output_file, index=False)
            st.download_button(
                label="Download Loan Approval Predictions",
                data=open(output_file, 'rb').read(),
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"Missing one or more required columns: {', '.join(required_columns)}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
