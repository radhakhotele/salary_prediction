import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest Regressor model
model_filename = 'random_forest_regressor_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Re-create and fit LabelEncoders using the original data's unique values
# (Assuming the original df was available during deployment setup)
# In a real deployment, these encoders would ideally be saved alongside the model.

# Create a dummy DataFrame with unique values to fit the encoders correctly
# This assumes the categorical values are consistent with the training data
original_gender_values = ['Male', 'Female']
original_education_values = ['Bachelor's', 'Master's', 'PhD']
# Extract unique job titles from the 'df' DataFrame (from the notebook context)
# For a true deployment, you would save these unique values or the fitted encoder
# Note: This requires 'df' to be available. In a production app, you'd load these from a saved file.
# For now, we'll try to use the 'original_job_title_values' if 'df' is not directly accessible here.
# A more robust solution for deployment is to save the LabelEncoder objects directly.

# Fallback for job titles if 'df' is not directly available in this scope during app.py generation
# In a real deployment, ensure 'original_job_title_values' are persisted or derived from a fixed source.
# For this example, we will use a placeholder or assume the original 'df' values are known and fixed.

# To ensure the Streamlit app can be run independently, we must get the unique job titles 
# that were used during training. Since we can't directly access 'df' here if this code
# were truly isolated, we would need a saved list of job titles. For this Colab context,
# we'll use the 'original_job_title_values' from the kernel, which represents the training data's titles.
# If this were a true standalone app, these would be loaded from a file or hardcoded.

# Using original_job_title_values from the kernel state, which is LabelEncoded. 
# This implies the Streamlit app needs to decode them for display, then re-encode user input.
# This is a bit complex for a simple app. A better approach is to save the LabelEncoders.
# For a proper deployment, `le_job_title` should be saved as well.

# As a workaround for this in-notebook `app.py` generation, we will manually provide a list
# of potential job titles for the Streamlit dropdown/input, representing the original ones.
# In a real deployment, you would load this from a JSON/CSV file alongside your model.

# Given the current notebook state, original_job_title_values is an array of encoded integers.
# To get the original string values, we would need the inverse transform of le_job_title, 
# but le_job_title itself was fitted on already-encoded values in cell 68075e2b within the Streamlit code block. 
# This is a recursive problem if not handled by saving the fitted encoders. 

# Let's assume for this app generation that we have a way to get the original string job titles.
# For demonstration, we will use a small sample or require user to input known titles.

# Correct approach for deployment: save the fitted LabelEncoders (le_gender, le_education, le_job_title)
# as well as the model. Since the notebook only used one global `le` object in cell 214100fd 
# for all columns, and then recreated specific ones in cell 68075e2b, we need to adapt.

# Let's simplify for `app.py` generation: we'll load the original dataframe to get job titles if possible,
# or just provide a placeholder list. Given the notebook state, `df` is available. We can re-extract original
# job titles for the `LabelEncoder` fitting inside the `app.py` for simplicity.

# Re-create LabelEncoders based on the *original* categorical values (before encoding)
# This is crucial for the app to correctly encode user input.
# We need the unique, original string values for Job Title. The 'df' in the notebook context
# has already been label-encoded. So `df['Job Title'].unique()` will return encoded integers.
# We must rely on having the *original* string values.

# Since `original_gender_values` and `original_education_values` are hardcoded in the provided Streamlit code,
# we need a similar approach for `original_job_title_values` for `app.py` to be self-contained for testing.
# If the actual `df` is available from the context, we can use its column `Job Title` to get original string values.
# However, `df` in the notebook is already label-encoded. 

# To make this app self-contained and runnable, it is best practice to save the LabelEncoders along with the model.
# As a workaround, we will temporarily reload `Salary Data.csv` within the `app_code` to get the original strings
# to fit the LabelEncoders. This is not ideal but makes `app.py` runnable.

# Alternative for `app.py` generation: 
# Instead of trying to use `df` from current context (which is already encoded),
# we will re-read the CSV temporarily inside the Streamlit app script to get the original strings.

try:
    temp_df_for_encoders = pd.read_csv('/content/Salary Data.csv') # Adjust path if needed
    # Handle missing values as done in the notebook for categorical columns to get mode
    for col in ['Gender', 'Education Level', 'Job Title']:
        temp_df_for_encoders[col] = temp_df_for_encoders[col].fillna(temp_df_for_encoders[col].mode()[0])
    
    original_job_title_strings = sorted(temp_df_for_encoders['Job Title'].unique())
except Exception as e:
    st.error(f"Error loading data for encoders: {e}. Please ensure 'Salary Data.csv' is accessible.")
    st.stop()
    original_job_title_strings = [] # Fallback

le_gender = LabelEncoder()
le_gender.fit(original_gender_values)

le_education = LabelEncoder()
le_education.fit(original_education_values)

le_job_title = LabelEncoder()
le_job_title.fit(original_job_title_strings) # Fit with original string values

# Streamlit App Interface
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields
age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', original_gender_values)
education = st.selectbox('Education Level', original_education_values)
job_title_input = st.selectbox('Job Title', original_job_title_strings) # Use selectbox for known titles
years_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, step=0.5)

# Prediction button
if st.button('Predict Salary'):
    try:
        # Encode categorical features
        gender_encoded = le_gender.transform([gender])[0]
        education_encoded = le_education.transform([education])[0]
        job_title_encoded = le_job_title.transform([job_title_input])[0]

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title_encoded, years_experience]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        # Make prediction
        predicted_salary = model.predict(input_data)[0]

        st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
