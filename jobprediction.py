
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle

# loading the model
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
# loading the dataset
df=pd.read_csv('project3_cleaned.csv')


page=st.sidebar.radio('Navigation',['Home','Visualizations','Job prediction','About'])

if page == 'Home':
    st.title('JOB PREDICTION APPLICATION')
    st.image('https://media.istockphoto.com/id/537503733/photo/businessman-search-for-dream-job.jpg?s=612x612&w=0&k=20&c=RxaI6YNAOQKyJTnaP_xX9HN6dfFbPwIb68MAKUjehAg=')
    st.write('The project is to analyze the data collected by the HR department and to build a model that predicts whether the candidate will placed in the company or not.')

if page == 'Visualizations':

       st.write("visualization")   
   
   


      
   

   
if  page == 'Job prediction':    

   st.title("JOB ACCEPTANCE PREDICTION")
   
   age_years = st.slider("Age", 18, 60)
   gender = st.selectbox("Gender", ['Male','Female'])
   ssc_percentage = st.number_input("Enter ssc_percentage")
   hsc_percentage = st.number_input("Enter hsc_percentage")
   degree_percentage = st.number_input("Enter degree_percentage")
   degree_specialization = st.selectbox("Degree Specialization",["Computer Science", "Mechanical", "Information Technology", "Electronics", "Others"])
   aptitude_score=st.number_input("Enter aptitude score")
   communication_score=st.number_input("Enter communication score")
   technical_score=st.number_input("Enter technical score")
   skills_match_percentage=st.number_input("Enter skills match percentage")
   certifications_count=st.selectbox("Count",['1','2','3'])
   internship_experience = st.selectbox("Internship Experience", ['yes', 'No'])
   years_of_experience = st.number_input(" Enter Years of Experience")
   career_switch_willingness = st.selectbox("Career Switch Willingness", ['Willing', 'not willing'])
   relevant_experience = st.selectbox("Relevant Experience ", ['Relevant','Not Relevant'])
   previous_ctc_lpa=st.number_input("Enter previous ctc lpa")
   ecpected_ctc_lpa=st.number_input("Enter expected ctc lpa")
   company_tier = st.selectbox("Company Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
   job_role_match = st.selectbox("Job Role Match ", ['Matched','Not Matched'])
   competition_level = st.selectbox("Competition Level", ['Low', 'Medium', 'High'])
   bond_requirement = st.selectbox("Bond Requirement", ['Required', 'Not Required'])
   notice_period = st.number_input("Enter Notice Period (Days)")
   layoff_history = st.selectbox("Layoff History", ['Yes', 'No'])
   employment_gap_months = st.number_input("Employment Gap (Months)")
   relocation_willingness = st.selectbox("Relocation Willingness", ['Willing','Not Willing'])
   interview_score = st.slider("Interview Score", 0, 100)
   
   # encoding
   gender_encoded=encoder.transform([gender])
   degree_specialization_encoded=encoder.transform([degree_specialization])
   internship_experience_encoded=encoder.transform([internship_experience])
   career_switch_willingness_encoded= encoder.transform([career_switch_willingness])
   relevant_experience_encoded=encoder.transform([relevant_experience])
   company_tier_encoded=encoder.transform([company_tier])
   job_role_match_encoded=encoder.transform([job_role_match])
   competition_level_encoded=encoder.transform([competition_level])
   bond_requirement_encoded=encoder.transform([bond_requirement])
   layoff_history_encoded=encoder.transform([layoff_history])
   relocation_willingness_encoded=encoder.transform([relocation_willingness])
   
   encoder = OneHotEncoder(handle_unknown='ignore')
   
if st.button("Predict"):
    data = np.array([['age_years', 'gender', 'ssc_percentage', 'hsc_percentage',
       'degree_percentage', 'degree_specialization', 'technical_score',
       'aptitude_score', 'communication_score', 'skills_match_percentage',
       'certifications_count', 'internship_experience', 'years_of_experience',
       'career_switch_willingness', 'relevant_experience', 'previous_ctc_lpa',
       'expected_ctc_lpa', 'company_tier', 'job_role_match',
       'competition_level', 'bond_requirement', 'notice_period_days',
       'layoff_history', 'employment_gap_months', 'relocation_willingness','interview_score']])
    
# scaling    
data_scaled = scaler.transform(data)
    
prediction = model.predict(data)
    
if prediction[0] == 1:
        st.success("placed")
else:
        st.error("Not placed")




if page == 'About':
    st.title('JOB PREDICTION APPLICATION')
    st.write('This application is used  predicts whether a candidate will get placed in the company based on the academis scores,interview scores,communication skills,technical skills and other criterias.')
    st.subheader("Features:")
    st.write("""
       - Predict placement outcome  
       - Analyze candidate performance  
       - Visualize insights """)
    st.subheader("Dataset Includes:")
    st.write("""
    - Age
    - Gender        
    - SSC Percentage  
    - HSC Percentage  
    - Degree Percentage
    - previous lpa
    - Expected lpa  
    - Technical Score 
    - Interview score 
    - Placement Status  """)
    st.subheader(" Models Used:")
    st.write("""
    - Logistic Regression 
    - Decision trees 
    - Random Forest """)
    st.subheader("Libraries used:")
    st.write("""
    - Python  
    - Scikit-learn  
    - Pandas
    - NumPy 
    - Matplotlib
    - Seaborn 
    - Streamlit""")
    st.subheader('Confirmed model with accuracy:')
    st.write("Random Forest Accuracy: 81%")
    st.subheader('Evaluation metrics used:')
    st.write("""
         - Accuracy
         - Precision score
         - Recall
         - F1 score""")
    st.subheader(" Developed By")
    st.write("Praveena Ramesh")