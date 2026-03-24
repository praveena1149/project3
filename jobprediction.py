
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle

# loading the model
model = pickle.load(open("model2.pkl", "rb"))
encoder = pickle.load(open("encoder2.pkl", "rb"))
scaler=pickle.load(open("scaler2.pkl","rb"))
# loading the dataset
df=pd.read_csv('project3_cleaned.csv')

page=st.sidebar.radio('Navigation',['Home','Job prediction','About'])

if page == 'Home':
    st.title('JOB PREDICTION APPLICATION')
    st.image('https://media.istockphoto.com/id/537503733/photo/businessman-search-for-dream-job.jpg?s=612x612&w=0&k=20&c=RxaI6YNAOQKyJTnaP_xX9HN6dfFbPwIb68MAKUjehAg=')
    st.write('The project is to analyze the data collected by the HR department and to build a model that predicts whether the candidate will placed in the company or not.')  
   
if  page == 'Job prediction':    

   st.title("JOB ACCEPTANCE PREDICTION")
   
   age_years = st.slider("Age", 18, 60)
   gender=st.selectbox('Gender:',['Male','Female'])
   ssc_percentage = st.number_input("Enter ssc_percentage")
   hsc_percentage = st.number_input("Enter hsc_percentage")
   degree_percentage = st.number_input("Enter degree_percentage")
   degree_specialization = st.selectbox("Degree Specialization",["Computer Science", "Mechanical", "Information Technology", "Electronics", "Others"])
   aptitude_score=st.number_input("Enter aptitude score")
   communication_score=st.number_input("Enter communication score")
   technical_score=st.number_input("Enter technical score")
   skills_match_percentage=st.number_input("Enter skills match percentage")
   certifications_count=st.selectbox("Count",['1','2','3'])
   internship_experience = st.selectbox("Internship Experience", ['Yes', 'No'])
   years_of_experience = st.number_input(" Enter Years of Experience")
   career_switch_willingness = st.selectbox("Career Switch Willingness", ['Willing', 'not willing'])
   relevant_experience = st.selectbox("Relevant Experience ", ['Relevant','Not Relevant'])
   previous_ctc_lpa=st.number_input("Enter previous ctc lpa")
   expected_ctc_lpa=st.number_input("Enter expected ctc lpa")
   company_tier = st.selectbox("Company Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
   job_role_match = st.selectbox("Job Role Match ", ['Matched','Not Matched'])
   competition_level = st.selectbox("Competition Level", ['Low', 'Medium', 'High'])
   bond_requirement = st.selectbox("Bond Requirement", ['Required', 'Not Required'])
   notice_period_days = st.number_input("Enter Notice Period (Days)")
   layoff_history = st.selectbox("Layoff History", ['Yes', 'No'])
   employment_gap_months = st.number_input("Employment Gap (Months)")
   relocation_willingness = st.selectbox("Relocation Willingness", ['Willing','Not Willing'])
   experience_category=st.selectbox("experience_category",["Junior","Fresher","Senior"])
   academic_performance_band=st.selectbox("academic_performance_band",["Medium","High","Low"])
   skills_match_level=st.selectbox("skills_match_level",["High","Medium","Low"])
   interview_score = st.slider("Interview Score", 0, 100)
   interview_performance_category=st.selectbox("interview_performance_category",["Average","Excellent","Poor"])
   placement_probability_score=st.number_input("Enter the placement probablity score")
   
   
   # encoding the categorical features
   
   gender_mapping={'Male':0,'Female':1}
   gender=gender_mapping[gender]
   
   degree_specialization_mapping={'Computer Science':0,'Electronics':1,'Information Technology':2,'Mechanical':3,'Others':4}
   degree_specialization=degree_specialization_mapping[degree_specialization]

   internship_experience_mapping={'No':0,'Yes':1}
   internship_experience=internship_experience_mapping[internship_experience]

   career_switch_willingness_mapping={'Not Willing':0,'Willing':1}
   career_switch_willingness=career_switch_willingness_mapping[career_switch_willingness]  

   relevant_experience_mapping={'Not Relevant':0,'Relevant':1}
   relevant_experience=relevant_experience_mapping[relevant_experience]

   company_tier_mapping={'Tier 1':0,'Tier 2':1,'Tier 3':2}
   company_tier=company_tier_mapping[company_tier]

   job_role_match_mapping={'Not Matched':0,'Matched':1}
   job_role_match=job_role_match_mapping[job_role_match]

   competition_level_mapping={'Low':0,'Medium':1,'High':2}
   competition_level=competition_level_mapping[competition_level]

   bond_requirement_mapping={'Not Required':0,'Required':1}
   bond_requirement=bond_requirement_mapping[bond_requirement]   

   layoff_history_mapping={'No':0,'Yes':1}
   layoff_history=layoff_history_mapping[layoff_history]

   relocation_willingness_mapping={'Not Willing':0,'Willing':1}
   relocation_willingness=relocation_willingness_mapping[relocation_willingness]

   
   experience_category_mapping={'Fresher':0,'Junior':1,'Senior':2}
   experience_category=experience_category_mapping[experience_category]
   
   academic_performance_band_mapping={'Low':0,'Medium':1,'High':2}
   academic_performance_band=academic_performance_band_mapping[academic_performance_band]
   
   skills_match_level_mapping={'Low':0,'Medium':1,'High':2}
   skills_match_level=skills_match_level_mapping[skills_match_level]   

   interview_performance_category_mapping={'Poor':0,'Average':1,'Excellent':2}
   interview_performance_category=interview_performance_category_mapping[interview_performance_category]

   if st.button("Predict"):
    data = np.array([[age_years,gender,ssc_percentage,hsc_percentage,
       degree_percentage, degree_specialization, technical_score,
       aptitude_score,communication_score,skills_match_percentage,
       certifications_count,internship_experience,years_of_experience,
       career_switch_willingness,relevant_experience,previous_ctc_lpa,expected_ctc_lpa,company_tier,job_role_match,
       competition_level,bond_requirement,notice_period_days,
       layoff_history,employment_gap_months,relocation_willingness,experience_category,academic_performance_band,
       skills_match_level,interview_score,interview_performance_category,placement_probability_score]])
    
    # scaling input features
    
    num_cols=np.array([[age_years,ssc_percentage,hsc_percentage,degree_percentage,
         technical_score,aptitude_score,communication_score,
         skills_match_percentage,certifications_count,
         years_of_experience,previous_ctc_lpa,expected_ctc_lpa,
         notice_period_days,employment_gap_months]])
    
       
    scaling = scaler.transform(num_cols)
    
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
    st.write("Random Forest Accuracy: 88%")
    st.subheader('Evaluation metrics used:')
    st.write("""
    - Accuracy
    - Precision score
    - Recall
    - F1 score""")
    st.subheader(" Developed By")
    st.write("Praveena Ramesh")