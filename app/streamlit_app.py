import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import os

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.predict import load_model, predict
from src.preprocess import clean_data

# Page configuration
st.set_page_config(
    page_title="Mental Health Survey Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        color: #2E86AB;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2E86AB;
        font-size: 1.3em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #E8F4F8;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Mental Health Survey Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict mental health treatment likelihood based on survey responses</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Home", "Make Prediction", "About"])

# Home Page
if app_mode == "Home":
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 About This App</div>', unsafe_allow_html=True)
        st.write("""
        This application uses machine learning to predict whether someone is likely to seek 
        mental health treatment based on their survey responses.
        
        **Key Features:**
        - 🤖 AI-powered predictions
        - 📋 Comprehensive survey form
        - 📈 Real-time results
        
        **How it works:**
        1. Fill out the survey questionnaire
        2. Provide your personal and workplace information
        3. Get an instant prediction about mental health treatment likelihood
        """)
    
    with col2:
        st.markdown('<div class="section-header">ℹ️ Survey Information</div>', unsafe_allow_html=True)
        st.info("""
        The survey covers:
        - **Demographics**: Age, Gender, Country
        - **Employment**: Role, Company size, Remote work
        - **Mental Health**: Family history, Previous treatment
        - **Workplace**: Benefits, Wellness programs, Work interference
        - **Attitudes**: Willingness to discuss, Consequences
        """)

# Make Prediction Page
elif app_mode == "Make Prediction":
    st.markdown("---")
    st.markdown('<div class="section-header">📝 Survey Form</div>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("survey_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=15, max_value=70, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            country = st.text_input("Country", "United States")
            state = st.text_input("State", "NA")
            
        with col2:
            st.subheader("Employment Info")
            self_employed = st.selectbox("Self-employed?", ["Yes", "No"])
            no_employees = st.selectbox("Company size", 
                ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
            remote_work = st.selectbox("Work remotely?", ["Yes", "No"])
            tech_company = st.selectbox("Tech company?", ["Yes", "No"])
        
        with col3:
            st.subheader("Mental Health Background")
            family_history = st.selectbox("Family history of mental illness?", ["Yes", "No"])
            work_interfere = st.selectbox("Work interference due to mental health", 
                ["Never", "Rarely", "Sometimes", "Often"])
            mental_health_interview = st.selectbox("Mental health affects interview?",
                ["Yes", "No", "Maybe"])
            phys_health_interview = st.selectbox("Physical health affects interview?",
                ["Yes", "No", "Maybe"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Workplace Support")
            benefits = st.selectbox("Employer provides health insurance?", 
                ["Yes", "No", "Don't know"])
            care_options = st.selectbox("Aware of care options?", 
                ["Yes", "No", "Not sure"])
            wellness_program = st.selectbox("Wellness program available?", 
                ["Yes", "No", "Don't know"])
        
        with col2:
            st.subheader("Attitudes & Support")
            seek_help = st.selectbox("Willing to seek help?", 
                ["Yes", "No", "Maybe"])
            anonymity = st.selectbox("Concerned about anonymity?", 
                ["Yes", "No"])
            coworkers = st.selectbox("Comfortable discussing with coworkers?", 
                ["Yes", "No", "Some of them"])
        
        with col3:
            st.subheader("Work Environment")
            supervisor = st.selectbox("Supervisor could affect career?", 
                ["Yes", "No", "Maybe"])
            mental_health_consequence = st.selectbox("Mental health interview affects hiring?", 
                ["Yes", "No", "Don't know"])
            phys_health_consequence = st.selectbox("Physical health interview affects hiring?", 
                ["Yes", "No", "Don't know"])
        
        col1, col2 = st.columns(2)
        with col1:
            leave = st.selectbox("Easy to take mental health days?", 
                ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
        with col2:
            mental_vs_physical = st.selectbox("Mental vs Physical health priority?",
                ["Mental health", "Physical health", "Both equal", "Don't know"])
            obs_consequence = st.selectbox("Observed negative consequences at work?", 
                ["Yes", "No"])
        
        # Submit button
        submitted = st.form_submit_button("🔮 Make Prediction", use_container_width=True)
    
    if submitted:
        try:
            # Prepare input data (exclude 'treatment' - that's what we're predicting)
            input_data = {
                'Age': age,
                'Gender': gender,
                'Country': country,
                'state': state,
                'self_employed': self_employed,
                'family_history': family_history,
                'work_interfere': work_interfere,
                'no_employees': no_employees,
                'remote_work': remote_work,
                'tech_company': tech_company,
                'benefits': benefits,
                'care_options': care_options,
                'wellness_program': wellness_program,
                'seek_help': seek_help,
                'anonymity': anonymity,
                'leave': leave,
                'mental_health_consequence': mental_health_consequence,
                'phys_health_consequence': phys_health_consequence,
                'coworkers': coworkers,
                'supervisor': supervisor,
                'mental_health_interview': mental_health_interview,
                'phys_health_interview': phys_health_interview,
                'mental_vs_physical': mental_vs_physical,
                'obs_consequence': obs_consequence
            }
            
            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
            
            # Make prediction
            prediction = predict(input_data)
            
            if prediction is not None:
                # Display result with confidence indicator
                if prediction == 1 or prediction == 'Yes':
                    st.success(f"✅ Prediction: **Likely to seek mental health treatment**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Result", "Treatment Seeking", delta="Positive")
                else:
                    st.info(f"⚠️ Prediction: **May not seek mental health treatment**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Result", "No Treatment Seeking", delta="Cautionary")
                
                # Display input summary
                st.write("\n**Input Summary:**")
                summary_df = pd.DataFrame([input_data])
                st.dataframe(summary_df, width='stretch')
            else:
                st.error("❌ Error making prediction. Please check your inputs.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.write("Please ensure all fields are filled correctly and try again.")

# About Page
elif app_mode == "About":
    st.markdown("---")
    st.markdown('<div class="section-header">📖 About this Application</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Dataset**')
        st.write("""
        This application is built on a comprehensive mental health survey dataset 
        containing responses from tech workers regarding mental health attitudes 
        and workplace support.
        
        **Dataset Features:**
        - 27 survey questions
        - Multiple choice responses
        - Covers demographics, employment, and mental health aspects
        """)
    
    with col2:
        st.markdown('**Model**')
        st.write("""
        The prediction model is trained using machine learning algorithms 
        to identify patterns in mental health treatment-seeking behavior.
        
        **Technologies:**
        - Scikit-learn for ML
        - Pandas for data processing
        - Streamlit for UI
        """)
    
    st.markdown("---")
    st.markdown('**Privacy & Disclaimer**')
    st.warning("""
    ⚠️ **Important Notice:**
    - This tool is for educational and informational purposes only
    - Predictions are not a substitute for professional mental health advice
    - Please consult a mental health professional for proper diagnosis and treatment
    - Your data is not stored or shared with any third parties
    """)
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #999; font-size: 0.9em;">Mental Health Survey Predictor v1.0</div>', unsafe_allow_html=True)
