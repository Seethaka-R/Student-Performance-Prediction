"""
streamlit_app.py
Full interactive Streamlit dashboard for student performance prediction.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #7F77DD;
        margin: 0.5rem 0;
    }
    .risk-low    { border-left-color: #1D9E75 !important; }
    .risk-medium { border-left-color: #EF9F27 !important; }
    .risk-high   { border-left-color: #E24B4A !important; }
    .grade-badge {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .st-emotion-cache-1r6slb0 { background: #f8f9ff; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    try:
        model    = joblib.load('models/best_model.pkl')
        scaler   = joblib.load('models/scaler.pkl')
        le_dict  = joblib.load('models/label_encoders.pkl')
        features = joblib.load('models/feature_names.pkl')
        return model, scaler, le_dict, features
    except FileNotFoundError:
        return None, None, None, None


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('data/processed/cleaned_data.csv')
    except:
        return None


def predict(student_dict, model, scaler, le_dict, features):
    """Make prediction for given student."""
    edu_map = {'No Education': 0, 'High School': 1, 'Graduate': 2, 'Postgraduate': 3}
    inc_map = {'Low': 0, 'Middle': 1, 'High': 2}
    
    df = pd.DataFrame([student_dict])
    
    df['study_efficiency']      = df['assignments_completed'] / (df['study_hours_per_day'] + 0.1)
    df['attendance_risk']       = (df['attendance_percent'] < 75).astype(int)
    df['parental_support_index']= df['parent_education'].map(edu_map) + df['family_income'].map(inc_map)
    df['academic_momentum']     = df['prev_gpa'] * df['attendance_percent'] / 100
    df['absence_severity']      = pd.cut(df['absences'], bins=[-1,2,6,12,30], labels=[0,1,2,3]).astype(int)
    df['sleep_quality']         = df['sleep_hours'].apply(lambda x: 1 if 7<=x<=9 else 0)
    df['total_support']         = df['tutoring_sessions'] + df['internet_access']*3 + df['parental_support_index']
    
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    df = df[features]
    X_scaled = scaler.transform(df)
    score = float(np.clip(model.predict(X_scaled)[0], 0, 100))
    
    if score >= 90: grade = 'A'
    elif score >= 75: grade = 'B'
    elif score >= 60: grade = 'C'
    elif score >= 45: grade = 'D'
    else: grade = 'F'
    
    if score >= 75: risk = 'Low Risk'
    elif score >= 50: risk = 'Medium Risk'
    else: risk = 'High Risk'
    
    return round(score, 1), grade, risk


# ── App Layout ────────────────────────────────────────────────────
def main():
    # Header
    st.title("🎓 Student Performance Prediction System")
    st.markdown("*Predict academic outcomes using machine learning — built for EdTech & school analytics*")
    st.divider()
    
    # Load data and model
    model, scaler, le_dict, features = load_model_artifacts()
    df = load_dataset()
    
    if model is None:
        st.error("⚠️ Models not found. Please run `main.py` first to train the models.")
        st.code("python main.py")
        return
    
    # ── Navigation tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict Performance",
        "📊 Dataset Explorer",
        "🤖 Model Insights",
        "📈 Batch Analysis"
    ])
    
    # ═══════════════════════════════════════════════════
    # TAB 1: Individual Prediction
    # ═══════════════════════════════════════════════════
    with tab1:
        st.subheader("Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Info**")
            age = st.slider("Age", 14, 20, 17)
            gender = st.selectbox("Gender", ["Male", "Female"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
        
        with col2:
            st.markdown("**Family Background**")
            parent_education = st.selectbox(
                "Parent Education",
                ["No Education", "High School", "Graduate", "Postgraduate"]
            )
            family_income = st.selectbox("Family Income", ["Low", "Middle", "High"])
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        
        with col3:
            st.markdown("**Academic Details**")
            prev_gpa = st.slider("Previous GPA (0–10)", 0.0, 10.0, 6.5, 0.1)
            attendance = st.slider("Attendance (%)", 40, 100, 80)
            study_hours = st.slider("Study Hours/Day", 0.5, 10.0, 3.5, 0.5)
        
        col4, col5 = st.columns(2)
        with col4:
            assignments = st.slider("Assignments Completed (out of 20)", 0, 20, 15)
            tutoring = st.slider("Tutoring Sessions/Month", 0, 7, 2)
        with col5:
            absences = st.slider("Absences (this semester)", 0, 30, 5)
            sleep_hours = st.slider("Sleep Hours/Night", 4.0, 10.0, 7.5, 0.5)
            extracurricular = st.slider("Extracurricular Activities", 0, 5, 1)
        
        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            student_data = {
                'age': age,
                'gender': gender,
                'school_type': school_type,
                'parent_education': parent_education,
                'family_income': family_income,
                'study_hours_per_day': study_hours,
                'attendance_percent': float(attendance),
                'prev_gpa': prev_gpa,
                'assignments_completed': assignments,
                'extracurricular_activities': extracurricular,
                'internet_access': 1 if internet_access == "Yes" else 0,
                'tutoring_sessions': tutoring,
                'absences': absences,
                'sleep_hours': sleep_hours
            }
            
            score, grade, risk = predict(student_data, model, scaler, le_dict, features)
            
            st.divider()
            st.subheader("📋 Prediction Results")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            with res_col1:
                st.metric("🎯 Predicted Score", f"{score}/100")
            with res_col2:
                st.metric("📝 Predicted Grade", grade)
            with res_col3:
                risk_icon = "🟢" if "Low" in risk else ("🟡" if "Medium" in risk else "🔴")
                st.metric(f"{risk_icon} Risk Level", risk)
            with res_col4:
                pass_fail = "Pass ✅" if score >= 45 else "Fail ❌"
                st.metric("📋 Status", pass_fail)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Predicted Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#7F77DD'},
                    'steps': [
                        {'range': [0, 45],  'color': '#FCEBEB'},
                        {'range': [45, 60], 'color': '#FAEEDA'},
                        {'range': [60, 75], 'color': '#E6F1FB'},
                        {'range': [75, 90], 'color': '#E1F5EE'},
                        {'range': [90, 100],'color': '#EAF3DE'},
                    ],
                    'threshold': {
                        'line': {'color': '#E24B4A', 'width': 3},
                        'thickness': 0.8,
                        'value': 45
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Personalized recommendations
            st.subheader("💡 Personalized Recommendations")
            recs = []
            if attendance < 75: recs.append("🔴 **Improve attendance** — currently below 75%, a key risk factor")
            if study_hours < 3:  recs.append("🟡 **Increase study time** — aim for at least 3–4 hours daily")
            if absences > 10:    recs.append("🔴 **Reduce absences** — high absences strongly impact performance")
            if tutoring == 0 and score < 60: recs.append("🟡 **Consider tutoring** — even 1–2 sessions/month helps")
            if sleep_hours < 6:  recs.append("🟡 **Improve sleep** — less than 6 hours reduces concentration")
            if not recs:         recs.append("🟢 **Keep it up!** Student is on a strong academic track.")
            
            for rec in recs:
                st.markdown(rec)
    
    # ═══════════════════════════════════════════════════
    # TAB 2: Dataset Explorer
    # ═══════════════════════════════════════════════════
    with tab2:
        if df is None:
            st.warning("Dataset not found. Run main.py first.")
        else:
            st.subheader("Dataset Overview")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Students", len(df))
            m2.metric("Avg Score", f"{df['final_score'].mean():.1f}")
            m3.metric("Pass Rate", f"{(df['pass_fail']=='Pass').mean()*100:.1f}%")
            m4.metric("Avg Study Hrs", f"{df['study_hours_per_day'].mean():.1f}")
            m5.metric("Avg Attendance", f"{df['attendance_percent'].mean():.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, x='final_score', nbins=30, color='grade',
                    title='Score Distribution by Grade',
                    color_discrete_map={'A':'#1D9E75','B':'#7F77DD','C':'#EF9F27','D':'#D85A30','F':'#E24B4A'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df, x='study_hours_per_day', y='final_score',
                    color='grade', size='attendance_percent',
                    title='Study Hours vs Final Score',
                    color_discrete_map={'A':'#1D9E75','B':'#7F77DD','C':'#EF9F27','D':'#D85A30','F':'#E24B4A'},
                    opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig = px.box(
                    df, x='family_income', y='final_score',
                    color='family_income', title='Score by Family Income',
                    category_orders={'family_income': ['Low', 'Middle', 'High']}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = px.box(
                    df, x='parent_education', y='final_score',
                    color='parent_education', title='Score by Parent Education'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table with filters
            st.subheader("📄 Raw Data Explorer")
            grade_filter = st.multiselect("Filter by Grade", ['A','B','C','D','F'], default=['A','B','C','D','F'])
            filtered = df[df['grade'].isin(grade_filter)]
            st.dataframe(
                filtered.head(200).style.background_gradient(subset=['final_score'], cmap='RdYlGn'),
                use_container_width=True
            )
    
    # ═══════════════════════════════════════════════════
    # TAB 3: Model Insights
    # ═══════════════════════════════════════════════════
    with tab3:
        st.subheader("Model Performance Comparison")
        
        try:
            comp_df = pd.read_csv('outputs/model_comparison.csv')
            st.dataframe(
                comp_df.style.highlight_max(subset=['R²', 'CV R² (mean)'], color='#d4f0d4')
                             .highlight_min(subset=['RMSE', 'MAE'], color='#d4f0d4'),
                use_container_width=True
            )
        except:
            st.info("Run main.py to generate model comparison results.")
        
        # Feature importance chart
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('outputs/feature_importance.png'):
                st.image('outputs/feature_importance.png', caption='Feature Importances', use_column_width=True)
        
        with col2:
            if os.path.exists('outputs/actual_vs_predicted.png'):
                st.image('outputs/actual_vs_predicted.png', caption='Actual vs Predicted', use_column_width=True)
        
        if os.path.exists('outputs/correlation_heatmap.png'):
            st.image('outputs/correlation_heatmap.png', caption='Correlation Heatmap', use_column_width=True)
    
    # ═══════════════════════════════════════════════════
    # TAB 4: Batch Analysis
    # ═══════════════════════════════════════════════════
    with tab4:
        st.subheader("Batch Predict from CSV File")
        st.markdown("Upload a CSV file with student data to get predictions for all students.")
        
        # Show expected format
        with st.expander("📋 Expected CSV format"):
            sample = pd.DataFrame([{
                'age': 17, 'gender': 'Female', 'school_type': 'Public',
                'parent_education': 'Graduate', 'family_income': 'Middle',
                'study_hours_per_day': 4.0, 'attendance_percent': 80.0,
                'prev_gpa': 7.0, 'assignments_completed': 15,
                'extracurricular_activities': 2, 'internet_access': 1,
                'tutoring_sessions': 3, 'absences': 5, 'sleep_hours': 7.5
            }])
            st.dataframe(sample)
        
        uploaded = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded:
            upload_df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(upload_df)} students")
            
            results = []
            for _, row in upload_df.iterrows():
                try:
                    score, grade, risk = predict(row.to_dict(), model, scaler, le_dict, features)
                    results.append({'Predicted Score': score, 'Grade': grade, 'Risk': risk})
                except:
                    results.append({'Predicted Score': None, 'Grade': 'Error', 'Risk': 'Error'})
            
            result_df = pd.concat([upload_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False)
            st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")


if __name__ == '__main__':
    main()