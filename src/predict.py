"""
predict.py
Load saved model and make predictions on new student data.
"""

import joblib
import numpy as np
import pandas as pd

def load_artifacts():
    """Load saved model, scaler, encoders."""
    model   = joblib.load('models/best_model.pkl')
    scaler  = joblib.load('models/scaler.pkl')
    le_dict = joblib.load('models/label_encoders.pkl')
    features= joblib.load('models/feature_names.pkl')
    return model, scaler, le_dict, features


def predict_student(student_dict):
    """
    Predict final score for one student.
    
    Parameters:
        student_dict: dict with all student feature values
    
    Returns:
        predicted_score (float), grade (str), risk_level (str)
    """
    model, scaler, le_dict, features = load_artifacts()
    
    df = pd.DataFrame([student_dict])
    
    # Derived features (same as preprocessing.py)
    edu_map = {'No Education': 0, 'High School': 1, 'Graduate': 2, 'Postgraduate': 3}
    inc_map = {'Low': 0, 'Middle': 1, 'High': 2}
    
    df['study_efficiency'] = df['assignments_completed'] / (df['study_hours_per_day'] + 0.1)
    df['attendance_risk']  = (df['attendance_percent'] < 75).astype(int)
    df['parental_support_index'] = (
        df['parent_education'].map(edu_map) + df['family_income'].map(inc_map)
    )
    df['academic_momentum']  = df['prev_gpa'] * df['attendance_percent'] / 100
    df['absence_severity']   = pd.cut(df['absences'], bins=[-1,2,6,12,30], labels=[0,1,2,3]).astype(int)
    df['sleep_quality']      = df['sleep_hours'].apply(lambda x: 1 if 7<=x<=9 else 0)
    df['total_support']      = df['tutoring_sessions'] + df['internet_access']*3 + df['parental_support_index']
    
    # Encode categoricals
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    # Align columns with training feature order
    df = df[features]
    
    # Scale
    X_scaled = scaler.transform(df)
    
    # Predict
    score = float(np.clip(model.predict(X_scaled)[0], 0, 100))
    
    # Assign grade
    if score >= 90: grade = 'A'
    elif score >= 75: grade = 'B'
    elif score >= 60: grade = 'C'
    elif score >= 45: grade = 'D'
    else: grade = 'F'
    
    # Risk level
    if score >= 75: risk = 'Low Risk'
    elif score >= 50: risk = 'Medium Risk'
    else: risk = 'High Risk'
    
    return round(score, 1), grade, risk


if __name__ == '__main__':
    # Example prediction
    student = {
        'age': 17,
        'gender': 'Female',
        'school_type': 'Public',
        'parent_education': 'Graduate',
        'family_income': 'Middle',
        'study_hours_per_day': 4.5,
        'attendance_percent': 82.0,
        'prev_gpa': 7.5,
        'assignments_completed': 16,
        'extracurricular_activities': 2,
        'internet_access': 1,
        'tutoring_sessions': 3,
        'absences': 5,
        'sleep_hours': 7.5
    }
    
    score, grade, risk = predict_student(student)
    print(f"\n🎓 Prediction Results:")
    print(f"   Predicted Score : {score}/100")
    print(f"   Predicted Grade : {grade}")
    print(f"   Risk Level      : {risk}")