"""
preprocessing.py
Cleans the raw dataset and engineers new features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean(filepath='data/raw/student_data.csv'):
    """Load raw data and perform cleaning."""
    df = pd.read_csv(filepath)
    
    print(f"📥 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop student_id (not a feature)
    df = df.drop(columns=['student_id'], errors='ignore')
    
    # Handle missing values (fill with median for numeric, mode for categorical)
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Cap outliers using IQR (only for numeric features, not targets)
    feature_cols = ['study_hours_per_day', 'attendance_percent', 'prev_gpa',
                    'assignments_completed', 'absences', 'sleep_hours']
    
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    
    print(f"✅ Cleaned: {df.shape[0]} rows remaining")
    return df


def engineer_features(df):
    """Create new meaningful features from existing ones."""
    df = df.copy()
    
    # 1. Study efficiency: ratio of assignments done to study hours
    df['study_efficiency'] = df['assignments_completed'] / (df['study_hours_per_day'] + 0.1)
    
    # 2. Attendance risk flag (binary: below 75% is at-risk)
    df['attendance_risk'] = (df['attendance_percent'] < 75).astype(int)
    
    # 3. Parental support index (education + income combined)
    edu_map = {'No Education': 0, 'High School': 1, 'Graduate': 2, 'Postgraduate': 3}
    inc_map = {'Low': 0, 'Middle': 1, 'High': 2}
    df['parental_support_index'] = (
        df['parent_education'].map(edu_map) + df['family_income'].map(inc_map)
    )
    
    # 4. Academic momentum (GPA × attendance as combined signal)
    df['academic_momentum'] = df['prev_gpa'] * df['attendance_percent'] / 100
    
    # 5. Absence severity (absences binned)
    df['absence_severity'] = pd.cut(
        df['absences'],
        bins=[-1, 2, 6, 12, 30],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # 6. Sleep quality (optimal 7-9 hrs)
    df['sleep_quality'] = df['sleep_hours'].apply(
        lambda x: 1 if 7 <= x <= 9 else 0
    )
    
    # 7. Total support score
    df['total_support'] = (
        df['tutoring_sessions'] + 
        df['internet_access'] * 3 + 
        df['parental_support_index']
    )
    
    print(f"✅ Features engineered: {df.shape[1]} columns total")
    return df


def encode_and_scale(df, target_col='final_score', save_encoders=True):
    """Encode categoricals, scale numerics, and split data."""
    df = df.copy()
    
    # Drop classification target columns if doing regression (and vice versa)
    drop_cols = ['grade', 'pass_fail']
    df_reg = df.drop(columns=drop_cols, errors='ignore')
    
    # Categorical encoding
    le_dict = {}
    cat_cols = df_reg.select_dtypes(include='object').columns.tolist()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_reg[col] = le.fit_transform(df_reg[col])
        le_dict[col] = le
    
    # Features and target
    X = df_reg.drop(columns=[target_col])
    y = df_reg[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Save encoders + scaler for inference
    if save_encoders:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(le_dict, 'models/label_encoders.pkl')
        joblib.dump(list(X.columns), 'models/feature_names.pkl')
        
        # Also save processed data
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/cleaned_data.csv', index=False)
    
    print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


if __name__ == '__main__':
    df = load_and_clean()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = encode_and_scale(df)
    print(f"✅ Preprocessing complete. Features: {features}")