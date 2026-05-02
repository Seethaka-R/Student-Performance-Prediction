"""
data_generator.py
Generates realistic synthetic student performance data.
Run this first to create your dataset.
"""

import numpy as np
import pandas as pd
import os

def generate_student_data(n_students=1200, random_seed=42):
    """Generate synthetic student performance dataset."""
    np.random.seed(random_seed)
    
    n = n_students
    
    # ── Demographics ──────────────────────────────────────────────
    gender = np.random.choice(['Male', 'Female'], n)
    age = np.random.randint(15, 19, n)
    school_type = np.random.choice(['Public', 'Private'], n, p=[0.65, 0.35])
    
    # ── Family background ─────────────────────────────────────────
    parent_education = np.random.choice(
        ['No Education', 'High School', 'Graduate', 'Postgraduate'], 
        n, p=[0.10, 0.35, 0.40, 0.15]
    )
    family_income = np.random.choice(['Low', 'Middle', 'High'], n, p=[0.30, 0.50, 0.20])
    
    # ── Academic inputs ───────────────────────────────────────────
    # Study hours: 1-8 hours/day (normally distributed around 3.5)
    study_hours = np.clip(np.random.normal(3.5, 1.5, n), 0.5, 10)
    
    # Attendance: 50-100%, correlated with study hours
    attendance_base = np.clip(np.random.normal(75, 15, n), 40, 100)
    attendance = np.clip(attendance_base + (study_hours - 3.5) * 3, 40, 100)
    
    # Previous term GPA: 4.0 to 10.0 scale
    prev_gpa = np.clip(np.random.normal(6.5, 1.8, n), 3.0, 10.0)
    
    # Assignments completed (out of 20)
    assignments_done = np.clip(
        np.random.normal(15, 3, n) + (study_hours - 3.5) * 1.2, 0, 20
    ).astype(int)
    
    # Extracurricular activities: 0-5 activities
    extracurricular = np.random.randint(0, 6, n)
    
    # Internet access
    internet_access = np.random.choice([0, 1], n, p=[0.25, 0.75])
    
    # Tutoring sessions per month
    tutoring = np.clip(np.random.randint(0, 8, n), 0, 7)
    
    # Absences per semester
    absences = np.clip(
        np.random.normal(8, 5, n) - (attendance - 75) * 0.3, 0, 30
    ).astype(int)
    
    # Sleep hours per night
    sleep_hours = np.clip(np.random.normal(7, 1.2, n), 4, 10)
    
    # ── Target variable: Final Score (0–100) ──────────────────────
    # Built from a realistic combination of features
    parent_edu_map = {'No Education': 0, 'High School': 1, 'Graduate': 2, 'Postgraduate': 3}
    income_map = {'Low': 0, 'Middle': 1, 'High': 2}
    
    parent_edu_num = np.array([parent_edu_map[p] for p in parent_education])
    income_num = np.array([income_map[f] for f in family_income])
    
    final_score = (
        prev_gpa * 5.0
        + study_hours * 4.0
        + attendance * 0.25
        + assignments_done * 1.2
        + tutoring * 1.5
        + parent_edu_num * 2.0
        + income_num * 1.5
        + internet_access * 3.0
        - absences * 0.8
        + sleep_hours * 0.5
        + np.random.normal(0, 4, n)   # noise
    )
    
    # Normalize to 0–100
    final_score = np.clip(final_score, 0, 100)
    final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 100
    final_score = np.round(final_score, 1)
    
    # ── Grade label ───────────────────────────────────────────────
    def assign_grade(score):
        if score >= 90: return 'A'
        elif score >= 75: return 'B'
        elif score >= 60: return 'C'
        elif score >= 45: return 'D'
        else: return 'F'
    
    grade = [assign_grade(s) for s in final_score]
    
    # ── Pass/Fail binary ──────────────────────────────────────────
    pass_fail = ['Pass' if s >= 45 else 'Fail' for s in final_score]
    
    # ── Assemble dataframe ────────────────────────────────────────
    df = pd.DataFrame({
        'student_id': [f'STU{i:04d}' for i in range(1, n+1)],
        'age': age,
        'gender': gender,
        'school_type': school_type,
        'parent_education': parent_education,
        'family_income': family_income,
        'study_hours_per_day': np.round(study_hours, 2),
        'attendance_percent': np.round(attendance, 1),
        'prev_gpa': np.round(prev_gpa, 2),
        'assignments_completed': assignments_done,
        'extracurricular_activities': extracurricular,
        'internet_access': internet_access,
        'tutoring_sessions': tutoring,
        'absences': absences,
        'sleep_hours': np.round(sleep_hours, 1),
        'final_score': final_score,
        'grade': grade,
        'pass_fail': pass_fail
    })
    
    return df


if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    df = generate_student_data(1200)
    df.to_csv('data/raw/student_data.csv', index=False)
    print(f"✅ Dataset created: {len(df)} students")
    print(f"   Grade distribution:\n{df['grade'].value_counts()}")
    print(f"   Pass rate: {(df['pass_fail']=='Pass').mean()*100:.1f}%")
    print(f"   Avg final score: {df['final_score'].mean():.1f}")