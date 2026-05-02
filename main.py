"""
main.py
Complete pipeline: generate data → preprocess → EDA → train → save.
Run this once to set up the entire project.
"""

import os
import sys

# Ensure `src/` is on sys.path so we can import local modules when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("   STUDENT PERFORMANCE PREDICTION SYSTEM")
print("   Complete ML Pipeline")
print("=" * 60)

# Step 1: Generate data
print("\n[1/5] Generating synthetic student dataset...")
from data_generator import generate_student_data

os.makedirs('data/raw', exist_ok=True)
df = generate_student_data(1200)
df.to_csv('data/raw/student_data.csv', index=False)
print(f"      ✅ {len(df)} student records created")

# Step 2: Preprocess
print("\n[2/5] Cleaning and feature engineering...")
from preprocessing import load_and_clean, engineer_features, encode_and_scale

df_clean = load_and_clean('data/raw/student_data.csv')
df_clean = engineer_features(df_clean)
X_train, X_test, y_train, y_test, features = encode_and_scale(df_clean)

# Step 3: EDA
print("\n[3/5] Running EDA and generating plots...")
from eda import run_eda
run_eda('data/processed/cleaned_data.csv')

# Step 4: Train models
print("\n[4/5] Training ML models...")
from train_model import train_all_models, save_best_model, plot_results

results, trained_models = train_all_models(X_train, X_test, y_train, y_test)
best_name, best_model = save_best_model(results, trained_models)
plot_results(results, features, best_name, X_test, y_test)

# Step 5: Test prediction
print("\n[5/5] Testing prediction pipeline...")
from predict import predict_student

test_student = {
    'age': 17, 'gender': 'Female', 'school_type': 'Public',
    'parent_education': 'Graduate', 'family_income': 'Middle',
    'study_hours_per_day': 4.5, 'attendance_percent': 82.0,
    'prev_gpa': 7.5, 'assignments_completed': 16,
    'extracurricular_activities': 2, 'internet_access': 1,
    'tutoring_sessions': 3, 'absences': 5, 'sleep_hours': 7.5
}
score, grade, risk = predict_student(test_student)

print(f"\n{'='*60}")
print("   PIPELINE COMPLETE!")
print(f"   Best model: {best_name}")
print(f"   Test prediction: Score={score}, Grade={grade}, Risk={risk}")
print(f"{'='*60}")
print("\n🚀 Launch the dashboard:")
print("   streamlit run app/streamlit_app.py")