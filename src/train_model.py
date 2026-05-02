"""
train_model.py
Trains multiple ML models, evaluates them, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor

# Import our modules (ensure src directory is on sys.path)
import sys
import os
# Add the current file's directory (src/) to sys.path so we can import local modules
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_clean, engineer_features, encode_and_scale


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Train and evaluate a single model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    # Cross-validation R²
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_r2 = cv_scores.mean()
    
    return {
        'Model': name,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'R²': round(r2, 4),
        'CV R² (mean)': round(cv_r2, 4),
        'Predictions': y_pred,
        'model_obj': model
    }


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all candidate models and return results."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=42, verbosity=0
        ),
    }
    
    results = []
    trained = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        trained[name] = result['model_obj']
        print(f"     RMSE={result['RMSE']}, R²={result['R²']}, CV R²={result['CV R² (mean)']}")
    
    return results, trained


def plot_results(results, feature_names, best_model_name, X_test, y_test):
    """Generate and save all evaluation plots."""
    os.makedirs('outputs', exist_ok=True)
    
    # ── 1. Model comparison bar chart ────────────────────────────
    df_res = pd.DataFrame(results).drop(columns=['Predictions', 'model_obj'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    metrics = ['RMSE', 'MAE', 'R²']
    colors = ['#EF9F27', '#1D9E75', '#7F77DD']
    
    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(df_res['Model'], df_res[metric], color=color, alpha=0.85, edgecolor='white')
        ax.set_title(metric, fontweight='bold')
        ax.set_xlabel(metric)
        for i, v in enumerate(df_res[metric]):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/model_comparison.png")
    
    # Save model comparison table
    df_res.to_csv('outputs/model_comparison.csv', index=False)
    
    # ── 2. Actual vs Predicted scatter ────────────────────────────
    best_result = next(r for r in results if r['Model'] == best_model_name)
    y_pred = best_result['Predictions']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color='#7F77DD', s=30, edgecolors='none')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title(f'Actual vs Predicted — {best_model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/actual_vs_predicted.png")
    
    # ── 3. Feature importance ─────────────────────────────────────
    best_models = {r['Model']: r['model_obj'] for r in results}
    best_obj = best_models[best_model_name]
    
    if hasattr(best_obj, 'feature_importances_'):
        imp = pd.Series(best_obj.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=True).tail(15)
        
        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(imp.index, imp.values, color='#1D9E75', alpha=0.85, edgecolor='white')
        ax.set_title(f'Top Feature Importances — {best_model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: outputs/feature_importance.png")
    
    # ── 4. Residuals distribution ─────────────────────────────────
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color='#EF9F27', alpha=0.8, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', lw=2)
    ax.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual (Actual − Predicted)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/residuals.png")


def save_best_model(results, trained_models):
    """Save the best model based on R²."""
    best = max(results, key=lambda x: x['R²'])
    best_name = best['Model']
    best_model = trained_models[best_name]
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print(f"\n🏆 Best model: {best_name}")
    print(f"   RMSE: {best['RMSE']}, MAE: {best['MAE']}, R²: {best['R²']}")
    print(f"   ✅ Saved to: models/best_model.pkl")
    
    return best_name, best_model


if __name__ == '__main__':
    print("📊 Loading and preprocessing data...")
    df = load_and_clean()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, features = encode_and_scale(df)
    
    print("\n🤖 Training models...")
    results, trained_models = train_all_models(X_train, X_test, y_train, y_test)
    
    print("\n📈 Generating evaluation plots...")
    best_name, best_model = save_best_model(results, trained_models)
    plot_results(results, features, best_name, X_test, y_test)
    
    print("\n✅ Training complete! Check outputs/ folder for plots.")