"""
eda.py
Generates all EDA plots and saves them to outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath='data/processed/cleaned_data.csv'):
    os.makedirs('outputs', exist_ok=True)
    df = pd.read_csv(filepath)
    
    print("📊 Running EDA...")
    
    # ── 1. Grade distribution ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    grade_order = ['A', 'B', 'C', 'D', 'F']
    colors = ['#1D9E75', '#7F77DD', '#EF9F27', '#D85A30', '#E24B4A']
    
    grade_counts = df['grade'].value_counts().reindex(grade_order)
    axes[0].bar(grade_order, grade_counts, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Grade Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Grade')
    axes[0].set_ylabel('Number of Students')
    for i, v in enumerate(grade_counts):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    axes[1].pie(
        grade_counts, labels=grade_order, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    axes[1].set_title('Grade Share', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/grade_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── 2. Correlation heatmap ────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.drop(
        ['final_score'], errors='ignore'
    )
    corr_with_target = df[num_cols].corrwith(df['final_score']).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = ['#1D9E75' if c > 0 else '#E24B4A' for c in corr_with_target]
    ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1],
            color=colors_bar[::-1], edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('Feature Correlation with Final Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('outputs/correlation_with_target.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Full correlation heatmap
    corr_matrix = df[num_cols.tolist() + ['final_score']].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0, ax=ax,
        linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    ax.set_title('Correlation Heatmap (All Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── 3. Study hours vs Score ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(
        df['study_hours_per_day'], df['final_score'],
        alpha=0.4, c=df['final_score'], cmap='RdYlGn', s=20, edgecolors='none'
    )
    axes[0].set_xlabel('Study Hours per Day')
    axes[0].set_ylabel('Final Score')
    axes[0].set_title('Study Hours vs Final Score', fontsize=13, fontweight='bold')
    
    axes[1].scatter(
        df['attendance_percent'], df['final_score'],
        alpha=0.4, c=df['final_score'], cmap='RdYlGn', s=20, edgecolors='none'
    )
    axes[1].set_xlabel('Attendance (%)')
    axes[1].set_ylabel('Final Score')
    axes[1].set_title('Attendance vs Final Score', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── 4. Score distribution by categorical variables ────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cat_vars = ['gender', 'school_type', 'family_income', 'parent_education']
    grade_order_cat = ['Low', 'Middle', 'High']
    
    for ax, col in zip(axes.flat, cat_vars):
        order = df[col].value_counts().index.tolist()
        sns.boxplot(
            data=df, x=col, y='final_score', order=order,
            palette='Set2', ax=ax, linewidth=0.8
        )
        ax.set_title(f'Score by {col.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Final Score')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/categorical_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ EDA complete. All plots saved to outputs/")


if __name__ == '__main__':
    run_eda()