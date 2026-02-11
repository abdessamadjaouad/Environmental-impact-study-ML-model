"""
EIE Impact Assessment — ML Model Training & Comparison
=======================================================
Trains and compares 4 classifiers on the EIE unified dataset:
  1. Decision Tree
  2. Random Forest
  3. XGBoost
  4. Neural Network (MLP)

Produces:
  - Model comparison table
  - Confusion matrices
  - Feature importance charts
  - Saved best model (.joblib)

Designed to run locally or on Kaggle.

Usage:
    python train_models.py
    python train_models.py --dataset eie_unified.csv
    python train_models.py --dataset eie_eau.csv --component-mode
"""

import os
import sys
import argparse
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available — skipping XGB model")

warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare(csv_path, component_mode=False):
    """
    Load dataset and prepare features/targets.
    
    Args:
        csv_path: Path to CSV dataset
        component_mode: If True, use per-component detailed features
    
    Returns:
        X, y, feature_names, label_encoder
    """
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['importance_relative'].value_counts()}\n")
    
    # Target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['importance_relative'])
    
    print(f"Classes: {list(le_target.classes_)}")
    print(f"Encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}\n")
    
    # Features — encode categoricals
    # Drop intermediate target columns (they leak the answer)
    drop_cols = [
        'importance_relative', 'importance', 'sensitivity',
        'impact_apprehende', 'value_initial', 'intensity_class'
    ]
    
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Encode categorical columns
    le_dict = {}
    for col in feature_df.select_dtypes(include='object').columns:
        le_dict[col] = LabelEncoder()
        feature_df[col] = le_dict[col].fit_transform(feature_df[col].astype(str))
    
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)
    
    # Handle any NaN
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Features ({len(feature_names)}): {feature_names[:15]}{'...' if len(feature_names) > 15 else ''}")
    
    return X, y, feature_names, le_target, le_dict


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    """Define all models to train."""
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        ),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
    
    return models


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(models, X_train, X_test, y_train, y_test,
                       feature_names, le_target, output_dir):
    """Train all models and evaluate."""
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Scale features for MLP
        if "MLP" in name or "Neural" in name:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
        else:
            X_tr = X_train
            X_te = X_test
            scaler = None
        
        # Train
        t0 = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - t0
        
        # Predict
        t0 = time.time()
        y_pred = model.predict(X_te)
        pred_time = time.time() - t0
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': f"{acc:.4f}",
            'Precision': f"{prec:.4f}",
            'Recall': f"{rec:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Train Time (s)': f"{train_time:.2f}",
            'Predict Time (s)': f"{pred_time:.4f}",
        })
        
        trained_models[name] = {
            'model': model,
            'scaler': scaler,
            'accuracy': acc,
            'f1': f1,
        }
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Time:      {train_time:.2f}s train, {pred_time:.4f}s predict")
        
        # Classification report
        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred, 
                                       target_names=le_target.classes_,
                                       digits=4)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, le_target.classes_, name, output_dir)
    
    return results, trained_models


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    fname = f"cm_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {fname}")


def plot_feature_importance(trained_models, feature_names, output_dir):
    """Plot feature importance for tree-based models."""
    
    for name, data in trained_models.items():
        model = data['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Top 15 features
            top_n = min(15, len(feature_names))
            top_idx = indices[:top_n]
            top_names = [feature_names[i] for i in top_idx]
            top_values = importances[top_idx]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
            bars = ax.barh(range(top_n), top_values[::-1], color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_names[::-1])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Features — {name}', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, top_values[::-1]):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            fname = f"fi_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Feature importance saved: {fname}")
            
            # Print top features
            print(f"\n{'='*50}")
            print(f"Top Features — {name}")
            print(f"{'='*50}")
            for i, idx in enumerate(top_idx):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def plot_model_comparison(results_df, output_dir):
    """Plot model comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['Accuracy', 'F1-Score']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for ax, metric in zip(axes, metrics):
        values = results_df[metric].astype(float).values
        models = results_df['Model'].values
        
        bars = ax.bar(range(len(models)), values, 
                      color=colors[:len(models)], alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Model Comparison — EIE Impact Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Model comparison chart saved: model_comparison.png")


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(models, X, y, cv=5):
    """Run cross-validation for all models."""
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({cv}-fold)")
    print(f"{'='*60}")
    
    cv_results = []
    
    for name, model in models.items():
        print(f"\n  {name}...")
        
        if "MLP" in name or "Neural" in name:
            # Scale for MLP
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_results.append({
            'Model': name,
            'CV Mean': f"{scores.mean():.4f}",
            'CV Std': f"{scores.std():.4f}",
            'CV Min': f"{scores.min():.4f}",
            'CV Max': f"{scores.max():.4f}",
        })
        
        print(f"    Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"    Range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    return cv_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EIE ML models")
    parser.add_argument("--dataset", type=str, default="eie_unified.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--component-mode", action="store_true",
                        help="Use per-component dataset (detailed features)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for charts and models")
    
    args = parser.parse_args()
    
    # Setup output directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(base_dir, dataset_path)
    
    print("=" * 60)
    print("EIE IMPACT ASSESSMENT — ML MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset:    {dataset_path}")
    print(f"Test size:  {args.test_size}")
    print(f"Output:     {output_dir}")
    print(f"CV folds:   {args.cv}")
    print("=" * 60)
    
    # Load data
    X, y, feature_names, le_target, le_dict = load_and_prepare(
        dataset_path, args.component_mode
    )
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {X_train.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Get models
    models = get_models()
    
    # Train & evaluate
    results, trained_models = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        feature_names, le_target, output_dir
    )
    
    # Results table
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Visualizations
    plot_model_comparison(results_df, output_dir)
    plot_feature_importance(trained_models, feature_names, output_dir)
    
    # Cross-validation
    if not args.skip_cv:
        cv_results = run_cross_validation(get_models(), X, y, cv=args.cv)
        cv_df = pd.DataFrame(cv_results)
        print(f"\n{cv_df.to_string(index=False)}")
        cv_df.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    
    # Save best model
    best_name = max(trained_models, key=lambda k: trained_models[k]['f1'])
    best_data = trained_models[best_name]
    
    model_path = os.path.join(output_dir, 'best_model.joblib')
    joblib.dump({
        'model': best_data['model'],
        'scaler': best_data['scaler'],
        'feature_names': feature_names,
        'label_encoder': le_target,
        'categorical_encoders': le_dict,
        'accuracy': best_data['accuracy'],
        'f1': best_data['f1'],
        'model_name': best_name,
    }, model_path)
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   Accuracy: {best_data['accuracy']:.4f}")
    print(f"   F1-Score: {best_data['f1']:.4f}")
    print(f"   Saved to: {model_path}")
    print(f"{'='*60}")
    
    # Summary
    print(f"\n📁 Output files in: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"   {f} ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
