# Predictive Analytics Model for House Price Forecasting
# Task: Develop a predictive analytics model to forecast future trends based on historical data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=== PREDICTIVE ANALYTICS MODEL ===")
print("Task: House Price Forecasting using Historical Data\n")

# =============================================================================
# 1. DATA PREPARATION: Clean and preprocess historical data
# =============================================================================
print("1. DATA PREPARATION")
print("-" * 50)

# Load datasets
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
except FileNotFoundError:
    print("⚠ Files not found. Please ensure 'train.csv' and 'test.csv' are available.")
    print("Using sample data for demonstration...")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    train_df = pd.DataFrame({
        'Id': range(1, n_samples + 1),
        'feature1': np.random.normal(50, 15, n_samples),
        'feature2': np.random.normal(100, 30, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'SalePrice': np.random.normal(200000, 50000, n_samples)
    })
    test_df = pd.DataFrame({
        'Id': range(n_samples + 1, n_samples + 501),
        'feature1': np.random.normal(50, 15, 500),
        'feature2': np.random.normal(100, 30, 500),
        'feature3': np.random.choice(['A', 'B', 'C'], 500)
    })

# Data exploration
print(f"\nInitial data overview:")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Target variable (SalePrice) statistics:")
if 'SalePrice' in train_df.columns:
    print(train_df['SalePrice'].describe())

# Combine datasets for unified preprocessing
print("\n2. Data Preprocessing:")
train_df["is_train"] = True
test_df["is_train"] = False
combined_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

# Handle missing values
print(f"Missing values before cleaning:")
missing_before = combined_df.isnull().sum().sum()
print(f"Total missing values: {missing_before}")

# Remove columns with more than 50% missing values
missing_threshold = 0.5
high_missing_cols = combined_df.columns[combined_df.isnull().mean() > missing_threshold]
if len(high_missing_cols) > 0:
    print(f"Dropping {len(high_missing_cols)} columns with >50% missing values")
    combined_df.drop(columns=high_missing_cols, inplace=True)

# Fill missing values
# For categorical variables: use mode
categorical_cols = combined_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if combined_df[col].isnull().sum() > 0:
        mode_value = combined_df[col].mode()
        if len(mode_value) > 0:
            combined_df[col].fillna(mode_value[0], inplace=True)

# For numerical variables: use median
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if combined_df[col].isnull().sum() > 0:
        combined_df[col].fillna(combined_df[col].median(), inplace=True)

print(f"Missing values after cleaning: {combined_df.isnull().sum().sum()}")

# Encode categorical variables
categorical_cols = combined_df.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop(['is_train'], errors='ignore')

if len(categorical_cols) > 0:
    print(f"Encoding {len(categorical_cols)} categorical variables...")
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Separate back into train and test
train_processed = combined_df[combined_df["is_train"] == True].drop(columns=["is_train"])
test_processed = combined_df[combined_df["is_train"] == False].drop(columns=["is_train"])

# Remove target variable from test set if it exists
if 'SalePrice' in test_processed.columns:
    test_processed = test_processed.drop(columns=['SalePrice'])

print(f"✓ Data preprocessing completed")
print(f"Final training set shape: {train_processed.shape}")
print(f"Final test set shape: {test_processed.shape}")

# =============================================================================
# 2. MODEL SELECTION: Choose appropriate machine learning models
# =============================================================================
print(f"\n2. MODEL SELECTION")
print("-" * 50)

# Prepare features and target
if 'SalePrice' in train_processed.columns:
    X = train_processed.drop(columns=['SalePrice'])
    y = train_processed['SalePrice']
else:
    print("⚠ Target variable 'SalePrice' not found. Using synthetic target.")
    X = train_processed.drop(columns=['Id'], errors='ignore')
    y = np.random.normal(200000, 50000, len(train_processed))

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print(f"\nTesting {len(models)} different models...")

# =============================================================================
# 3. MODEL TRAINING: Train models and validate performance
# =============================================================================
print(f"\n3. MODEL TRAINING & VALIDATION")
print("-" * 50)

model_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    model_results[name] = {
        'model': model,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'val_mae': val_mae
    }
    
    print(f"  Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std()*2:.2f})")
    print(f"  Validation RMSE: {val_rmse:.2f}")
    print(f"  Validation R²: {val_r2:.3f}")
    print(f"  Validation MAE: {val_mae:.2f}")

# Select best model
best_model_name = min(model_results.keys(), 
                     key=lambda x: model_results[x]['val_rmse'])
best_model = model_results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"Validation RMSE: {model_results[best_model_name]['val_rmse']:.2f}")
print(f"Validation R²: {model_results[best_model_name]['val_r2']:.3f}")

# Hyperparameter tuning for best model (if it's Random Forest)
if best_model_name == 'Random Forest':
    print(f"\nPerforming hyperparameter tuning for Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Re-evaluate with tuned model
    y_val_pred_tuned = best_model.predict(X_val)
    tuned_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
    tuned_r2 = r2_score(y_val, y_val_pred_tuned)
    
    print(f"Tuned model RMSE: {tuned_rmse:.2f}")
    print(f"Tuned model R²: {tuned_r2:.3f}")
    print(f"Best parameters: {grid_search.best_params_}")

# =============================================================================
# 4. PREDICTIONS AND INSIGHTS
# =============================================================================
print(f"\n4. PREDICTIONS & INSIGHTS")
print("-" * 50)

# Make predictions on test set
X_test = test_processed.drop(columns=['Id'], errors='ignore')

# Ensure test set has same features as training set
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

extra_cols = set(X_test.columns) - set(X_train.columns)
for col in extra_cols:
    X_test = X_test.drop(columns=[col])

X_test = X_test[X_train.columns]  # Ensure same order

test_predictions = best_model.predict(X_test)

print(f"Test predictions generated: {len(test_predictions)} samples")
print(f"Prediction statistics:")
print(f"  Mean: ${np.mean(test_predictions):,.2f}")
print(f"  Median: ${np.median(test_predictions):,.2f}")
print(f"  Std: ${np.std(test_predictions):,.2f}")
print(f"  Min: ${np.min(test_predictions):,.2f}")
print(f"  Max: ${np.max(test_predictions):,.2f}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

# Create submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission file created: submission.csv")

# =============================================================================
# 5. REPORTING: Model accuracy and insights
# =============================================================================
print(f"\n5. MODEL PERFORMANCE REPORT")
print("=" * 50)

print(f"📊 FINAL MODEL PERFORMANCE SUMMARY")
print(f"{'='*50}")
print(f"Selected Model: {best_model_name}")
print(f"Training Dataset Size: {X_train.shape[0]:,} samples")
print(f"Validation Dataset Size: {X_val.shape[0]:,} samples")
print(f"Number of Features: {X_train.shape[1]}")
print(f"")
print(f"📈 ACCURACY METRICS:")
print(f"  • Root Mean Square Error (RMSE): ${model_results[best_model_name]['val_rmse']:,.2f}")
print(f"  • Mean Absolute Error (MAE): ${model_results[best_model_name]['val_mae']:,.2f}")
print(f"  • R-squared (R²): {model_results[best_model_name]['val_r2']:.3f}")
print(f"  • Cross-validation RMSE: ${model_results[best_model_name]['cv_rmse_mean']:,.2f} ± ${model_results[best_model_name]['cv_rmse_std']*2:,.2f}")
print(f"")
print(f"🎯 MODEL INSIGHTS:")
print(f"  • The model explains {model_results[best_model_name]['val_r2']*100:.1f}% of the variance in house prices")
print(f"  • Average prediction error: ${model_results[best_model_name]['val_mae']:,.0f}")
print(f"  • Model shows {'good' if model_results[best_model_name]['val_r2'] > 0.8 else 'moderate' if model_results[best_model_name]['val_r2'] > 0.6 else 'limited'} predictive power")
print(f"")
print(f"🔮 PREDICTIONS SUMMARY:")
print(f"  • Test set size: {len(test_predictions):,} properties")
print(f"  • Average predicted price: ${np.mean(test_predictions):,.0f}")
print(f"  • Price range: ${np.min(test_predictions):,.0f} - ${np.max(test_predictions):,.0f}")
print(f"")
print(f"📋 RECOMMENDATIONS:")
if model_results[best_model_name]['val_r2'] > 0.8:
    print(f"  • Model shows excellent performance and is ready for deployment")
elif model_results[best_model_name]['val_r2'] > 0.6:
    print(f"  • Model shows good performance but could benefit from feature engineering")
else:
    print(f"  • Model needs improvement - consider additional features or different algorithms")
print(f"  • Consider collecting more data for better predictions")
print(f"  • Monitor model performance over time and retrain as needed")
print(f"")
print(f"✅ Analysis completed successfully!")
print(f"📁 Results saved to: submission.csv")