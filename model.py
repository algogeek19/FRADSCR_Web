import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load data
df = pd.read_csv('data/FRADSCR.csv')
df['date'] = pd.to_datetime(df['start.date'])

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Define features and targets
features = ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'duration.s', 'peak.c/s', 'total.counts', 'x.pos.asec', 'y.pos.asec', 'radial']
X = df[features]
y = df[['duration.s', 'peak.c/s', 'total.counts', 'x.pos.asec', 'y.pos.asec', 'radial']]

# Preprocessing
numeric_features = ['year', 'month', 'day', 'duration.s', 'peak.c/s', 'total.counts', 'x.pos.asec', 'y.pos.asec', 'radial']
categorical_features = ['day_of_week', 'is_weekend']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Setup model pipeline
model = XGBRegressor(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# Define parameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__subsample': [0.7, 0.9]
}

# Time series split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform grid search with time-based cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)

# Flask routes would remain similar but use the new model for predictions

from sklearn.metrics import r2_score

# Assuming you've already trained the model with grid_search

# Best model from grid search
best_model = grid_search.best_estimator_

# Fit the best model to the entire dataset again to get train R-squared (though typically you'd use your training set)
best_model.fit(X, y)
y_pred_train = best_model.predict(X)
train_r2 = r2_score(y, y_pred_train)
print(f'Training R-squared: {train_r2:.4f}')

# Now for test R-squared, you need to split your data into train and test again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on test data
y_pred_test = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred_test)
print(f'Test R-squared: {test_r2:.4f}')

if __name__ == '__main__':
    app.run(debug=False)


