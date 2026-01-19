import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# 1. Prepare Target and Features
# Assuming df2 is your dummied dataframe from the lab
y = df2['generous']
X = df2.drop('generous', axis=1)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. Instantiate and Fit Champion Model
# Using the best hyperparameters found during GridSearchCV
rf_champion = RandomForestClassifier(
    max_depth=None,
    max_features=1.0,
    max_samples=0.7,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=300,
    random_state=42,
    n_jobs=-1 # Uses all processors for speed
)

rf_champion.fit(X_train, y_train)

# 4. Evaluation
preds = rf_champion.predict(X_test)

print(f"F1 Score: {f1_score(y_test, preds):.4f}")
print(f"Recall: {recall_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds):.4f}")
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# 5. Visualizing Feature Importance
importances = pd.Series(rf_champion.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
importances.plot(kind='bar', title='Top 10 Predictors of Generous Tipping')
