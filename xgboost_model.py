import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv('Verizon Data.csv')

# Define features and target with corrected column name, if necessary
X = data.drop('default', axis=1)  # replace 'default' with the actual name
y = data['default']  # replace 'default' with the actual name

X = pd.get_dummies(X, columns=['gender', 'pmttype'], drop_first=True)

data_balanced = pd.concat([X, y], axis=1)

# Separate majority and minority classes
majority_class = data_balanced[data_balanced['default'] == 0]
minority_class = data_balanced[data_balanced['default'] == 1]

# Oversample minority class to match the number of majority samples
minority_oversampled = minority_class.sample(len(majority_class), replace=True, random_state=42)

# Combine majority class with oversampled minority class
data_balanced = pd.concat([majority_class, minority_oversampled])

# Separate features and target after balancing
X_balanced = data_balanced.drop('default', axis=1)
y_balanced = data_balanced['default']

# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Convert data into DMatrix (optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train model
bst = xgb.train(params, dtrain, num_boost_round=100)

# Predict on the test set
y_pred = bst.predict(dtest)
y_pred_class = [1 if y > 0.5 else 0 for y in y_pred]