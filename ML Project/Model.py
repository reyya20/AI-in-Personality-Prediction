import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

# Define the file path
file_path = r'C:\Users\User\Desktop\Paper work\handwriting_features_all_traits.txt'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist.")
else:
    # Load data from the text file
    data = pd.read_csv(file_path)

    # Strip extra spaces from column names
    data.columns = data.columns.str.strip()

    # Separate features and target
    X = data[['Size', 'Slant', 'Avg Distance Between Characters']]
    y = data['Trait']

    # Convert 'Slant' column from string representation to numerical values
    def convert_slant(slant_str):
        match = re.search(r'\d+\.?\d*', slant_str)
        if match:
            return float(match.group(0))
        else:
            return None

    X.loc[:, 'Slant'] = X['Slant'].apply(lambda x: convert_slant(str(x)))
    X = X.dropna()

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Define base models
    log_reg = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
    svm = SVC(kernel='linear', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('rf', rf), ('gb', gb)], voting='soft')

    # Randomized Search
    param_distributions = {
        'lr__C': [0.01, 0.1, 1, 10],
        'svc__C': [0.01, 0.1, 1, 10],
        'rf__n_estimators': [50, 100, 200],
        'gb__n_estimators': [50, 100, 200]
    }

    print("Starting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(voting_clf, param_distributions, n_iter=20, cv=3, scoring='accuracy', verbose=3, n_jobs=-1, random_state=42)
    random_search.fit(X_train_res, y_train_res)
    print("RandomizedSearchCV completed.")

    best_model = random_search.best_estimator_

    # Predict and evaluate
    y_pred_best = best_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_best)
    print(f'Accuracy: {accuracy}')

    # Classification Report
    print(classification_report(y_test, y_pred_best))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC-AUC Score (only for binary classification)
    if len(set(y_test)) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f'ROC-AUC Score: {roc_auc}')

    # Save the best model
    model_path = r'C:\Users\User\Desktop\handwriting_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
