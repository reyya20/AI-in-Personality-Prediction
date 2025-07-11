🧠 AI-in-Personality-Prediction
AI-in-Personality-Prediction is a machine learning-based project that analyzes handwriting to predict personality traits. Handwriting is a distinctive biometric marker that not only reflects identity but also reveals insights into a person's behavior, emotional state, creativity, honesty, self-perception, and other psychological attributes.

📝 Overview
This project proposes a multi-layered classification approach using a Voting Classifier to analyze handwriting and accurately infer personality characteristics. The Voting Classifier integrates predictions from multiple optimized base models to enhance overall performance.

🔍 Models Used
The ensemble Voting Classifier is built by combining the strengths of the following models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

Gradient Boosting Classifier

Each model is finely tuned using RandomizedSearchCV to optimize key hyperparameters:

Regularization strength (C) for Logistic Regression

Kernel type and gamma for SVM

Number of estimators and depth for Random Forest and Gradient Boosting

📊 Results
✅ Accuracy: 93%

📌 The model shows high consistency across diverse handwriting samples

🔐 Reliable for real-world personality insights based on handwriting analysis

💡 Key Features
Ensemble learning using VotingClassifier for improved generalization

Hyperparameter tuning with RandomizedSearchCV

Support for varied handwriting styles

Potential for integration into psychological and HR tools

🚀 Applications
Personality assessment in HR or recruitment

Psychological analysis and therapy support

Educational behavior tracking

Forensic or behavioral science use cases

Tech Stack
Python

Scikit-learn

Pandas & NumPy

Matplotlib / Seaborn (for visualization)
