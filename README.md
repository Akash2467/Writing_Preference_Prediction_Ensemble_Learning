# Writing_Preference_Prediction_Ensemble_Learning
This project aims to predict the writing preference (left-handed, right-handed, or ambidextrous) of individuals using anthropometric measurements from the ANSUR II Male dataset. The project leverages ensemble machine learning models and statistical feature selection techniques to achieve high prediction accuracy.

Objective
The goal is to explore whether physical body measurements can serve as reliable predictors of a person's writing preference. The task is framed as a multi-class classification problem.

Dataset
Source: ANSUR II - Male Dataset

Records: 4000+ male individuals

Target Variable: WritingPreference

Features: Over 100 anthropometric measurements, including arm length, shoulder width, hand breadth, and more

Methodology
Data Cleaning and Preprocessing
Dropped non-informative columns: Gender, Date, PrimaryMOS, SubjectsBirthLocation, Ethnicity, Branch, Component, and Installation

Encoded the target variable using LabelEncoder

Standardized numeric features using StandardScaler (optional, depending on the model)

Feature Selection
Used Generalized Linear Model (GLM) from the statsmodels library to compute p-values for each feature

Selected features with p-values less than 0.05 as statistically significant

Removed the constant term from the list of significant features before model training

Models Trained
Gradient Boosting Classifier

Accuracy: 87.6%

Stacking Classifier

Base Models: Random Forest, Decision Tree, SVC

Meta Model: Logistic Regression

Accuracy: 87.6%

Naive Bayes Classifier

Accuracy: 87.2%

Bagging Classifier

Base Estimator: Decision Tree

Accuracy: 87.7%

Evaluation Metrics
Accuracy Score

Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

Key Insights
Ensemble models, especially stacking and bagging, performed well in predicting writing preference

Anthropometric features like arm length, shoulder breadth, and hand size showed significant correlation with handedness

Feature selection using p-values helped improve model interpretability and reduce overfitting

Technologies Used
Python

Pandas, NumPy

Scikit-learn

Statsmodels

Matplotlib, Seaborn

Conclusion
The project demonstrates that ensemble machine learning models, combined with statistically selected features, can effectively predict writing preferences from physical measurements. This approach can potentially be extended to other behavioral or biometric predictions using anthropometric data.
