# disease-prediction-ml
Multi-Disease Chronic Condition Risk Prediction 
# Integrative Multi-Disease Risk Prediction using Deep Learning and Ensemble Methods

This repository contains the code and resources for the research paper titled "A Multi-Disease Deep Learning Risk Prediction Model Using Heterogeneous Clinical Data: An Integrative Approach to Chronic Disease Screening."

The project focuses on developing and evaluating robust machine learning models to predict the likelihood of several common chronic conditions:
*   Diabetes (using the PIMA Indians Diabetes dataset)
*   Heart Disease (using the UCI Cleveland Heart Disease dataset)
*   Chronic Kidney Disease (CKD) (using the UCI CKD dataset)

We implement and compare four distinct predictive algorithms:
1.  **Random Forest:** A powerful ensemble tree-based method.
2.  **XGBoost:** An efficient gradient boosting algorithm.
3.  **Multi-Layer Perceptron (MLP):** A feedforward deep neural network.
4.  **TabTransformer (Conceptual):** A Transformer-based architecture adapted for tabular data, explored for its potential in handling categorical features and complex interactions.

**Workflow Includes:**
*   Comprehensive data loading and cleaning tailored to each dataset.
*   Standardized preprocessing steps including missing value imputation and feature scaling.
*   Hyperparameter tuning for each model using GridSearchCV or manual search strategies.
*   Rigorous model evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and AUC.
*   Generation of visualizations: ROC curves, confusion matrices, and feature importance plots to interpret model behavior.

The goal is to identify high-performing models for each disease and discuss the potential of an integrative framework for early and comprehensive chronic disease screening.

**Technologies Used:**
*   Python
*   Pandas & NumPy for data manipulation
*   Scikit-learn for traditional ML models, preprocessing, and metrics
*   TensorFlow & Keras for deep learning models (MLP and TabTransformer)
*   XGBoost library
*   Matplotlib & Seaborn for visualizations
