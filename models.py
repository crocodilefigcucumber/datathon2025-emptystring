import pandas as pd
import os
import numpy as np
import joblib
from datetime import datetime
import json

# Scikit-learn utilities
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Base models
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC

# Additional diverse models for tabular data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# For randomness in hyperparameter distributions and for ensemble uncertainty
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from scipy.stats import mode

SEED = 1337
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

from utilities.evaluate import evaluate
from collect_data_new import collect_enriched, load_or_create
from get_clean_dataframe import clean_dataframe

from rules.trusted import RM_contact
from rules.passportdate import check_passport_expiry
from rules.names_check import check_names
from rules.consistency import check_inconsistency
from rules.email_check import check_email_name

import sys

if __name__ == "__main__":

    rules = [check_passport_expiry, RM_contact, check_inconsistency, check_names]
    embedding = True
    # ------------------------------------------------------------------------------
    # 1. Load pre-split datasets
    # ------------------------------------------------------------------------------

    # Read mode from flags
    mode = "train"
    filename =  "enriched_" + mode + ".csv"

    data = load_or_create(filename=filename,
                          rules=rules,
                          embedding=5,
                          mode="train",
                          llm = True)
    val_df = load_or_create(filename="enriched_val.csv",
                            rules=rules,
                            embedding=5,
                            mode="val",
                            llm=True)

    train_df = clean_dataframe(data)
    val_df = clean_dataframe(val_df)
    """  
    categorical_features = [
        "inheritance_details_relationship", "investment_risk_profile",
        "investment_horizon", "investment_experience", "currency"
    ]
    for rule in rules:
        categorical_features.append(rule.__name__)

    numerical_features = [
        "aum_savings", "aum_inheritance", "aum_real_estate_value"
    ]
    if embedding:
        numerical_features += ["pc_1", "pc_2", "pc_3", "pc_4", "pc_5"]


    # ------------------------------------------------------------------------------
    # 2. Define feature lists and target
    # ------------------------------------------------------------------------------
    # Provided lists of features

    # Define the target column (assumed to be in "label_label")
    target_col = "label_label"

    # Convert target to binary: 1 if "Reject", 0 if "Accept" (adjust as needed)
    train_df[target_col] = (train_df[target_col] == "Reject").astype(int)
    val_df[target_col]   = (val_df[target_col]   == "Reject").astype(int)

    # For modeling, select only the features we want
    feature_columns = numerical_features + categorical_features

    X_train = train_df[feature_columns]
    y_train = train_df[target_col]

    X_val = val_df[feature_columns]
    y_val = val_df[target_col]
    # ------------------------------------------------------------------------------
    # 3. Build preprocessing pipelines for numerical and categorical features
    # ------------------------------------------------------------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # ------------------------------------------------------------------------------
    # 4. Define candidate models (diverse and high-performing on tabular data)
    # ------------------------------------------------------------------------------
    models = {
        #"RandomForest":    RandomForestClassifier(random_state=SEED),
        #"LightGBM":        LGBMClassifier(random_state=SEED),
        #"Lasso":           LogisticRegression(penalty='l1', solver='liblinear', random_state=SEED),
        #"Ridge":           RidgeClassifier(random_state=SEED),
        "SVM":             SVC(probability=True, random_state=SEED),
        #"KNN":             KNeighborsClassifier(),
        #"GradientBoosting":GradientBoostingClassifier(random_state=SEED),
        #"AdaBoost":        AdaBoostClassifier(random_state=SEED),
        #"ExtraTrees":      ExtraTreesClassifier(random_state=SEED),
        #"XGBoost":         XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss')
    }

    # ------------------------------------------------------------------------------
    # 5. Define hyperparameter distributions for RandomizedSearchCV for each model
    # ------------------------------------------------------------------------------
    param_distributions = {
        "RandomForest": {
            "classifier__n_estimators": [50], #sp_randint(50, 200),
            "classifier__max_depth":    [10], #sp_randint(2, 15),
            "classifier__min_samples_split": [4], #sp_randint(2, 10),
        },
        "LightGBM": {
            "classifier__num_leaves": sp_randint(20, 50),
            "classifier__learning_rate": uniform(0.01, 0.3),
            "classifier__max_depth": sp_randint(2, 15),
        },
        "Lasso": {
            "classifier__C": uniform(0.001, 10.0),
        },
        "Ridge": {
            "classifier__alpha": uniform(0.001, 10.0),
        },
        "SVM": {
            "classifier__C": uniform(0.01, 5),
            "classifier__kernel": ["linear", "rbf"],
        },
        "KNN": {
            "classifier__n_neighbors": sp_randint(3, 20),
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        },
        "GradientBoosting": {
            "classifier__n_estimators": sp_randint(50, 200),
            "classifier__learning_rate": uniform(0.01, 0.3),
            "classifier__max_depth": sp_randint(2, 10),
        },
        "AdaBoost": {
            "classifier__n_estimators": sp_randint(50, 200),
            "classifier__learning_rate": uniform(0.01, 1.0),
        },
        "ExtraTrees": {
            "classifier__n_estimators": sp_randint(50, 200),
            "classifier__max_depth": sp_randint(2, 15),
            "classifier__min_samples_split": sp_randint(2, 10),
        },
        "XGBoost": {
            "classifier__n_estimators": sp_randint(50, 200),
            "classifier__learning_rate": uniform(0.01, 0.3),
            "classifier__max_depth": sp_randint(2, 10),
        }
    }

    # Use K-fold cross-validation on the training set
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)


    # ------------------------------------------------------------------------------
    # 6. Train models with hyperparameter search and evaluate on validation set
    # ------------------------------------------------------------------------------
    results = {}
    best = {model_name: None for model_name in models.keys()}
    for model_name, model in models.items():
        print(f"\n\n=== Model: {model_name} ===")
        
        # Build pipeline: preprocessing + model
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Get hyperparameter distribution for this model
        search_params = param_distributions.get(model_name, {})
        
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_params,
            n_iter=3,  # number of parameter settings sampled
            scoring="accuracy",
            cv=kfold,
            random_state=SEED,
            verbose=1,
            n_jobs=-1  # use all available cores
        )
        
        random_search.fit(X_train, y_train)
        
        best_estimator = random_search.best_estimator_
        best[model_name] = best_estimator
        print(f"Best params: {random_search.best_params_}")
        print(f"CV best score (train set): {random_search.best_score_:.4f}")
        
        # Evaluate on validation set
        y_val_pred = best_estimator.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        cm_val = confusion_matrix(y_val, y_val_pred)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("Validation Confusion Matrix:")
        print(cm_val)
        
        results[model_name] = {
            "model": model_name,
            "best_params": random_search.best_params_,
            "cv_best_score": random_search.best_score_,
            "val_accuracy": val_accuracy,
            "val_confusion_matrix": cm_val
        }

        print(results)

    path = "saved_models"
    if not os.path.exists("saved_models"):
        os.makedirs('saved_models')

    
    for model in models.keys():
        # Save each trained model
        timestamp = datetime.now().strftime("%H:%M")
        joblib.dump(best[model], f'saved_models/{model}_{timestamp}_model.pkl')
        
        # Description
        details = {key: str(value) for key, value in results[model].items()}
        details["trained on"] = categorical_features + numerical_features
        with open(f"saved_models/details_{model}_{timestamp}.json", 'w') as fp:
            json.dump(details, fp)
    """