# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:23:08 2021

@author: sebbe
"""
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

st.write(""" # Nettside for å automatisere låneprosessen""")

train = pd.read_csv("data/train.csv", index_col=0)
testt = pd.read_csv("data/test.csv", index_col=0)  # does not contain targets


st.sidebar.header("Input verdier")


def verdier_fra_bruker():
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Married = st.sidebar.selectbox("Married?", ["Yes", "No"])
    Dependents = st.sidebar.slider("Dependents",0,10)
    Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.sidebar.slider("ApplicantIncome",float(train.ApplicantIncome.min()),float(train.ApplicantIncome.max()),float(train.ApplicantIncome.mean()))
    CoapplicantIncome = st.sidebar.slider("CoapplicantIncome",float(train.CoapplicantIncome.min()),float(train.CoapplicantIncome.max()),float(train.CoapplicantIncome.mean()))
    LoanAmount = st.sidebar.slider("Loan_Amount",float(train.LoanAmount.min()),float(train.LoanAmount.max()),float(train.LoanAmount.mean()))
    Loan_Amount_Term = st.sidebar.slider("Loan_Amount_Term",float(train.Loan_Amount_Term.min()),float(train.Loan_Amount_Term.max()),float(train.Loan_Amount_Term.mean()))
    Credit_History = st.sidebar.slider("Credit_History",0,1)
    Property_Area = st.sidebar.selectbox("Property_Area", ["Urban", "Rural"])
    
    
    data = {
            "Dependents": Dependents,
            "Gender" : Gender,
            "Married":Married,
            "Education": Education,
            "Self_Employed" : Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome" : CoapplicantIncome,
            "Loan_Amount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History" : Credit_History,
            "Property_Area" : Property_Area   
            }
    featurs = pd.DataFrame(data, index = [0])
    return featurs 

test = verdier_fra_bruker()
st.write(test)
test = testt.iloc[1]
st.write(test)
st.write("--")

#Gender
train_no_gender = train.copy().drop(columns="Gender")

print(test)
test_no_gedner = test.copy().drop(columns="Gender")
print(test_no_gedner)


# Married
train_no_nan_married = train_no_gender.copy().dropna(axis=0, subset=["Married"])
train_no_nan_married = pd.get_dummies(train_no_nan_married, columns=["Married"], drop_first=True)


test_no_nan_married = test_no_gedner.copy().dropna(axis=0)
print(test_no_nan_married)
#test_no_nan_married = pd.get_dummies(test_no_nan_married, columns=["Married"], drop_first=True)
print(test_no_nan_married)

# Dependents
train_dependent_only_int = train_no_nan_married.copy().replace("3+", 3)
for number in range(0, 3):
    train_dependent_only_int = train_dependent_only_int.replace(f"{number}", number)

train_dependents_no_nan = train_dependent_only_int.copy()
median = np.nanmedian(train_dependent_only_int.Dependents)
train_dependents_no_nan["Missing_Dependents"] = [int(x) for x in train_dependent_only_int.Dependents.isnull().values]
train_dependents_no_nan.Dependents = train_dependent_only_int.copy().Dependents.fillna(median)



test_dependent_only_int = test_no_nan_married.copy().replace("3+", 3)
for number in range(0, 3):
    test_dependent_only_int = test_dependent_only_int.replace(f"{number}", number)

test_dependents_no_nan = test_dependent_only_int.copy()
median = np.nanmedian(test_dependent_only_int.Dependents)
#test_dependents_no_nan["Missing_Dependents"] = [int(x) for x in test_dependent_only_int.Dependents.isnull().values]
#test_dependents_no_nan.Dependents = test_dependent_only_int.copy().Dependents.fillna(median)

# Education
train_education_dummies = pd.get_dummies(train_dependents_no_nan.copy(), columns=["Education"], drop_first=True)

missing_ind = MissingIndicator(error_on_new=True, features="missing-only")
train_self_employed_encoded = train_education_dummies.copy()

test_education_dummies = pd.get_dummies(test_dependents_no_nan.copy(), columns=["Education"], drop_first=True)
missing_ind = MissingIndicator(error_on_new=True, features="missing-only")
test_self_employed_encoded = test_education_dummies.copy()


# Self_Employed
train_self_employed_encoded["Missing_Self_Employed"] = missing_ind.fit_transform(train_self_employed_encoded.Self_Employed.values.reshape(-1, 1))
train_self_employed_encoded["Missing_Self_Employed"] = train_self_employed_encoded["Missing_Self_Employed"].replace([True, False], [1, 0])
train_self_employed_encoded.Self_Employed = train_self_employed_encoded.Self_Employed.replace([np.nan, "No", "Yes"], [0, 0, 1]) 


test_self_employed_encoded["Missing_Self_Employed"] = missing_ind.fit_transform(test_self_employed_encoded.Self_Employed.values.reshape(-1, 1))
test_self_employed_encoded["Missing_Self_Employed"] = test_self_employed_encoded["Missing_Self_Employed"].replace([True, False], [1, 0])
test_self_employed_encoded.Self_Employed = test_self_employed_encoded.Self_Employed.replace([np.nan, "No", "Yes"], [0, 0, 1]) 
# Loan_Amount_Term
si = SimpleImputer(strategy="median")

train_imputed_loan_amount_term = train_self_employed_encoded.copy()
train_imputed_loan_amount_term.Loan_Amount_Term = si.fit_transform(train_imputed_loan_amount_term.Loan_Amount_Term.values.reshape(-1, 1))

si = SimpleImputer(strategy="median")

test_imputed_loan_amount_term = test_self_employed_encoded.copy()
test_imputed_loan_amount_term.Loan_Amount_Term = si.fit_transform(test_imputed_loan_amount_term.Loan_Amount_Term.values.reshape(-1, 1))


# Credit_History
train_credit_history_no_nan = train_imputed_loan_amount_term.copy()

missing_ind = MissingIndicator(error_on_new=True, features="missing-only")
si = SimpleImputer(strategy="median")

train_credit_history_no_nan["Missing_Credit_History"] = missing_ind.fit_transform(train_credit_history_no_nan.Credit_History.values.reshape(-1, 1))
train_credit_history_no_nan["Missing_Credit_History"] = train_credit_history_no_nan["Missing_Credit_History"].replace([True, False], [1, 0])

train_credit_history_no_nan.Credit_History = si.fit_transform(train_credit_history_no_nan.Credit_History.values.reshape(-1, 1))

#test
test_credit_history_no_nan = test_imputed_loan_amount_term.copy()

missing_ind = MissingIndicator(error_on_new=True, features="missing-only")
si = SimpleImputer(strategy="median")

test_credit_history_no_nan["Missing_Credit_History"] = missing_ind.fit_transform(test_credit_history_no_nan.Credit_History.values.reshape(-1, 1))
test_credit_history_no_nan["Missing_Credit_History"] = test_credit_history_no_nan["Missing_Credit_History"].replace([True, False], [1, 0])

test_credit_history_no_nan.Credit_History = si.fit_transform(test_credit_history_no_nan.Credit_History.values.reshape(-1, 1))

# Property_Area and Loan_Status
train_property_area_n_target = pd.get_dummies(train_credit_history_no_nan.copy(), columns=["Property_Area", "Loan_Status"], drop_first=True)

test_property_area_n_target = pd.get_dummies(test_credit_history_no_nan.copy(), columns=["Property_Area"], drop_first=True)
# Loan amount
train_LoanAmount_itterative_imputer = train_property_area_n_target.copy()

X = train_LoanAmount_itterative_imputer.iloc[:, :-1]
y = train_LoanAmount_itterative_imputer.iloc[:, -1]
imp_mean = IterativeImputer(random_state=0)
X = imp_mean.fit_transform(X)
X = pd.DataFrame(X, columns=train_LoanAmount_itterative_imputer.iloc[:, :-1].columns)
X

test_LoanAmount_itterative_imputer = test_property_area_n_target.copy()

X_pred = test_LoanAmount_itterative_imputer.iloc[:, :]

imp_mean = IterativeImputer(random_state=0)
X_pred = imp_mean.fit_transform(X_pred)

X_pred = pd.DataFrame(X_pred, columns=test_LoanAmount_itterative_imputer.iloc[:, :].columns)

X_pred




#Test train split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


pipe_lr = make_pipeline(
    XGBClassifier(booster="gbtree", learning_rate=0.05, max_depth=5, n_estimators=100, min_child_weight=4, nthread=8, subsample=0.5)
)

pipe_lr.fit(X_train, y_train)

y_train_pred = pipe_lr.predict(X_train)
y_test_pred = pipe_lr.predict(X_test)
prediksjon = pipe_lr.predict(X_pred)

st.header("Prediksjon")
st.write("Basert på ødine tall:%s " %prediksjon)


























