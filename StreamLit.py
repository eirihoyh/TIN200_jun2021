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
    Property_Area = st.sidebar.selectbox("Property_Area", ["Urban", "Rural","Semiurban"])
    
    
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

pred_user = verdier_fra_bruker()
st.write(pred_user)
st.write("--")

train = train.dropna()
train["Dependents"].replace({"0":0 , "1": 1,"2":2,"3":3,"3+":3, "4" : 4 }, inplace = True)
train["Gender"].replace({"Male":0 , "Female": 1 }, inplace = True)
train["Married"].replace({"Yes":0 , "No": 1 }, inplace = True)
train["Education"].replace({"Graduate":0 , "Not Graduate": 1 }, inplace = True)
train["Self_Employed"].replace({"Yes": 0 , "No": 1}, inplace = True )
train["Property_Area"].replace({"Urban": 0 , "Rural": 1,"Semiurban" : 2 }, inplace = True )
train["Loan_Status"].replace({"Y": 0,"N" : 1 }, inplace = True )
print(train["Credit_History"].value_counts())


pred_user["Gender"].replace({"Male":0 , "Female": 1 }, inplace = True)
pred_user["Married"].replace({"Yes":0 , "No": 1 }, inplace = True)
pred_user["Education"].replace({"Graduate":0 , "Not Graduate": 1 }, inplace = True)
pred_user["Self_Employed"].replace({"Yes": 0 , "No": 1}, inplace = True )
pred_user["Property_Area"].replace({"Urban": 0 , "Rural": 1,"Semiurban" : 2 }, inplace = True )




X = train.iloc[:, :-1]
y = train.iloc[:, -1]


#Test train split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

pipe_lr = make_pipeline(
    XGBClassifier(booster="gbtree", learning_rate=0.05, max_depth=5, n_estimators=100, min_child_weight=4, nthread=8, subsample=0.5,use_label_encoder=False)
)

pipe_lr.fit(X_train, y_train)

y_train_pred = pipe_lr.predict(X_train)
y_test_pred = pipe_lr.predict(X_test)
prediksjon = pipe_lr.predict(pred_user)

st.write("Basert på dette får du %s " %prediksjon)



























