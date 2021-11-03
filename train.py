#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


from io import StringIO
import os
import numpy as np
import pickle
import pandas as pd
from requests import get
import seaborn as sn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold


def train(df_train, y_train, C=1.0):
    dicts = df_train[x_column_names].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[x_column_names].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


if __name__ == "__main__":

    feature_names_heart=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    label_name_heart = "target"

    # get only non-missing data
    heart_df = pd.read_csv('heart.csv')
    rows_with_missing = heart_df.eq("?").any(1)
    heart_df_nm = heart_df[~rows_with_missing]

    # one-hot encoding
    a = pd.get_dummies(heart_df_nm['cp'], prefix="cp")
    b = pd.get_dummies(heart_df_nm['ca'], prefix="ca")
    c = pd.get_dummies(heart_df_nm['slope'], prefix="slope")
    d = pd.get_dummies(heart_df_nm['restecg'], prefix="restecg")
    f = pd.get_dummies(heart_df_nm['thal'], prefix="thal")
    data = [heart_df_nm, a, b, c, d, f]
    heart_df_concat = pd.concat(data, axis=1)

    # finalize the data
    heart_df_final = heart_df_concat.drop(columns=['cp', 'ca', 'slope', 'restecg', 'thal'])
    y = heart_df_final.target.values
    X = heart_df_final.drop(['target'], axis=1)

    # scaling
    x_column_names = X.columns
    scaler = preprocessing.MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=x_column_names)

    # data split
    x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=0)

    # train & predict
    dv, model = train(X_normalized, y, C=1.0)
    y_pred = predict(x_test, dv, model)
    auc = roc_auc_score(y_test, y_pred)
    print(f'auc={auc}')

    # save the model into a file
    output_file = 'model.bin'
    if os.path.exists(output_file):
        os.remove(output_file)
    f_out = open(output_file, 'wb')
    pickle.dump((dv, model), f_out)
    f_out.close()
    print(f'the model is saved to {output_file}')




