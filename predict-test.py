#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:2021/predict'

customer_id = 'patient-0'

patient = {
    'age': 0.708333,
    'sex': 1.0,
    'trestbps': 0.481132,
    'chol': 0.244292,
    'fbs': 1.0,
    'thalach': 0.603053,
    'exang': 0.0,
    'oldpeak': 0.370968,
    'cp_0': 0.0,
    'cp_1': 0.0,
    'cp_2': 0.0,
    'cp_3': 1.0	,
    'ca_0': 1.0,
    'ca_1': 0.0,
    'ca_2': 0.0,
    'ca_3': 0.0,
    'ca_4': 0.0,
    'slope_0': 1.0,
    'slope_1': 0.0,
    'slope_2': 0.0,
    'restecg_0': 1.0,
    'restecg_1': 0.0,
    'restecg_2': 0.0,
    'thal_0': 0.0,
    'thal_1': 1.0,
    'thal_2': 0.0,
    'thal_3': 0.0
}


response = requests.post(url, json=patient).json()
print(response)

if bool(response['heart_disease']):
    print("Likely to have heart disease")
else:
    print("In good health")