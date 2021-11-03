============ BACKGROUND =============

Name: Eric Kiedian Liu


============ Problem =============

The goal of the ML project is to predict if a patient will have a heart disease or not.
1=Yes, 0=No.

The dataset can be found on Kaggle

<https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv>

The features are age, sex, chest pain type, resting blood pressure, serum cholestoral in mg/dl
, fasting blood sugar > 120 mg/dl, resting electro-cardiographic results, maximum heart rate achieved,
exercise induced angina, oldpeak, the slope of the peak exercise ST segment, number of major vessels,
thal

The label is 'target'


============ How to Run =============

First, go to
<https://colab.research.google.com/drive/1cuax-m7Sci1mDjwK4HUh_353dJrDhDhJ?usp=sharing>
to see the EDA, Model Selection, Model Scores

Second, open a terminal and navigate to this folder where this file is at.

This command will generate the model -
$ python train.py

Third, enter the pipenv shell and start the flask service

$ pipenv shell

$ python predict.py

Fourth, open up another terminal prompt and enter the pipenv shell,
and to send a request to the prediction service

$ pipenv shell

$ python predict-test.py

======== Run with docker ===========

If you prefer to use docker.

$ docker build -t eric_liu_zoomcamp_midterm .

$ docker run -it --rm eric_liu_zoomcamp_midterm

Open up another command prompt.

$ python predict-test.py