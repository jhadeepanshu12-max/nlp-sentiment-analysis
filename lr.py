from sklearn import datasets
iris = datasets.load_iris()

import pandas as pd 
data = pd.read_csv("C:\\Users\\jhavi\\Downloads\\iris (3).csv")
print (data.head())
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
X = data[['sepal_length','sepal_width','petal_length','petal_width']]
Y = data['species']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=200)
model.fit (X_train,Y_train)
Y_pred=model.predict(X_test)
print("accuracy:",accuracy_score(Y_test,Y_pred))
print("classification_report:",classification_report(Y_test,Y_pred))
