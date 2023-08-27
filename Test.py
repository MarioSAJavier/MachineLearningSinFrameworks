import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from MachineLearning import LogisticRegression
import numpy as np


iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 0).astype(int) # return 1 if Iris Setosa (0 = setosa, 1 = versicolor, 2 = virginica), and 0 if not

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression() 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def accuracy(yPredicted, yReal):
   return np.sum(yPredicted == yReal)/len(yReal)

acc = accuracy(y_pred, y_test)
print(acc)