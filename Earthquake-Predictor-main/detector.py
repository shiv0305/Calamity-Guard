import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("E:/Coding/Major/Earthquake-Predictor-main/dataset.csv")

data = np.array(data)
print(data)
X = data[:, 0:-1]
y = data[:, -1]
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#  print(X_train,y_train)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)

# print(metrics.accuracy_score(y_test, y_pred))


pickle.dump(rfc,open('E:/Coding/Major/Earthquake-Predictor-main/model2.pkl','wb'))




