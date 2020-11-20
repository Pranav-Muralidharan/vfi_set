import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import joblib

data = pd.read_csv('new_model.csv')

x = data.drop(['Class'],axis=1)
y = data['Class']
lab_enc=LabelEncoder()
y=lab_enc.fit_transform(y)

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# clf=RandomForestClassifier(n_estimators=100)
# clf.fit(X_train,Y_train)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(x,y)

# joblib.dump(clf, "./model.joblib")

def result(queries):
    val = []
    val.append(queries)
    val = pd.DataFrame(val)
    return clf.predict(val)


# l = [76.0,1.01,3.0,0.0,1.0,57.0,3.07,137.53,4.63,12.53,8406.0,4.71,0.0]
# print(result(l))