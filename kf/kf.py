import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = joblib.load("./model.joblib")

def result(queries):
    val = []
    val.append(queries)
    val = pd.DataFrame(val)
    return model.predict(val)


# l = [76.0,1.01,3.0,0.0,1.0,57.0,3.07,137.53,4.63,12.53,8406.0,4.71,0.0]
# print(result(l))
