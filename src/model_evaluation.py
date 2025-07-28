import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/raw/creditcard.csv")

from src.data_preprocessing import Precessor_data
from src.feature_engineering import featureEngin

df = featureEngin(df)
X_train, X_test, y_train, y_test = Precessor_data(df)


def model_eval(model, X_test, y_test):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import json
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    rou = roc_auc_score(y_test,y_pred_proba)
    metrics_dict = {
        'confusion matrix':cm.tolist(),
        'classifcation report':cr,
        'AUC': rou
    }

    with open('metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent =4)

model_eval(model, X_test, y_test)

