import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from src.data_preprocessing import Precessor_data
from src.feature_engineering import featureEngin

def build_model(X_train_res, y_train_res):
    try: 
        lr_model = LogisticRegression(solver='liblinear')
        xgb_model = XGBClassifier(
          n_estimators=10,
          max_depth=3,
          learning_rate=0.3,
          verbosity=1,
          random_state=42,
          n_jobs=2
            )
        rf_model = RandomForestClassifier(
            n_estimators=15, 
            max_depth=6, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            n_jobs=-1, 
            random_state=42
            )
        from sklearn.ensemble import VotingClassifier
        clf = VotingClassifier(
            estimators=[
                ('lr', lr_model),
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            voting='soft'
            )
        
        clf.fit(X_train_res, y_train_res)
    except:
        print(f"there is a problem..")   

    return clf

   


def save_model(clf, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(clf, f)

df = pd.read_csv("data/raw/creditcard.csv")

df = featureEngin(df)
X_train, X_test, y_train, y_test = Precessor_data(df)
  

clf = build_model(X_train, y_train)
if clf:
    save_model(clf)