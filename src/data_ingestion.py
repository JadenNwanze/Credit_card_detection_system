import pandas as pd 

def data_loader(path="data/raw/creditcard.csv"):
    df = pd.read_csv(path)
    return df

