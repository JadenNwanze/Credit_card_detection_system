from src.data_ingestion import data_loader
from src.data_preprocessing import Precessor_data
from src.feature_engineering import featureEngin
from src.model_building import build_model
from src.model_evaluation import model_eval

def main():
    data = data_loader()
    X_train_res, X_test, y_train_res, y_test = Precessor_data(data)

    X_train_fe = featureEngin(X_train_res)
    X_test_fe = featureEngin(X_test)

    model = build_model(X_train_fe, y_train_res)

    model_eval(model, X_test_fe, y_test)

if __name__ == "__main__":
    main()
