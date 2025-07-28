def Precessor_data(df):
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test
