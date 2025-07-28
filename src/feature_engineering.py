def featureEngin(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    return df
