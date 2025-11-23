import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()
    features = df.drop("label", axis=1)
    labels = df["label"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, scaler
