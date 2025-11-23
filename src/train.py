import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from preprocess import load_data, preprocess  # corrected import name if needed

def train_model():
    df = load_data("data/sample.csv")
    X, y, scaler = preprocess(df)

    base_model = LogisticRegression(max_iter=500)
    calibrated = CalibratedClassifierCV(base_model, cv=5)

    calibrated.fit(X, y)

    joblib.dump(calibrated, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
