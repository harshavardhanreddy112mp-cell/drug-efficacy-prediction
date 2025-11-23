import joblib
from sklearn.metrics import roc_auc_score, classification_report
from preprocessing import load_data, preprocess

def evaluate():
    model = joblib.load("models/model.pkl")
    df = load_data("data/sample.csv")
    X, y, _ = preprocess(df)

    preds = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, preds)

    print("ROC-AUC:", roc)
    print(classification_report(y, preds > 0.5))

if __name__ == "__main__":
    evaluate()
