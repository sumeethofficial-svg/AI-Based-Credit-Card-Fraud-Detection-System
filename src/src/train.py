import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess
from src.config import TEST_SIZE, RANDOM_STATE

def train_model(X, y):
    X_resampled, y_resampled = preprocess(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_resampled
    )

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    joblib.dump(model, "models/fraud_model.pkl")

    return model, X_test, y_test
