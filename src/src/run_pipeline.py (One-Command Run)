from src.data_loader import load_data
from src.train import train_model
from src.evaluate import evaluate
from src.config import TARGET_COLUMN

df = load_data("data/raw/creditcard.csv")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

model, X_test, y_test = train_model(X, y)
evaluate(model, X_test, y_test)
