from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled
