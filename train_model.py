import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import sys

# Load dataset
df = pd.read_csv("symptoms.csv")

# Remove diseases with fewer than 2 samples
disease_counts = df["disease"].value_counts()
df = df[df["disease"].isin(disease_counts[disease_counts >= 2].index)]

# Check if anything is left
if df.shape[0] == 0:
    print("❌ Not enough data after filtering. Please add more samples per disease.")
    sys.exit()

# Features and labels
X = df.drop("disease", axis=1)
y = df["disease"].astype(str)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# Print info
print("Encoded classes:", list(le.classes_))
print("Training classes:", sorted(set(y_train)))

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("✅ Model Accuracy:", accuracy)

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
