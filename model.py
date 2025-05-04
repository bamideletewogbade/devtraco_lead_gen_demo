import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and prepare data
data = pd.read_csv("devtraco_leads.csv")
encoder = LabelEncoder()
data["source_encoded"] = encoder.fit_transform(data["source"])

# Train model
X = data[["source_encoded", "budget", "time_on_website"]]
y = data["converted"].apply(lambda x: 1 if x == "Yes" else 0)
model = RandomForestClassifier()
model.fit(X, y)

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(encoder, "encoder.joblib")