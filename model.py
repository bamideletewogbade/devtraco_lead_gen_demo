# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import joblib

# # Load and prepare data
# data = pd.read_csv("devtraco_leads.csv")
# encoder = LabelEncoder()
# data["source_encoded"] = encoder.fit_transform(data["source"])

# # Train model
# X = data[["source_encoded", "budget", "time_on_website"]]
# y = data["converted"].apply(lambda x: 1 if x == "Yes" else 0)
# model = RandomForestClassifier()
# model.fit(X, y)

# # Save artifacts
# joblib.dump(model, "model.joblib")
# joblib.dump(encoder, "encoder.joblib")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and prepare data
data = pd.read_csv("devtraco_leads2.csv")  # Use the updated dataset with consultants

# Encode categorical features
encoders = {}
for col in ["source", "consultant", "property_type"]:
    encoders[col] = LabelEncoder()
    data[f"{col}_encoded"] = encoders[col].fit_transform(data[col])

# Create interaction features
data["source_consultant"] = data["source"] + "_" + data["consultant"]

# Train model with more features
X = data[[
    "source_encoded", 
    "budget", 
    "time_on_website",
    "consultant_encoded",
    "property_type_encoded",
]]
y = data["converted"].apply(lambda x: 1 if x == "Yes" else 0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(encoders, "encoders.joblib")

# Generate consultant performance report
consultant_performance = data.groupby(['consultant', 'source']).agg({
    'converted': lambda x: (x == 'Yes').mean(),
    'budget': 'mean',
    'lead_id': 'count'
}).rename(columns={
    'converted': 'conversion_rate',
    'budget': 'avg_budget',
    'lead_id': 'lead_count'
}).reset_index()

consultant_performance.to_csv("consultant_performance.csv", index=False)