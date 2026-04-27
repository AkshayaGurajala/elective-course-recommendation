import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv("data/multidept_dataset.csv")

print(df.head())

# Features
X = df[
[
"department",
"math_score",
"programming_score",
"electronics_score",
"mechanical_score",
"civil_score"
]
]

# Label
y = df["elective"]


# Encode department column
preprocessor = ColumnTransformer(
transformers=[
(
"dept",
OneHotEncoder(handle_unknown="ignore"),
["department"]
)
],
remainder="passthrough"
)


# Random Forest model
model = RandomForestClassifier(
n_estimators=200,
random_state=42
)

pipeline = Pipeline([
("preprocess", preprocessor),
("model", model)
])


# Train
pipeline.fit(X,y)


# Create model folder if missing
os.makedirs("model", exist_ok=True)


# Save model
joblib.dump(
pipeline,
"model/elective_rf_model.joblib"
)

print("MODEL CREATED SUCCESSFULLY")