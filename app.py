from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from collections import defaultdict

app = Flask(__name__)

MODEL_PATH = "model/elective_rf_model.joblib"
ELECTIVES_PATH = "data/electives_dataset.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. First run: python train_model.py")

if not os.path.exists(ELECTIVES_PATH):
    raise FileNotFoundError("electives_dataset.csv not found inside data folder")

model = joblib.load(MODEL_PATH)

# -----------------------------
# LOAD ELECTIVE MAPPING DATA
# -----------------------------
elective_df = pd.read_csv(ELECTIVES_PATH)

elective_df.columns = elective_df.columns.str.strip().str.upper()

required_cols = ["ELECTIVE", "DEPARTMENT", "TYPE", "CATEGORY"]
for col in required_cols:
    if col not in elective_df.columns:
        raise Exception(f"Missing column in electives_dataset.csv: {col}")

elective_df["ELECTIVE"] = elective_df["ELECTIVE"].astype(str).str.strip().str.upper()
elective_df["DEPARTMENT"] = elective_df["DEPARTMENT"].astype(str).str.strip().str.upper()
elective_df["TYPE"] = elective_df["TYPE"].astype(str).str.strip().str.upper()
elective_df["CATEGORY"] = elective_df["CATEGORY"].astype(str).str.strip().str.upper()


def avg(values):
    return sum(values) / len(values) if values else 0


def extract_features(subjects):
    math_scores = []
    programming_scores = []
    electronics_scores = []
    mechanical_scores = []
    civil_scores = []

    for subject, mark in subjects.items():
        name = str(subject).lower().strip()

        try:
            score = float(mark)
        except:
            continue

        if any(k in name for k in [
            "math", "algebra", "calculus", "statistics",
            "probability", "optimization", "fourier", "laplace"
        ]):
            math_scores.append(score)

        elif any(k in name for k in [
            "program", "algorithm", "data structure", "database",
            "dbms", "software", "compiler", "operating system",
            "machine learning", "computer", "network", "cloud",
            "cryptography", "full stack"
        ]):
            programming_scores.append(score)

        elif any(k in name for k in [
            "electronics", "electrical", "circuit", "signal",
            "communication", "vlsi", "microcontroller",
            "antenna", "digital", "power electronics"
        ]):
            electronics_scores.append(score)

        elif any(k in name for k in [
            "mechanics", "mechanical", "machine", "thermal",
            "thermodynamics", "fluid", "manufacturing",
            "heat transfer", "refrigeration", "automobile"
        ]):
            mechanical_scores.append(score)

        elif any(k in name for k in [
            "civil", "structural", "structure", "surveying",
            "geotechnical", "construction", "concrete",
            "hydrology", "foundation", "highway", "environmental"
        ]):
            civil_scores.append(score)

    return {
        "math_score": avg(math_scores),
        "programming_score": avg(programming_scores),
        "electronics_score": avg(electronics_scores),
        "mechanical_score": avg(mechanical_scores),
        "civil_score": avg(civil_scores)
    }
def score_elective_by_features(category_string, features):
    categories = str(category_string).upper().split("|")

    score = 0

    for cat in categories:
        cat = cat.strip()

        if cat in ["MATHEMATICS", "MATH"]:
            score += features["math_score"] / 100

        elif cat in ["PROGRAMMING", "SOFTWARE", "DATA", "AI"]:
            score += features["programming_score"] / 100

        elif cat in ["HARDWARE", "ELECTRONICS", "NETWORKS"]:
            score += features["electronics_score"] / 100

        elif cat in ["CORE_ENGINEERING", "MECHANICAL"]:
            score += features["mechanical_score"] / 100

        elif cat in ["CIVIL", "STRUCTURAL", "CONSTRUCTION"]:
            score += features["civil_score"] / 100

        elif cat in ["INTERDISCIPLINARY", "SECURITY", "BUSINESS"]:
            score += 0.5

    return score / len(categories) if categories else 0

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        data = request.get_json()

        subjects = data.get("subjects", {})
        department = data.get("department", "").upper().strip()

        if not department:
            return jsonify({"error": "Department is required"}), 400

        features = extract_features(subjects)

        input_df = pd.DataFrame([{
            "department": department,
            "math_score": features["math_score"],
            "programming_score": features["programming_score"],
            "electronics_score": features["electronics_score"],
            "mechanical_score": features["mechanical_score"],
            "civil_score": features["civil_score"]
        }])

        probabilities = model.predict_proba(input_df)[0]
        classes = [str(c).upper().strip() for c in model.classes_]

        prob_map = {
            elective: prob
            for elective, prob in zip(classes, probabilities)
        }

        dept_df = elective_df[elective_df["DEPARTMENT"] == department]

        if dept_df.empty:
            return jsonify({"error": f"No electives found for {department}"}), 400

        results = {}

        pe_order = [
            "PE-I", "PE-II", "PE-III", "PE-IV",
            "PE-V", "PE-VI", "PE-VII", "PE-VIII"
        ]

        for pe_type in pe_order:
            group_data = dept_df[dept_df["TYPE"] == pe_type]

            if group_data.empty:
                continue

            scored = []

            for _, row in group_data.iterrows():
                elective = str(row["ELECTIVE"]).upper().strip()

                # ML probability if model knows the elective
                ml_score = prob_map.get(elective, 0)

                # fallback category score so every PE gets meaningful ranking
                category_score = score_elective_by_features(
                    str(row["CATEGORY"]),
                    features
                )

                final_score = (0.7 * ml_score) + (0.3 * category_score)

                scored.append((elective, final_score))

            scored.sort(key=lambda x: x[1], reverse=True)

            results[pe_type] = [
                elective for elective, score in scored[:2]
            ]

        return jsonify(results)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)