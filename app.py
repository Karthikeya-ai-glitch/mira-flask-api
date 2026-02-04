from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# =====================================================
# LOAD PRICE PREDICTION MODEL
# =====================================================
price_model = joblib.load("price_model.pkl")
model_features = joblib.load("model_features.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# =====================================================
# LOAD PROPERTY DATA FOR MATCHING
# =====================================================
property_df = pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")
property_df = property_df.dropna().reset_index(drop=True)

def price_to_number(x):
    if isinstance(x, str):
        x = x.lower().replace("$", "").replace(",", "")
        if "k" in x:
            return float(x.replace("k", "")) * 1000
        return float(x)
    return float(x)

property_df["Price"] = property_df["Price"].apply(price_to_number)

# =====================================================
# LOAD TEXT MODEL + EMBEDDINGS
# =====================================================
text_model = SentenceTransformer("all-mpnet-base-v2")
property_embeddings = text_model.encode(
    property_df["Qualitative Description"].tolist(),
    normalize_embeddings=True
)

# =====================================================
# PRICE PREDICTION API
# =====================================================
@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        if "Property ID" in df.columns:
            df = df.drop(columns=["Property ID"])

        if "Date Sold" in df.columns:
            df["Date Sold"] = pd.to_datetime(df["Date Sold"])

        cat_cols = ["Location", "Condition", "Type"]
        df[cat_cols] = encoder.transform(df[cat_cols])

        df = df.reindex(columns=model_features)
        prediction = price_model.predict(df)[0]

        return jsonify({"predicted_price": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =====================================================
# MATCHING HELPERS
# =====================================================
def soft_score(diff, tolerance):
    return np.exp(-diff / tolerance)

def numeric_scores(user, prop):
    bed = soft_score(abs(user["bedrooms"] - prop["Bedrooms"]), 1)
    bath = soft_score(abs(user["bathrooms"] - prop["Bathrooms"]), 1)
    price = soft_score(abs(user["budget"] - prop["Price"]), 0.1 * user["budget"])
    return bed, bath, price

def text_similarity(user_embedding, idx):
    return cosine_similarity(
        user_embedding.reshape(1, -1),
        property_embeddings[idx].reshape(1, -1)
    )[0][0]

def get_weights_from_priority(priority_order):
    base_weights = [0.40, 0.30, 0.20, 0.10]
    return {priority_order[i]: base_weights[i] for i in range(4)}

def final_match_score(user, user_embedding, idx, priority_order):
    weights = get_weights_from_priority(priority_order)

    bed, bath, price = numeric_scores(user, property_df.iloc[idx])
    text = text_similarity(user_embedding, idx)

    score = (
        weights["bedrooms"] * bed +
        weights["price"] * price +
        weights["bathrooms"] * bath +
        weights["quality_description"] * text
    )

    return round(score * 100, 2)

# =====================================================
# MATCHING API (WITH PRIORITY ORDER)
# =====================================================
@app.route("/match", methods=["POST"])
def match_properties():
    payload = request.get_json()

    required_fields = ["budget", "bedrooms", "bathrooms", "qualitative_description"]
    for field in required_fields:
        if field not in payload:
            return jsonify({"error": f"Missing field: {field}"}), 400

    priority_order = payload.get(
        "priority_order",
        ["bedrooms", "price", "bathrooms", "quality_description"]
    )

    if set(priority_order) != {"bedrooms", "price", "bathrooms", "quality_description"}:
        return jsonify({"error": "Invalid priority order"}), 400

    user = {
        "budget": float(payload["budget"]),
        "bedrooms": int(payload["bedrooms"]),
        "bathrooms": int(payload["bathrooms"])
    }

    user_embedding = text_model.encode(
        payload["qualitative_description"],
        normalize_embeddings=True
    )

    results = []
    for i in range(len(property_df)):
        score = final_match_score(user, user_embedding, i, priority_order)
        row = property_df.iloc[i].to_dict()
        row["Match_Score"] = score
        results.append(row)

    results = sorted(results, key=lambda x: x["Match_Score"], reverse=True)
    return jsonify(results[:5])

# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
