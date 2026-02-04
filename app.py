from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# =====================================================
# LOAD LIGHTWEIGHT MODELS
# =====================================================
# Pre-loading smaller models to stay under 512MB
price_model = joblib.load("price_model.pkl")
model_features = joblib.load("model_features.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Use MiniLM (80MB) instead of MPNet (420MB)
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# LOAD & PRE-PROCESS DATA
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

# Pre-calculate embeddings once at startup to save time during requests
# This fits in memory with MiniLM, but NOT with MPNet
property_embeddings = text_model.encode(
    property_df["Qualitative Description"].tolist(),
    normalize_embeddings=True,
    convert_to_numpy=True
)

# =====================================================
# HELPERS
# =====================================================
def soft_score(diff, tolerance):
    return np.exp(-diff / tolerance)

def get_weights_from_priority(priority_order):
    base_weights = [0.40, 0.30, 0.20, 0.10]
    return {priority_order[i]: base_weights[i] for i in range(4)}

# =====================================================
# API ENDPOINTS
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
        
        # Encode and Align
        df[ ["Location", "Condition", "Type"] ] = encoder.transform(df[["Location", "Condition", "Type"]])
        df = df.reindex(columns=model_features)
        
        prediction = price_model.predict(df)[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/match", methods=["POST"])
def match_properties():
    try:
        payload = request.get_json()
        priority_order = payload.get("priority_order", ["bedrooms", "price", "bathrooms", "quality_description"])
        weights = get_weights_from_priority(priority_order)

        # 1. Encode user description
        user_emb = text_model.encode(payload["qualitative_description"], normalize_embeddings=True)

        # 2. Vectorized Text Similarity (Much faster/lighter than a loop)
        # Cosine similarity on normalized vectors is just the dot product
        text_sims = np.dot(property_embeddings, user_emb)

        # 3. Vectorized Numeric Scoring
        bed_diffs = np.abs(int(payload["bedrooms"]) - property_df["Bedrooms"].values)
        bath_diffs = np.abs(int(payload["bathrooms"]) - property_df["Bathrooms"].values)
        price_diffs = np.abs(float(payload["budget"]) - property_df["Price"].values)

        bed_scores = np.exp(-bed_diffs / 1)
        bath_scores = np.exp(-bath_diffs / 1)
        price_scores = np.exp(-price_diffs / (0.1 * float(payload["budget"])))

        # 4. Final Weighted Score
        final_scores = (
            weights["bedrooms"] * bed_scores +
            weights["price"] * price_scores +
            weights["bathrooms"] * bath_scores +
            weights["quality_description"] * text_sims
        ) * 100

        # 5. Format results
        property_df["Match_Score"] = np.round(final_scores, 2)
        top_5 = property_df.sort_values(by="Match_Score", ascending=False).head(5)
        
        return jsonify(top_5.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Render uses the PORT env var
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
