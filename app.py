from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from google import genai
from google.genai import types

# =====================================================
# FLASK SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# GLOBALS (LAZY LOADED)
# =====================================================
price_model = None
model_features = None
encoder = None
property_df = None
property_embeddings = None

# =====================================================
# GEMINI CONFIG
# =====================================================
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# =====================================================
# HELPERS
# =====================================================
def price_to_number(x):
    if isinstance(x, str):
        x = x.lower().replace("$", "").replace(",", "")
        if "k" in x:
            return float(x.replace("k", "")) * 1000
        return float(x)
    return float(x)

def get_gemini_embedding(text, task_type):
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros(768, dtype=np.float32)

# =====================================================
# LAZY RESOURCE LOADER (CRITICAL FOR RENDER)
# =====================================================
def load_resources():
    global price_model, model_features, encoder
    global property_df, property_embeddings

    if price_model is None:
        print("Loading ML models...")
        price_model = joblib.load("price_model.pkl")
        model_features = joblib.load("model_features.pkl")
        encoder = joblib.load("ordinal_encoder.pkl")

    if property_df is None:
        print("Loading property data...")
        property_df = (
            pd.read_excel("Case Study 2 Data.xlsx", sheet_name="Property Data")
            .dropna()
            .reset_index(drop=True)
        )
        property_df["Price"] = property_df["Price"].apply(price_to_number)

    if property_embeddings is None:
        print("Generating property embeddings (lazy)...")
        property_embeddings = np.array(
            [
                get_gemini_embedding(desc, "RETRIEVAL_DOCUMENT")
                for desc in property_df["Qualitative Description"].tolist()
            ],
            dtype=np.float32
        )

# =====================================================
# ROUTES
# =====================================================
@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        load_resources()

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


@app.route("/match", methods=["POST"])
def match_properties():
    try:
        load_resources()

        payload = request.get_json()

        user_embedding = get_gemini_embedding(
            payload["qualitative_description"],
            "RETRIEVAL_QUERY"
        )

        text_sims = np.dot(property_embeddings, user_embedding)

        budget = float(payload["budget"])
        beds = int(payload["bedrooms"])
        baths = int(payload["bathrooms"])

        priority_order = payload.get(
            "priority_order",
            ["bedrooms", "price", "bathrooms", "quality_description"]
        )

        weights = [0.4, 0.3, 0.2, 0.1]
        weight_map = dict(zip(priority_order, weights))

        bed_scores = np.exp(-np.abs(beds - property_df["Bedrooms"].values))
        bath_scores = np.exp(-np.abs(baths - property_df["Bathrooms"].values))
        price_scores = np.exp(
            -np.abs(budget - property_df["Price"].values) / (0.1 * budget)
        )

        final_scores = (
            weight_map["bedrooms"] * bed_scores +
            weight_map["price"] * price_scores +
            weight_map["bathrooms"] * bath_scores +
            weight_map["quality_description"] * text_sims
        ) * 100

        property_df["Match_Score"] = np.round(final_scores, 2)

        top_5 = (
            property_df
            .sort_values(by="Match_Score", ascending=False)
            .head(5)
        )

        return jsonify(top_5.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =====================================================
# ENTRYPOINT (RENDER NEEDS THIS)
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
