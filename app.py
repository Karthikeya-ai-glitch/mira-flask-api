from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import platform

# =====================================================
# PLATFORM CHECK (FAISS SAFE)
# =====================================================
USE_FAISS = platform.system() == "Linux"

if USE_FAISS:
    import faiss

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
faiss_index = None

# =====================================================
# GEMINI CONFIG
# =====================================================
API_KEY = "AIzaSyANv7Wr-XdF_zuzZnalr-PYv0ZQtQyi0WE"
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
        return np.array(
            result.embeddings[0].values,
            dtype=np.float16
        )
    except Exception:
        return np.zeros(768, dtype=np.float16)

# =====================================================
# LAZY RESOURCE LOADER
# =====================================================
def load_resources():
    global price_model, model_features, encoder
    global property_df, property_embeddings, faiss_index

    if price_model is None:
        price_model = joblib.load("price_model.pkl")
        model_features = joblib.load("model_features.pkl")
        encoder = joblib.load("ordinal_encoder.pkl")

    if property_df is None:
        property_df = pd.read_csv("property_data.csv")
        property_df["Price"] = property_df["Price"].apply(price_to_number)

    if property_embeddings is None:
        property_embeddings = np.load(
            "property_embeddings.npy",
            mmap_mode="r"
        )

    if USE_FAISS and faiss_index is None:
        faiss_index = faiss.read_index("faiss.index")

# =====================================================
# ROUTES
# =====================================================
@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        load_resources()

        data = request.get_json()
        df = pd.DataFrame([data])

        # -------------------------------------------------
        # DROP NON-INFERENCE COLUMNS
        # -------------------------------------------------
        if "Property ID" in df.columns:
            df = df.drop(columns=["Property ID"])

        # -------------------------------------------------
        # HANDLE DATE SOLD (ENGINEER FEATURES)
        # -------------------------------------------------
        if "Date Sold" in df.columns:
            df["Date Sold"] = pd.to_datetime(df["Date Sold"])

            df["Sold_Year"] = df["Date Sold"].dt.year
            df["Sold_Month"] = df["Date Sold"].dt.month
            df["Sold_Quarter"] = df["Date Sold"].dt.quarter

            # Property age at time of sale
            if "Year Built" in df.columns:
                df["Property_Age_At_Sale"] = df["Sold_Year"] - df["Year Built"]

            df = df.drop(columns=["Date Sold"])

        # -------------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # -------------------------------------------------
        cat_cols = ["Location", "Condition", "Type"]
        df[cat_cols] = encoder.transform(df[cat_cols])

        # -------------------------------------------------
        # ALIGN FEATURES WITH TRAINING
        # -------------------------------------------------
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

        # -------------------------------------------------
        # USER EMBEDDING
        # -------------------------------------------------
        user_embedding = get_gemini_embedding(
            payload["qualitative_description"],
            "RETRIEVAL_QUERY"
        ).astype("float32").reshape(1, -1)

        # -------------------------------------------------
        # SEMANTIC SEARCH (FAISS OR FALLBACK)
        # -------------------------------------------------
        if USE_FAISS:
            K = 50
            sims, idxs = faiss_index.search(user_embedding, K)
            candidate_df = property_df.iloc[idxs[0]].reset_index(drop=True)
            semantic_scores = sims[0]
        else:
            semantic_scores = np.dot(
                property_embeddings,
                user_embedding.T
            ).ravel()
            top_idx = np.argsort(semantic_scores)[-50:]
            candidate_df = property_df.iloc[top_idx].reset_index(drop=True)
            semantic_scores = semantic_scores[top_idx]

        # -------------------------------------------------
        # NUMERIC SIMILARITY
        # -------------------------------------------------
        budget = float(payload["budget"])
        beds = int(payload["bedrooms"])
        baths = int(payload["bathrooms"])

        bed_scores = np.exp(-np.abs(beds - candidate_df["Bedrooms"].values))
        bath_scores = np.exp(-np.abs(baths - candidate_df["Bathrooms"].values))
        price_scores = np.exp(
            -np.abs(budget - candidate_df["Price"].values) / (0.1 * budget)
        )

        # -------------------------------------------------
        # PRIORITY-BASED WEIGHTING
        # -------------------------------------------------
        priority_order = payload.get(
            "priority_order",
            ["bedrooms", "price", "bathrooms", "quality_description"]
        )

        weights = [0.4, 0.3, 0.2, 0.1]
        weight_map = dict(zip(priority_order, weights))

        final_scores = (
            weight_map["bedrooms"] * bed_scores +
            weight_map["price"] * price_scores +
            weight_map["bathrooms"] * bath_scores +
            weight_map["quality_description"] * semantic_scores
        ) * 100

        # -------------------------------------------------
        # TOP-5 RESULTS
        # -------------------------------------------------
        top_idx = np.argsort(final_scores)[-5:][::-1]

        top_5 = candidate_df.iloc[top_idx].copy()
        top_5["Match_Score"] = np.round(final_scores[top_idx], 2)

        return jsonify(top_5.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
