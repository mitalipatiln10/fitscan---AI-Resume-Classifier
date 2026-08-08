"""
Loads Model B (Logistic Regression on SBERT embeddings) + label encoder + SBERT.
Load these ONCE at server startup, not per-request — SBERT model load is slow.
"""

import os
# Force transformers to use PyTorch only, skip TensorFlow entirely.
# Must be set BEFORE importing sentence_transformers/transformers, since a
# broken local TF install otherwise crashes the import even though we never
# use TF for anything here.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import joblib
from sentence_transformers import SentenceTransformer

MODEL_B_PATH = "model_B.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

model_B = joblib.load(MODEL_B_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

# --- Compatibility patch ---
# model_B was trained/saved under a different scikit-learn version than what's
# installed here. Newer sklearn dropped/renamed some LogisticRegression internals
# (e.g. multi_class), which predict_proba() references internally. Patch the
# missing attribute back on so the loaded model behaves correctly.
if not hasattr(model_B, "multi_class"):
    model_B.multi_class = "auto"


def predict_category(resume_clean_text: str):
    """Returns (predicted_category: str, confidence: float, embedding: np.ndarray)"""
    embedding = sbert_model.encode([resume_clean_text])
    pred = model_B.predict(embedding)
    category = le.inverse_transform(pred)[0]

    # Confidence via predict_proba if available (LogisticRegression supports this)
    confidence = None
    if hasattr(model_B, "predict_proba"):
        proba = model_B.predict_proba(embedding)[0]
        confidence = float(max(proba))

    return category, confidence, embedding[0]


def embed_text(text: str):
    """Returns a single SBERT embedding vector for arbitrary text (e.g. a JD)."""
    return sbert_model.encode([text])[0]
