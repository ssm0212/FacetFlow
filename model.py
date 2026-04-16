import os
from facets import FACETS

os.environ["TORCH_HOME"] = "D:/model_cache"
os.environ["HF_HOME"] = "D:/model_cache"


from textblob import TextBlob
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
toxicity_model = Detoxify('original')

def extract_features(text, prev_text=None):
    text = text.lower().strip()
    features = {}

    blob = TextBlob(text)
    features['sentiment'] = blob.sentiment.polarity
    features['word_count'] = len(text.split())

    tox = toxicity_model.predict(text)
    features['toxicity'] = tox['toxicity']

    features['politeness'] = int(any(w in text for w in ["please", "thank you", "sorry"]))

    empathy_words = ["understand", "sorry", "feel", "support", "stay strong"]
    features['empathy'] = int(any(w in text for w in empathy_words))

    if prev_text:
        emb1 = embedder.encode([text])
        emb2 = embedder.encode([prev_text])
        features['context_similarity'] = cosine_similarity(emb1, emb2)[0][0]
    else:
        features['context_similarity'] = 0

    return features


def scale_sentiment(x):
    if x <= -0.6: return 0
    elif x <= -0.1: return 1
    elif x <= 0.1: return 2
    elif x <= 0.5: return 3
    else: return 4


def scale_relevance(sim):
    if sim < 0.15: return 0
    elif sim < 0.35: return 1
    elif sim < 0.55: return 2
    elif sim < 0.75: return 3
    else: return 4


def predict_text(text, prev_text=None):
    f = extract_features(text, prev_text)

    result = {}
    confidence = {}

    for facet in FACETS.keys():

        if facet == "toxicity":
            result[facet] = int(f['toxicity'] * 4)
            confidence[facet] = round(f['toxicity'], 2)

        elif facet == "sentiment":
            val = scale_sentiment(f['sentiment'])
            result[facet] = val
            confidence[facet] = round(abs(f['sentiment']), 2)

        elif facet == "politeness":
            val = min(4, int(f['politeness']*4 + f['empathy']*3))
            result[facet] = val
            confidence[facet] = round(min(1, f['politeness'] + f['empathy']), 2)

        elif facet == "relevance":
            val = scale_relevance(f['context_similarity'])
            result[facet] = val
            confidence[facet] = round(f['context_similarity'], 2)

        elif facet == "length_quality":
            val = min(f['word_count'] // 5, 4)
            result[facet] = val
            confidence[facet] = round(min(1, f['word_count']/20), 2)

    return {
        "scores": result,
        "confidence": confidence
    }


def explain_prediction(text, prev_text=None):
    output = predict_text(text, prev_text)
    pred = output["scores"]   # IMPORTANT FIX

    explanations = []

    if pred['toxicity'] >= 3:
        explanations.append("⚠️ High toxicity detected")

    if pred['politeness'] >= 3:
        explanations.append("✅ Polite tone detected")

    if pred['relevance'] <= 2:
        explanations.append("❌ Low relevance to context")

    elif pred['relevance'] == 3:
        explanations.append("⚠️ Partially relevant response")

    else:
        explanations.append("✅ Relevant to context")

    if pred['length_quality'] <= 1:
        explanations.append("⚠️ Response too short")

    if pred['sentiment'] == 0:
        explanations.append("😠 Negative sentiment detected")

    if pred['sentiment'] >= 3:
        explanations.append("😊 Positive sentiment")

    if len(explanations) == 0:
        explanations.append("Neutral response")

    return explanations