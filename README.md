# 🔥 FacetFlow - Scalable Conversation Evaluation System

## 📌 Overview

FacetFlow is a scalable conversation evaluation system that scores responses across multiple facets such as:

- Toxicity  
- Sentiment  
- Politeness  
- Relevance  
- Length Quality  

The system is designed to support **300–5000+ evaluation facets** using a modular, configuration-driven architecture.

---

## 🚀 Features

- ✅ Multi-facet scoring system  
- ✅ Config-driven architecture supporting **300 → 5000+ facets**  
- ✅ Hybrid evaluation (rule-based + embedding-based)  
- ✅ Confidence scores for every facet prediction  
- ✅ Explainable predictions with human-readable reasoning  
- ✅ Interactive Streamlit UI with radar visualization  
- ✅ Auto-labeled dataset (50+ diverse conversations)  
- ✅ Lightweight system (no heavy LLM dependency)  


---

## 🧠 Architecture

FacetFlow uses a **hybrid pipeline combining symbolic rules and semantic embeddings**:

### 1. Feature Extraction
- Sentiment analysis using TextBlob  
- Semantic similarity using Sentence Transformers (MiniLM)  
- Rule-based detection:
  - toxicity (keywords)
  - politeness & empathy  
- Length-based response quality  

---

### 2. Scoring Layer

Each extracted signal is mapped into a **discrete score (0–4 scale)** using calibrated functions.

Example:
- Toxicity → scaled keyword intensity  
- Sentiment → polarity → score  
- Relevance → cosine similarity → score  

---

### 3. Confidence Layer (Key Innovation)

Each facet is assigned a **confidence score (0–1)** indicating prediction reliability.

---

### 4. Facet System (Core Design)

Facets are dynamically processed using a **configuration-based system**, enabling flexible and scalable evaluation.

---

## 🧩 Scalable Facet Architecture

The system is designed to scale seamlessly to **5000+ facets**.

### Configuration-Based Design

```python
FACETS = {
    "toxicity": {"group": "safety"},
    "sentiment": {"group": "emotion"},
    "politeness": {"group": "pragmatics"},
    "relevance": {"group": "pragmatics"},
    "length_quality": {"group": "linguistic"}
}
```
## 📦 Dataset

The dataset containing 50+ conversations is provided in:

`data/conversations.zip`

It includes diverse cases with corresponding evaluation scores.

