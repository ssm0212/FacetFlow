import streamlit as st
from model import predict_text, explain_prediction
from facets import FACETS   # 🔥 NEW (scalability)
import plotly.graph_objects as go

st.set_page_config(page_title="FacetFlow", layout="centered")

# -------------------------
# 🎯 HEADER
# -------------------------
st.markdown("""
### 🎯 Evaluate conversational responses across multiple facets:
- Toxicity  
- Sentiment  
- Politeness  
- Relevance  
- Quality  
""")

st.title("🔥 FacetFlow - Conversation Evaluator")

# -------------------------
# 💡 EXAMPLES
# -------------------------
st.markdown("### 💡 Try examples:")
st.code("I hate you, you are useless")
st.code("Thank you so much for your help!")

# -------------------------
# ✍ INPUT
# -------------------------
text = st.text_area("Enter Response")
prev = st.text_area("Previous Context (Optional)")

# -------------------------
# 🚀 EVALUATE
# -------------------------
if st.button("Evaluate"):

    output = predict_text(text, prev)
    scores = output["scores"]
    confidence = output["confidence"]

    # -------------------------
    # 📊 SCORES (CONFIG BASED)
    # -------------------------
    st.subheader("📊 Scores")

    for facet in FACETS.keys():
        st.write(f"**{facet.capitalize()}**: {scores[facet]}/4")

    # -------------------------
    # 🔍 CONFIDENCE
    # -------------------------
    st.subheader("🔍 Confidence")

    for facet in FACETS.keys():
        st.write(f"**{facet.capitalize()}**: {confidence[facet]}")

    # -------------------------
    # 📈 RADAR CHART
    # -------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[scores[f] for f in FACETS.keys()],
        theta=list(FACETS.keys()),
        fill='toself',
        name="Scores"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 4])
        ),
        showlegend=False
    )

    st.plotly_chart(fig)

    # -------------------------
    # 🧠 EXPLANATION
    # -------------------------
    st.subheader("🧠 Explanation")

    explanations = explain_prediction(text, prev)

    for e in explanations:
        st.write("-", e)