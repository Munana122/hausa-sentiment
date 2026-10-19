import streamlit as st
from transformers import pipeline

MODEL_REPO = "Merveille123/hausa-sentiment-afriberta"


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_REPO,
        truncation=True,
        top_k=None,
    )


st.title("Hausa Tweet Sentiment Classifier")
st.caption("AfriBERTa fine-tuned on NaijaSenti (Hausa). Labels: positive / neutral / negative.")

text = st.text_area("Enter a Hausa tweet or sentence", height=100)

if st.button("Classify") and text.strip():
    clf = load_model()
    results = clf(text)
    if results and isinstance(results[0], list):
        results = results[0]
    for r in sorted(results, key=lambda x: -x["score"]):
        label = r["label"]
        score = r["score"]
        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(label.lower(), "⚪")
        st.write(f"{icon} **{label}**: {score:.1%}")

st.markdown("---")
st.markdown("**Example tweets from the NaijaSenti test set** (labels from the dataset):")
st.markdown("- *PASTE POSITIVE TWEET* (positive)")
st.markdown("- *PASTE NEUTRAL TWEET* (neutral)")
st.markdown("- *PASTE NEGATIVE TWEET* (negative)")
