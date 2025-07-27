import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import torch

# Load ML model and genre encoder
model = joblib.load("success_model.pkl")
mlb = joblib.load("genre_encoder.pkl")

# Load SentenceTransformer model
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Load movie plot dataset and encode it
plots_df = pd.read_csv("movie_plots_sample.csv")
plots_df.dropna(inplace=True)
plots_df['embedding'] = plots_df['Plot'].apply(lambda x: sbert.encode(x, convert_to_tensor=True))

# Streamlit UI
st.title("🎬 Movie Success Predictor & Copycat Detector")

st.markdown("Fill in your movie details to get AI-based success prediction and plot similarity check.")

title = st.text_input("Movie Title")
genre = st.multiselect("Genres", mlb.classes_.tolist())
cast = st.text_area("Cast (comma-separated)")
budget = st.slider("Budget (in million USD)", 10, 300, 100)
runtime = st.slider("Runtime (in minutes)", 60, 200, 120)
popularity = st.slider("Popularity (0–100)", 0.0, 100.0, 50.0)
plot = st.text_area("Plot Summary")

if st.button("🎯 Predict & Detect"):
    # --- Success Prediction (ML)
    genre_encoded = mlb.transform([genre])[0]
    input_data = list(genre_encoded) + [budget * 1_000_000, runtime, popularity, len(cast.split(','))]
    df_input = pd.DataFrame([input_data], columns=model.feature_names_in_)

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.subheader("🎯 Success Prediction")
    if pred == 1:
        st.success(f"✅ Likely to Succeed! Confidence: {round(proba*100, 2)}%")
    else:
        st.error(f"❌ Might Not Succeed. Confidence: {round(proba*100, 2)}%")

    # --- Copycat Detection (NLP)
    st.subheader("🕵️ Copycat Detector")
    if plot:
        input_emb = sbert.encode(plot, convert_to_tensor=True)
        similarities = [float(util.pytorch_cos_sim(input_emb, emb)[0][0]) for emb in plots_df['embedding']]
        top_idx = int(torch.argmax(torch.tensor(similarities)))
        top_score = round(similarities[top_idx] * 100, 2)
        top_movie = plots_df.iloc[top_idx]['Title']

        if top_score > 75:
            st.error(f"⚠️ High similarity with: {top_movie} ({top_score}%)")
        else:
            st.success(f"✅ Looks original! Closest match: {top_movie} ({top_score}%)")