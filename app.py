import streamlit as st
import numpy as np
from src.embeddings import get_text_embedding, get_audio_embedding
from src.search import search_embedding
from src.dataset import load_and_preprocess_dataset

# Load and preprocess the dataset
dataset, index = load_and_preprocess_dataset()

st.title("Audio Semantic Search")

st.markdown("Upload an audio file or enter a text query to search for similar audio samples.")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

# Text input for query string
query_text = st.text_input("Or enter a text query")

# Number input for top N results
top_n = st.number_input("Number of closest matches to return", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if uploaded_file is not None:
        # Process the uploaded audio file
        st.audio(uploaded_file)
        audio_bytes = uploaded_file.read()
        embedding = get_audio_embedding(audio_bytes)
    elif query_text:
        # Process the text query
        embedding = get_text_embedding(query_text)
    else:
        st.error("Please upload an audio file or enter a text query.")
        st.stop()

    # Search for similar embeddings
    distances, indices = search_embedding(embedding, index, top_n)
    
    st.markdown("### Top matches:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        idx = int(idx)
        st.markdown(f"**Match {i + 1}:**")
        st.markdown(f"- **Distance:** {dist}")
        st.markdown(f"- **Audio ID:** {dataset[idx]['filename']}")
        st.markdown(f"- **Category:** {dataset[idx]['category']}")
        st.audio(dataset[idx]['audio']['array'], sample_rate=dataset[idx]['audio']['sampling_rate'])