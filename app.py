import streamlit as st
import numpy as np
from src.embeddings import get_text_embedding, get_audio_embedding
from src.search import search_embedding
from src.dataset import load_and_preprocess_dataset
from transformers import ClapModel, ClapProcessor

# Cache the model and processor loading
@st.cache_resource
def load_model_and_processor():
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    return model, processor

# Cache the dataset loading and preprocessing
@st.cache_resource
def load_dataset_and_index(_model, _processor):
    return load_and_preprocess_dataset(_model, _processor)

# Load model, processor, dataset, and index
model, processor = load_model_and_processor()
dataset, index = load_dataset_and_index(model, processor)

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
        embedding = get_audio_embedding(audio_bytes, model, processor)
    elif query_text:
        # Process the text query
        embedding = get_text_embedding(query_text, model, processor)
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