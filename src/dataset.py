import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from src.embeddings import get_audio_embedding, TARGET_SAMPLE_RATE
from src.search import build_index

def load_and_preprocess_dataset(model, processor):
    # Load and preprocess a subset of the dataset for the demonstration
    dataset = load_dataset("ashraq/esc50", split="train[:10%]")  # Load only 10% for POC
    audio_embeddings = []

    # Generate embeddings for the audio dataset
    print("Generating audio embeddings for dataset...")

    for sample in dataset:
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        audio_embed = get_audio_embedding(None, model, processor, audio_array, sample_rate)
        audio_embeddings.append(audio_embed)

    # Convert list of embeddings to a FAISS index
    audio_embeddings = np.vstack(audio_embeddings).astype("float32")
    index = build_index(audio_embeddings)
    print("Index built successfully.")
    
    return dataset, index