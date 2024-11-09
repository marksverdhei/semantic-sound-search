import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from embeddings import processor, model, TARGET_SAMPLE_RATE
from search import build_index  # Fix the import statement

def load_and_preprocess_dataset():
    # Load and preprocess a subset of the dataset for the demonstration
    dataset = load_dataset("ashraq/esc50", split="train[:10%]")  # Load only 10% for POC
    audio_embeddings = []

    # Generate embeddings for the audio dataset
    print("Generating audio embeddings for dataset...")

    for sample in dataset:
        audio_array = sample["audio"]["array"]
        # Resample to 48kHz if necessary and ensure float32 type
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Convert to float32 tensor
        audio_resampled = torchaudio.transforms.Resample(
            orig_freq=sample["audio"]["sampling_rate"], new_freq=TARGET_SAMPLE_RATE
        )(audio_tensor)
        audio_resampled = audio_resampled.squeeze().numpy().astype(np.float32)  # Convert to float32 numpy array

        # Process with CLAP
        inputs = processor(audios=audio_resampled, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").to("cuda")
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs).cpu().numpy()
        audio_embeddings.append(audio_embed)

    # Convert list of embeddings to a FAISS index
    audio_embeddings = np.vstack(audio_embeddings).astype("float32")
    index = build_index(audio_embeddings)
    print("Index built successfully.")
    
    return dataset, index