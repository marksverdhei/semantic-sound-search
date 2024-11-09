import torch
import faiss
import numpy as np
import torchaudio
from transformers import ClapModel, ClapProcessor
from datasets import load_dataset
import argparse
import soundfile as sf

# Define target sample rate
TARGET_SAMPLE_RATE = 48000

# Load model and processor
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

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
index = faiss.IndexFlatL2(audio_embeddings.shape[1])  # Using L2 distance
index.add(audio_embeddings)
print("Index built successfully.")

# Function to get embeddings for input text
def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs).cpu().numpy()
    return text_embed

# Function to get embeddings for input audio
def get_audio_embedding(file_path):
    audio_array, sample_rate = sf.read(file_path)
    # Resample to 48kHz if necessary and ensure float32 type
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Convert to float32 tensor
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_resampled = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )(audio_tensor)
        audio_array = audio_resampled.squeeze().numpy().astype(np.float32)  # Convert to float32 numpy array

    # Process with CLAP
    inputs = processor(audios=audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt").to("cuda")
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs).cpu().numpy()
    return audio_embed

# Function to perform search
def search_embedding(embedding, top_n=5):
    distances, indices = index.search(embedding, top_n)
    return distances[0], indices[0]

# CLI implementation
def main():
    parser = argparse.ArgumentParser(description="Semantic Search in Audio Files using CLAP and FAISS.")
    parser.add_argument("--input", default="a dog barking", type=str, help="Input text or path to audio file")
    parser.add_argument("--top_n", type=int, default=5, help="Number of closest matches to return")
    args = parser.parse_args()

    # Check if input is text or a file path
    if args.input.endswith(".wav"):
        print(f"Processing audio file: {args.input}")
        embedding = get_audio_embedding(args.input)
    else:
        print(f"Processing text input: {args.input}")
        embedding = get_text_embedding(args.input)

    distances, indices = search_embedding(embedding, args.top_n)
    print("\nTop matches:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        idx = int(idx)
        print(f"Match {i + 1}:")
        print(f" - Distance: {dist}")
        print(f" - Audio ID: {dataset[idx]['filename']}")
        print(f" - Category: {dataset[idx]['category']}")

if __name__ == "__main__":
    main()
