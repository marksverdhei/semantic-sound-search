import argparse
from embeddings import get_text_embedding, get_audio_embedding
from search import search_embedding, build_index
from dataset import load_and_preprocess_dataset

# Load and preprocess the dataset
dataset, index = load_and_preprocess_dataset()

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

    distances, indices = search_embedding(embedding, index, args.top_n)
    print("\nTop matches:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        idx = int(idx)
        print(f"Match {i + 1}:")
        print(f" - Distance: {dist}")
        print(f" - Audio ID: {dataset[idx]['filename']}")
        print(f" - Category: {dataset[idx]['category']}")

if __name__ == "__main__":
    main()
