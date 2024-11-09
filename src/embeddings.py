import torch
import soundfile as sf
import torchaudio
from transformers import ClapModel, ClapProcessor
import numpy as np
# Define target sample rate
TARGET_SAMPLE_RATE = 48000

# Load model and processor
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

def get_text_embedding(text, model, processor):
    inputs = processor(text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs).cpu().numpy()
    return text_embed

def get_audio_embedding(file_path, model, processor, audio_array=None, sample_rate=None):
    if file_path:
        audio_array, sample_rate = sf.read(file_path)
    elif audio_array is not None and sample_rate is not None:
        pass
    else:
        raise ValueError("Either file_path or both audio_array and sample_rate must be provided")

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