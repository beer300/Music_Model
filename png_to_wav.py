from PIL import Image
import numpy as np
from scipy.io.wavfile import write
import os

# Parameters
dpi = 120
sample_rate = 48000  # 48 kHz
hop_length = 1024
duration = 30  # Duration of audio in seconds
input_directory = "C:\\Users\\lukas\\Music\\conv" # Directory containing PNG files
output_directory = "C:\\Users\\lukas\\Music\\conv_wav" # Directory to save WAV files

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to process a single file
def process_file(png_path, wav_path):
    # Load and resize the PNG image
    image = Image.open(png_path)
    image = image.resize((2364, 1164))  # Resize to specified dimensions
    image = image.convert('L')  # Convert to grayscale

    # Normalize pixel data to range [-1, 1]
    pixel_data = np.array(image)
    normalized_data = (pixel_data / 255.0) * 2 - 1

    # Flatten the image data to a 1D array
    audio_data = normalized_data.flatten()

    # Scale the data to fit the hop length
    audio_data = np.interp(audio_data, (audio_data.min(), audio_data.max()), [-1, 1])

    # Repeat or pad to make it a valid waveform for WAV
    total_samples = sample_rate * duration
    audio_data = np.tile(audio_data, int(np.ceil(total_samples / len(audio_data))))[:total_samples]

    # Convert to 16-bit PCM format
    audio_data = (audio_data * 32767).astype(np.int16)

    # Save as a WAV file
    write(wav_path, sample_rate, audio_data)

# Iterate through all PNG files in the input directory
for file_name in os.listdir(input_directory):
    if file_name.endswith('.png'):
        png_path = os.path.join(input_directory, file_name)
        wav_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}.wav")
        process_file(png_path, wav_path)
        print(f"Processed {file_name} -> {wav_path}")
