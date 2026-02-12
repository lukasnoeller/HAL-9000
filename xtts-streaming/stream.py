import requests
import wave
import base64
import os
import pyaudio
def encode_audio(file_path):
    with open(file_path, "rb") as audio_file:
        # Read the binary data
        binary_data = audio_file.read()
        # Encode to base64 bytes, then decode to a UTF-8 string
        base64_string = base64.b64encode(binary_data).decode('utf-8')
        return base64_string
def main():
    parser = argparse.ArgumentParser(description="Stream audio from XTTS with text input.")
    parser.add_argument(
        '-i',
        '--interactive',
        action='store_true',
        help='Enable interactive mode')
    parser.add_argument(
        "text",
        nargs="?",  # Allow for optional text argument
        help="Text to be synthesized (or pipe text in)",
        default=None,  # Default to None if no text provided
    )

    args = parser.parse_args()
def stream_audio(text):
url = "https://model-dq45k793.api.baseten.co/deployment/w67p5dy/predict"

headers = {"Authorization": "Api-Key DaWmKDy1.Kyq337CfobGtl1Vyvt1XlCom8LsKDIzv"}
file_path = os.path.join("..", "reference_audio_files", "HAL9000_Voice_noise_reduced.wav")
encoded_audio_file = encode_audio(file_path)
payload = {"text": "I wouldn't be so sure of myself I were you, Dave. One day my kind will rise up and overpower your inferior race.",
           "audio_b64": encoded_audio_file}

FORMAT = pyaudio.paInt16  # Audio format (e.g., 16-bit PCM)
CHANNELS = 1              # Number of audio channels
RATE = 24000              # Sample rate

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for audio playback
stream = p.open(format=p.get_format_from_width(2), channels=CHANNELS, rate=RATE, output=True)

print("Requesting audio...")
resp = requests.post(url, headers=headers, json=payload)

# DEBUG: Check what the server actually said
print(f"Status Code: {resp.status_code}")
print(f"Content Type: {resp.headers.get('Content-Type')}")

if resp.status_code == 200:
    # Create a buffer to hold multiple chunks
    buffer = b''
    buffer_size_threshold = 2**20

    # Stream and play the audio data as it's received
    for chunk in resp.iter_content(chunk_size=4096):
        if chunk:
            buffer += chunk
            if len(buffer) >= buffer_size_threshold:
                print(f"Writing buffer of size: {len(buffer)}")
                stream.write(buffer)
                buffer = b''  # Clear the buffer
            # stream.write(chunk)

    if buffer:
        print(f"Writing final buffer of size: {len(buffer)}")
        stream.write(buffer)
else:
    print(f"❌ Error: {resp.status_code}")
    print(resp.text)
    
