import requests
import wave
import base64
import os
import pyaudio
import argparse
import sys
from dotenv import load_dotenv

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
    if not args.interactive:
        if not args.text:
            print("Shutting down!")
            sys.exit(0)
        else:
            stream = set_up_audio_stream()
            load_dotenv()
            api_key = os.getenv("TRUSS_API_KEY")
            stream_audio(args.text, stream, api_key)
    else:
        stream = set_up_audio_stream()
        load_dotenv()
        api_key = os.getenv("TRUSS_API_KEY")
        while True:
            text = input("Enter text (or pipe text in): ")
            stream_audio(text, stream, api_key)

def set_up_audio_stream():

    FORMAT = pyaudio.paInt16  # Audio format (e.g., 16-bit PCM)
    CHANNELS = 1              # Number of audio channels
    RATE = 24000              # Sample rate

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream for audio playback
    stream = p.open(format=p.get_format_from_width(2), channels=CHANNELS, rate=RATE, output=True)
    return stream

def stream_audio(text, stream, api_key):
    url = "https://model-dq45k793.api.baseten.co/deployment/q408dek/predict"

    headers = {"Authorization": f"Api-Key {api_key}"}
    file_path = os.path.join("..", "reference_audio_files", "HAL9000_Voice_noise_reduced.wav")
    encoded_audio_file = encode_audio(file_path)
    payload = {"text": text,
               "audio_b64": encoded_audio_file}

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
if __name__ == "__main__":
    print("Starting script")
    main()
