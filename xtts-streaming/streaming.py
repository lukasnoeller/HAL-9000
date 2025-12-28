import requests
import wave
import base64
import os
def encode_audio(file_path):
    with open(file_path, "rb") as audio_file:
        # Read the binary data
        binary_data = audio_file.read()
        # Encode to base64 bytes, then decode to a UTF-8 string
        base64_string = base64.b64encode(binary_data).decode('utf-8')
        return base64_string

url = "https://model-dq45k793.api.baseten.co/development/predict"
headers = {"Authorization": "Api-Key DaWmKDy1.Kyq337CfobGtl1Vyvt1XlCom8LsKDIzv"}
file_path = os.path.join("..", "reference_audio_files", "HAL9000_Voice_noise_reduced.wav")
encoded_audio_file = encode_audio(file_path)
payload = {"text": "I'm afraid, Dave",
           "audio_b64": encoded_audio_file}

channels = 1  # mono=1, stereo=2
sampwidth = 2  # Sample width in bytes, typical values: 2 for 16-bit audio, 1 for 8-bit audio
framerate = 24000  # Sampling rate, in samples per second (Hz)

print("Requesting audio...")
resp = requests.post(url, headers=headers, json=payload)

# DEBUG: Check what the server actually said
print(f"Status Code: {resp.status_code}")
print(f"Content Type: {resp.headers.get('Content-Type')}")

if resp.status_code == 200:
    data = resp.json()
    
    # Baseten usually puts the output in a key named 'model_output' or 'data'
    # Based on common XTTS deployments, it's often 'model_output'
    try:
        raw_audio_64 = data["output"]
        
        # Decode the Base64 string to bytes
        audio_bytes = base64.b64decode(raw_audio_64)
        
        with open("hal_decoded.wav", "wb") as f:
            f.write(audio_bytes)
        print("Success! Saved as hal_decoded.wav")
        
    except KeyError:
        print("Could not find audio key in JSON. Response keys were:")
        print(data.keys())
else:
    print(f"Error {resp.status_code}: {resp.text}")