import sys
import pyaudio
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import argparse


def load_model():
    print("Loading model...")
    config = XttsConfig()
    config.load_json("config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir="xtts-streaming/data/checkpoints/xtts_v2",
        use_deepspeed=False,
    )
    model.cuda()

    speaker = {
        "speaker_embedding": model.speaker_manager.speakers["Hal-9000"][
            "speaker_embedding"
        ]
        .cpu()
        .squeeze()
        .half()
        .tolist(),
        "gpt_cond_latent": model.speaker_manager.speakers["Hal-9000"]["gpt_cond_latent"]
        .cpu()
        .squeeze()
        .half()
        .tolist(),
    }

    speaker_embedding = (
        torch.tensor(speaker.get("speaker_embedding")).unsqueeze(0).unsqueeze(-1)
    )
    gpt_cond_latent = (
        torch.tensor(speaker.get("gpt_cond_latent")).reshape((-1, 1024)).unsqueeze(0)
    )

    return model, gpt_cond_latent, speaker_embedding


def generate_audio(text, model, gpt_cond_latent, speaker_embedding):
    # file_path = os.path.join("..", "reference_audio_files", "HAL9000_Voice_noise_reduced.wav")
    # encoded_audio_file = encode_audio(file_path)

    FORMAT = pyaudio.paInt16  # Audio format (e.g., 16-bit PCM)
    CHANNELS = 1  # Number of audio channels
    RATE = 24000  # Sample rate

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream for audio playback
    stream = p.open(format=pyaudio.paFloat32, channels=CHANNELS, rate=RATE, output=True)

    print("Generating audio...")
    # Create a buffer to hold multiple chunks
    buffer = b""
    buffer_size_threshold = 2**20
    streamer = model.inference_stream(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=4096,
        enable_text_splitting=True,
        temperature=0.2,
    )
    # Stream and play the audio data as it's received
    for chunk in streamer:
        if chunk is not None:
            chunk_bytes = chunk.cpu().numpy().tobytes()
            buffer += chunk_bytes
            if len(buffer) >= buffer_size_threshold:
                print(f"Writing buffer of size: {len(buffer)}")
                stream.write(buffer)
                buffer = b""  # Clear the buffer
            # stream.write(chunk)
    if buffer:
        print(f"Writing final buffer of size: {len(buffer)}")
        stream.write(buffer)

    # if buffer:
    #     print(f"Writing final buffer of size: {len(buffer)}")
    #     stream.write(buffer)


def main():
    parser = argparse.ArgumentParser(
        description="Stream audio from XTTS with text input."
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive mode"
    )
    parser.add_argument(
        "text",
        nargs="?",  # Allow for optional text argument
        help="Text to be synthesized (or pipe text in)",
        default=None,  # Default to None if no text provided
    )

    model, gpt_cond_latent, speaker_embedding = load_model()
    args = parser.parse_args()

    if not args.interactive:
        if not args.text:
            print("Shutting down!")
            sys.exit(0)
        else:
            generate_audio(args.text, model, gpt_cond_latent, speaker_embedding)
    else:
        while True:
            text = input("Enter text (or pipe text in): ")
            generate_audio(text, model, gpt_cond_latent, speaker_embedding)


if __name__ == "__main__":
    main()
