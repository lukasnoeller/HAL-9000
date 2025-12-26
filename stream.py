import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import queue  # Use a queue for thread-safe data transfer
import argparse

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

    # Get text input (from argument or stdin)
    if not args.interactive:
        print("Shutting down!")
        #text = args.text
    else:
        #text = input("Enter text (or pipe text in): ")
        # Load audio data and sample rate
        data, sr = sf.read("HAL9000 just a moment_background_reduced.mp3", dtype='float32')

        sd.play(data, sr)
        print("Loading model...")
        config = XttsConfig()
        config.load_json("config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=".", use_deepspeed=True)
        model.cuda()

        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["audio_files/im_afraid.ogg"])
        sd.play(data, sr)
        while True:
            user_input = input("Enter your text (or type 'quit' to exit): ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            print("Inference...")
            t0 = time.time()
            print(f"Token length = {len(user_input)}")
            # if len(user_input) > 250:
            #     print("ERROR: TOO MANY TOKENS")
            chunks = model.inference_stream(
                user_input,
                "en",
                gpt_cond_latent,
                speaker_embedding
            )


            
            chunks_buffer = []
            wav_chunks = []
            with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as stream:
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        print(f"Time to first chunk: {time.time() - t0}")
                    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                    chunks_buffer.append(chunk)
                    wav_chunks.append(chunk)
                    if len(chunks_buffer) < 5:
                        # Buffer until you have at least 2 chunks
                        continue
                    
                    if len(chunks_buffer) == 5:
                        # Start playback by writing both buffered chunks
                        for buffered_chunk in chunks_buffer:
                            stream.write(buffered_chunk.squeeze().cpu().numpy())
                        chunks_buffer = []
                   #
                        # For subsequent chunks, write directly
                    #stream.write(chunk.squeeze().cpu().numpy())
                for buffered_chunk in chunks_buffer:
                    stream.write(buffered_chunk.squeeze().cpu().numpy())

                    
            
                    

            for chunk in wav_chunks:
                sd.play(chunk.squeeze().cpu().numpy(), 24000)
                sd.wait() 
            wav = torch.cat(wav_chunks, dim=0)
            torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)


if __name__ == "__main__":
    main()