import torch
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Paths
checkpoint_dir = "xtts-streaming/model/checkpoints/xtts_v2"
speaker_file = os.path.join(checkpoint_dir, "speakers_xtts.pth")
reference_wav = "reference_audio_files/HAL9000_Voice_noise_reduced.wav"

print("Loading config...")
config = XttsConfig()
config.load_json(os.path.join(checkpoint_dir, "config.json"))
model = Xtts.init_from_config(config)

# MANUAL SURGERY: Instead of model.load_checkpoint, we do it manually
print("Performing manual weight load (bypassing security)...")
checkpoint = torch.load(os.path.join("XTTS-v2.0-HAL-9000", "model.pth"), weights_only=False)
model.load_state_dict(checkpoint["model"], strict=False)
model.to("cpu")

# Extract HAL's latents
print(f"Extracting embeddings for {reference_wav}...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_wav])

# Update the speaker file
print("Updating speakers_xtts.pth...")
speaker_dict = torch.load(speaker_file, weights_only=False)

speaker_dict["Hal-9000"] = {
    "gpt_cond_latent": gpt_cond_latent,
    "speaker_embedding": speaker_embedding,
}

torch.save(speaker_dict, speaker_file)
print("Success! HAL-9000 is now built-in.")