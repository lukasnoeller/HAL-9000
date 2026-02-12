import logging
import os

import numpy as np
import torch
import base64
import io
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
import sys
import pydantic

# Check if we are running Pydantic V2
if pydantic.VERSION.startswith("2"):
    print("Detected Pydantic V2. Applying V1 compatibility shim for DeepSpeed/XTTS...")
    from pydantic import v1 as pydantic_v1
    
    # This is the "magic" line: it tells Python that whenever any library 
    # (like DeepSpeed) tries to 'import pydantic', it should get the V1 version.
    sys.modules["pydantic"] = pydantic_v1
# This is one of the speaker voices that comes with xtts
SPEAKER_NAME = "Hal-9000"


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.speaker = None

    def load(self):
        device = "cuda"
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        # ModelManager().download_model(model_name)
        # model_path = os.path.join(
        #     get_user_data_dir("tts"), model_name.replace("/", "--")
        # )
        # Get the path to the 'checkpoints' folder inside your Truss directory
        # 'model_module_dir' in Truss is usually where this script lives
        model_path = os.path.join("/app/data", "checkpoints/xtts_v2")
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        logging.info("Loading model from disk...")
        self.model.load_checkpoint(
            config, checkpoint_dir=model_path, eval=True, use_deepspeed=True
        )
        self.model.to(device)
        # self.compiled_model = torch.compile(self.model.inference_stream)

        self.speaker = {
            "speaker_embedding": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "speaker_embedding"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
            "gpt_cond_latent": self.model.speaker_manager.speakers[SPEAKER_NAME][
                "gpt_cond_latent"
            ]
            .cpu()
            .squeeze()
            .half()
            .tolist(),
        }

        self.speaker_embedding = (
            torch.tensor(self.speaker.get("speaker_embedding"))
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.gpt_cond_latent = (
            torch.tensor(self.speaker.get("gpt_cond_latent"))
            .reshape((-1, 1024))
            .unsqueeze(0)
        )
        logging.info("🔥Model Loaded")

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def predict(self, model_input):
        text = model_input.get("text")
        language = model_input.get("language", "en")
        chunk_size = int(
            model_input.get("chunk_size", 20)
        )  # Ensure chunk_size is an integer
        add_wav_header = False
        # --- NEW: VOICE CLONING LOGIC ---
        reference_audio_b64 = model_input.get("audio_b64")

        if reference_audio_b64:
            logging.info("Generating latents for custom voice...")
            # Decode the base64 audio and load it with torchaudio
            audio_bytes = base64.b64decode(reference_audio_b64)
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Save to a temp file because XTTS get_conditioning_latents expects a path
            temp_path = "/tmp/ref_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            # Compute latents from the provided audio
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[temp_path]
            )
        else:
            # Fallback to the default speaker
            gpt_cond_latent = self.default_gpt_cond_latent
            speaker_embedding = self.default_speaker_embedding
        streamer = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True,
            temperature=0.2,
        )

        for chunk in streamer:
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes
