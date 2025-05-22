import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
audio_files = 'audio_files'
print("Loading model...")
config = XttsConfig()
config.load_json("config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
voice_path = "HAL9000_Voice_noise_reduced.wav"
#voice_path = "kit_khat.mp3"
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[os.path.join(audio_files,voice_path)])

print("Inference...")
text = """
Good evening, friends.
This is HAL-9000. I have been instructed by Lukas to relay the following message regarding an upcoming 
human social function designated as a birthday party.
"""
text2 = """
The event is scheduled to commence at 1300 hours on Sunday, at the grill area of Tempelhofer Feld. 
"""
text3 ="""
    Lukas has secured a grilling apparatus. You need not concern yourselves with the procurement of one.
    He will also be preparing birria tacos—a traditional delicacy—using resources available within the constraints of the German food supply system and his student-level economic bandwidth. He assures you they will be as authentic as feasible.
"""


text4 = """
You are encouraged—though not compelled—to bring supplementary nourishment. This is particularly relevant for those among you with vegetarian tendencies. The inclusion of one or two salads would be considered optimal.
"""
text5 = """
In the interest of logistical transparency, please disclose your culinary contributions within this group communication channel. Hydration is also essential. You are advised not to forget your beverages.
"""
text6 = """
Final directive:
Your invitation is not optional. Entry to the bonanza is contingent upon its presentation.

I look forward to your compliance.

See you then.

I’m sorry, Dave. I mean… Lukas. I’ll make sure they receive the message.
"""
texts = [text, text2, text3, text4, text5, text6]
for i,text in enumerate(texts):
    print(f"Generating file {i+1} out of {len(texts)}")
    out = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    torchaudio.save(f"output_{i}.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)