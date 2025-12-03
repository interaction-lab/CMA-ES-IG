import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import pickle
from tqdm import tqdm

print("Loading model...")
config = XttsConfig()
config.load_json("./xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./xtts_v2")

speaker_info = {}

files = os.listdir('./data/short_clips')
for file in tqdm(files):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[f"./data/short_clips/{file}"])
    speaker_info[file] = (gpt_cond_latent, speaker_embedding)
    

pickle.dump(speaker_info, open("./data/speaker_info.pkl", "wb"))