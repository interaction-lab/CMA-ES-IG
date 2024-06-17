import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
from tqdm import tqdm
import pickle
import torchaudio

embeds = pickle.load(open('./data/speaker_info.pkl', 'rb'))

print("Loading model...")
config = XttsConfig()
config.load_json("./xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./xtts_v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for key, vals in tqdm(embeds.items()):
    gpt_cond_latent, speaker_embedding = vals

    out = model.inference(
        "Hi! my name is blossom, and I will be your mindfulness companion.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    torchaudio.save(f"./data/generated_voice/{key[:-4]}.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

    