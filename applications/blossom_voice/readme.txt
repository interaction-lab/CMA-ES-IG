All speaker embeddings are calculated from the VoxCeleb voices in "data/short_clips"
These are stored in speaker info as a tuple of (gpt_cond_latent, speaker_embedding) with
the original file that was used to generate them as the key.

e.g., to get the embeddings for speaker with id00015, load in speaker_info.pkl, and access
the values for "id00015.wav", which will return two arrays.

GItHub doesn't allow the storage of large files, so you will have to download coqui
xtts-v2 from this link and place it in the xtts_v2 folder:
https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth

you will also have to generate the speaker embeddings with `generate_speaker_embeds.py`


