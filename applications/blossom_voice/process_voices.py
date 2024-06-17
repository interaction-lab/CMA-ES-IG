import os
from tqdm import tqdm
import librosa
import soundfile as sf

done = os.listdir('./data/short_clips')

for dir in tqdm(os.listdir('./data/aac')):
    # print(dir)
    voice_files = []

    if f'{dir}.wav' in done:
        continue

    for prefix, dirs , files in os.walk(f'./data/aac/{dir}'):
        if len(dirs) == 0:
            files = [f'{prefix}/{file}' for file in files if file.endswith('.m4a')]
            for file in files:
                voice_files.append(file)
        
    for file in voice_files:
        try:
            y,sr = librosa.load(file)
            duration = librosa.get_duration(y=y, sr=sr)

            if duration > 8 and duration < 12:
                sf.write(f'./data/short_clips/{dir}.wav', y, sr)
        except:
            print(f'Woopsie! Error in {file}')
