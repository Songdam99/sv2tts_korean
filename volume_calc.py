import os
from tqdm import tqdm
from pathlib import Path
import wave
import glob

root_path = 'D:\GraduateProject\AI 음성 데이터'
data_subpath = '131.다국어 통·번역 낭독체 데이터/01-1.정식개방데이터/Training/01.원천데이터'
data_path = os.path.join(root_path, data_subpath)

print(f"dataset : {data_subpath.split('/')[0]}")
wav_path_list = [f for f in Path(data_path).glob('**/*.wav')]

total_num = 0
total_duration = 0
total_size = 0
for wav_path in tqdm(wav_path_list):
    with wave.open(str(wav_path), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        total_duration += duration
    total_num += 1
    total_size += os.path.getsize(wav_path)

print(f'total num wav files : {total_num}')
print(f'total duration : {total_duration} sec')
print(f'total file volume : {total_size/(1024*1024):.2f} MB')