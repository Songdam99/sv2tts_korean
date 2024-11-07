import os
import torch
import atexit
import pickle
import librosa
import argparse
import parselmouth
import numpy as np
from pathlib import Path
from utils import logmmse
from pydub import AudioSegment
from parselmouth.praat import call
from synthesizer.hparams import hparams
from encoder import inference as encoder


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device : {device}')

sr=16000

def get_onset_and_offset(wav, relative_threshold_ratio=0.75):
    sound = parselmouth.Sound(wav)
    intensity = sound.to_intensity()

    # 최대 강도 계산
    # max_intensity = intensity(intensity, "Get maximum", 0, 0, "Parabolic")
    max_intensity = intensity.get_maximum()  # 파라미터 없이 호출
    relative_threshold = max_intensity * relative_threshold_ratio  # 필요에 따라 조정

    onsets = []
    offsets = []
    is_speaking = False  # 발화 상태를 추적

    # 10ms 간격으로 강도 체크
    time = intensity.xmin
    while time < intensity.xmax:
        intensity_value = call(intensity, "Get value at time", time, "Nearest")

        if intensity_value > relative_threshold:
            if not is_speaking:  # 발화 시작
                onsets.append(time)
                is_speaking = True
        else:
            if is_speaking:  # 발화 종료
                offsets.append(time)
                is_speaking = False

        time += 0.01

    return onsets, offsets, sound.xmax

def denoise(wav, end_times, hparams):
    # Load the audio waveform
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    
    # Find pauses that are too long
    mask = (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)
    
    return wav


def check_file_type_and_change_to_wav(ref_path):
    file_type = os.path.basename(str(ref_path)).split('.')[-1]
    if file_type == 'm4a':
        audio = AudioSegment.from_file(ref_path, format="m4a")
        export_path = f"converted_{os.path.splitext(os.path.basename(ref_path))[0]}.wav" 
        audio.export(export_path, format="wav")
        wav, _ = librosa.load(export_path, sr=sr)
        return wav
    elif file_type == 'wav':
        wav, _ = librosa.load(ref_path, sr=sr)
    else:
        raise ValueError("Unsupported file type. Please use 'm4a' or 'wav'.")

    return wav
    

# 파일에 임베딩 저장하는 함수
def save_embedding(embedding, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(embedding, f)


def delete_temp_file(temp_file_path):
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def main(ref_voice_path, hash):
    encoder_model_path = Path(__file__).parent / 'models' / 'encoder.pt'
    # encoder_model_path = Path(os.path.join(os.pardir(__file__), 'models', 'encoder.pt'))
    enc = encoder.load_model(encoder_model_path, device)

    wav = check_file_type_and_change_to_wav(ref_voice_path)

    # 부분 발화 시작, 끝 탐지
    onsets, offsets, _ = get_onset_and_offset(wav)

    # 부분 발화의 endtime, 부분 발화 사이 공백의 endtime 종합
    endtimes = []
    for i in range(len(onsets)):
        if i==len(onsets)-1:
            endtimes.append(offsets[i])
        else:
            endtimes.append(offsets[i])
            endtimes.append(onsets[i+1])

    # endtime 기반 노이즈 제거
    wav = denoise(wav, endtimes, hparams)
    
    # embedding 추출
    wav = wav / np.abs(wav).max() * hparams.rescaling_max
    encoder_wav = encoder.preprocess_wav(wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # embedding 저장
    splitted = hash.split('_')
    output_path = Path(__file__).parent / 'embeddings' / f'{splitted[0]}.pkl'
    save_embedding(embed, output_path)

    # ref가 wav파일 형식이 아니었다면 wav로 변환한 임시 파일 제거
    converted_path = f'converted_{os.path.splitext(os.path.basename(ref_voice_path))[0]}.wav'
    atexit.register(lambda: delete_temp_file(converted_path))
    
    return embed


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_voice_path', type=Path, help=\
        "Path to the user's reference voice.")
    parser.add_argument('--hash', type=str, help=\
        "User-specific hash string. ex) 3A184B")
    args = parser.parse_args()

    main(*vars(args).values())