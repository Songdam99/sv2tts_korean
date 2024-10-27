import os
import glob
from pathlib import Path
from itertools import chain
import parselmouth
from tqdm import tqdm
from parselmouth.praat import call
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_waveform(wav_path, onset, offset):
    # 오디오 파일 읽기
    sample_rate, data = wavfile.read(wav_path)
    
    # 시간 축 생성
    time = [i / sample_rate for i in range(len(data))]
    
    # 파형 시각화
    wav_name = wav_path.split('\\')[-1]
    plt.figure(figsize=(12, 4))
    plt.plot(time, data, color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"{wav_name} pred: {onset} ~ {offset}")
    plt.grid()
    plt.show()

    start_index = int(onset * sample_rate)
    end_index = int(offset * sample_rate)

    # 데이터 슬라이싱
    segment = data[start_index:end_index]

    # 새로운 오디오 파일로 저장
    wavfile.write(f'./mfa/sliced_samples/{wav_name[:-4]}_sliced.wav', sample_rate, segment)


def get_onset_and_offset(wav_path, relative_threshold_ratio=0.75):
    sound = parselmouth.Sound(wav_path)
    intensity = sound.to_intensity()

    # 최대 강도 계산
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
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
    
    # plot_waveform(wav_path, onsets[0], offsets[-1])
    return onsets[0], offsets[-1], sound.xmax


def create_textgrid(output_path, start_times, end_times, texts):
    audio_duration = end_times[-1]
    # Write the header of the TextGrid file
    with open(output_path, "w", encoding="UTF-8") as f:
        f.write("File type = \"ooTextFile\"\n")
        f.write("Object class = \"TextGrid\"\n\n")
        f.write(f"xmin = 0\n")
        f.write(f"xmax = {audio_duration}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 1\n")
        f.write("item []:\n")
        f.write("    item [1]:\n")
        f.write("        class = \"IntervalTier\"\n")
        f.write("        name = \"sentence\"\n")
        f.write(f"        xmin = 0\n")
        f.write(f"        xmax = {audio_duration}\n")
        f.write(f"        intervals: size = {len(start_times)}\n")

        # Write each interval
        for i, (start, end, text) in enumerate(zip(start_times, end_times, texts), 1):
            f.write(f"        intervals [{i}]:\n")
            f.write(f"            xmin = {start}\n")
            f.write(f"            xmax = {end}\n")
            f.write(f"            text = \"{text}\"\n")

    # print(f"TextGrid saved to: {output_path}")

datasets_root = Path("C:/Users/admin/Desktop/Narrify_data/한국어 음성")
dataset_root = datasets_root.joinpath("KSponSpeech")
input_dirs = [dataset_root.joinpath("KsponSpeech_01"),
dataset_root.joinpath("KsponSpeech_02"),
dataset_root.joinpath("KsponSpeech_03"),
dataset_root.joinpath("KsponSpeech_04"),
dataset_root.joinpath("KsponSpeech_05")]

speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))

for speaker_dir in tqdm(speaker_dirs, desc="Processing Speaker Directories"):
    wav_dir = speaker_dir.parent.parent.parent / "KSponSpeech_wav" / speaker_dir.relative_to(speaker_dir.parents[1])
    wav_file_paths = list(wav_dir.rglob("*.wav"))
    alignment_path = speaker_dir / f"{speaker_dir.name}_alignment.txt"
    with open(alignment_path, 'r', encoding='cp949') as file:
        lines = file.readlines()
    sentences = [line.split(" ")[-1][1:-2] for line in lines]   # 쌍따옴표 제거
    for i, (wav_file_path, sentence) in enumerate(zip(wav_file_paths, sentences)):
        start, end, audio_duration = get_onset_and_offset(str(wav_file_path))
        filename = str(wav_file_path).split('\\')[-1][:-4]
        
        textgrid_path = str(speaker_dir)+f"/{filename}.TextGrid"
        start_times = [0, start, end]
        end_times = [start, end, audio_duration]
        texts = ["", sentences[i], ""]
        create_textgrid(textgrid_path, start_times, end_times, texts)