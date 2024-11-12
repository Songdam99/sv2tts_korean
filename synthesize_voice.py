import os
import re
import argparse
import pickle
from synthesizer.inference import Synthesizer
from vocoder import inference as voc
from pathlib import Path
import numpy as np
from encoder.audio import preprocess_wav
import scipy.io.wavfile as wavfile

sentence_max_len = 50

import re

def split_text(text):
    # 문장부호로 문장을 나누기
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    result = []
    sentence_end_flag = []
    sentence_max_len = 50  # 한 문장의 최대 글자 수

    for sentence in sentences:
        chunks = []  # 분리된 덩어리들을 임시로 저장할 리스트

        # 문장이 50자를 넘으면 나누기
        while len(sentence) > sentence_max_len:
            split_point = sentence.rfind(" ", 0, sentence_max_len)
            if split_point == -1:  # 공백이 없을 경우 그냥 최대 길이에서 자르기
                split_point = sentence_max_len
            chunk = sentence[:split_point].strip()
            chunks.append(chunk)
            sentence = sentence[split_point:].strip()

        # 남은 문장을 마지막 덩어리에 추가
        chunks.append(sentence)

        # 마지막 덩어리가 3단어 미만이면 이전 덩어리와 합치기
        orig_len_chunks = len(chunks)
        if len(chunks) > 1 and len(chunks[-1].split()) < 3:
            chunks[orig_len_chunks - 2] += ' ' + chunks.pop(-1)

        # 분리된 덩어리들을 최종 결과에 추가
        for i, chunk in enumerate(chunks):
            result.append(chunk)
            sentence_end_flag.append(chunk[-1] in ".?!" if i == len(chunks) - 1 else False)

    return result, sentence_end_flag



# WAV 파일로 저장하는 함수
def save_audio(wav_data, sample_rate=Synthesizer.hparams.sample_rate, output_path="output.wav"):
    wav_data = np.int16(wav_data / np.max(np.abs(wav_data)) * 32767)
    wavfile.write(output_path, sample_rate, wav_data)


def main(text, hash, silence_duration_between_sentence=0.5):
    texts, sentence_end_flag = split_text(text)

    splitted = hash.split('_')
    assert len(splitted)==2
    embed_path = Path(__file__).parent / 'embeddings' / f'{splitted[0]}.pkl'
    with open(embed_path, 'rb') as file:
        embed = pickle.load(file)
    embeds = [embed] * len(texts)

    syn_model_path = Path(__file__).parent / 'models' / 'synthesizer.pt'
    synthesizer = Synthesizer(syn_model_path, True)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)

    vocoder_model_path = Path(__file__).parent / 'models' / 'vocoder.pt'
    vocoder = voc.load_model(vocoder_model_path)

    silent_wav = np.zeros(int(Synthesizer.sample_rate*silence_duration_between_sentence))
    result_wav = []
    no_add_silence_wav = []
    for i, spec in enumerate(specs):
        print(texts[i], sentence_end_flag[i])
        wav = voc.infer_waveform(spec)
        breaks = [spec.shape[1]]
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        wav = preprocess_wav(wav)

        wav = wav / np.abs(wav).max() * 0.97
        result_wav.append(wav)
        if sentence_end_flag[i]:
            result_wav.append(silent_wav)
        no_add_silence_wav.append(wav)
    result_wav = np.concatenate(result_wav)
    no_add_silence_wav = np.concatenate(no_add_silence_wav)
    
    output_path = Path(__file__).parent / 'synthesized_samples' / f'{splitted[0]}_{splitted[1]}.wav'
    save_audio(result_wav, Synthesizer.hparams.sample_rate, output_path)
    output_path = Path(__file__).parent / 'synthesized_samples' / f'{splitted[0]}_{splitted[1]}_no_add_silence.wav'
    save_audio(no_add_silence_wav, Synthesizer.hparams.sample_rate, output_path)

    return result_wav


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help=\
        'Text to synthesize.')
    parser.add_argument('--hash_and_time', type=str, help=\
        "User-specific hash string. ex) 3A184B_241107-140403")
    args = parser.parse_args()

    main(*vars(args).values())

