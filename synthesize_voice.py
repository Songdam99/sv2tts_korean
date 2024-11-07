import os
import argparse
import pickle
from synthesizer.inference import Synthesizer
from vocoder import inference as voc
from pathlib import Path
import numpy as np
from encoder.audio import preprocess_wav
import scipy.io.wavfile as wavfile

sentence_max_len = 50

def split_text(text):
    words = text.split()
    result = []
    current_line = ""

    for i, word in enumerate(words):
        if len(current_line) + len(word) + 1 > sentence_max_len and i < len(words) - 3: # 다음 잘라진 문장에 최소 3단어는 있도록 보장
            result.append(current_line.strip())
            current_line = word
        else:
            current_line += " " + word

    if current_line:
        result.append(current_line.strip())

    return result


# WAV 파일로 저장하는 함수
def save_audio(wav_data, sample_rate=Synthesizer.hparams.sample_rate, output_path="output.wav"):
    wav_data = np.int16(wav_data / np.max(np.abs(wav_data)) * 32767)
    wavfile.write(output_path, sample_rate, wav_data)


def main(text, hash):
    texts = split_text(text)

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

    result_wav = []
    for spec in specs:
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
    result_wav = np.concatenate(result_wav)

    
    output_path = Path(__file__).parent / 'synthesized_samples' / f'{splitted[0]}_{splitted[1]}.wav'
    save_audio(result_wav, Synthesizer.hparams.sample_rate, output_path)

    return result_wav


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help=\
        'Text to synthesize.')
    parser.add_argument('--hash_and_time', type=str, help=\
        "User-specific hash string. ex) 3A184B_241107-140403")
    args = parser.parse_args()

    main(*vars(args).values())

