from scipy.ndimage.morphology import binary_dilation
from encoder.params_data import *
from pathlib import Path
from typing import Optional, Union
from warnings import warn
import numpy as np
import librosa
import struct

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        if len(wav)==0:
            return np.array([])
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True, fpath_or_wav=fpath_or_wav)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav


# def wav_to_mel_spectrogram(wav):
#     """
#     Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
#     Note: this not a log-mel spectrogram.
#     """
#     frames = librosa.feature.melspectrogram(
#         wav,
#         sampling_rate,
#         n_fft=int(sampling_rate * mel_window_length / 1000),
#         hop_length=int(sampling_rate * mel_window_step / 1000),
#         n_mels=mel_n_channels
#     )
#     return frames.astype(np.float32).T


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav, # changed row
        sr=sampling_rate, # changed row
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False, fpath_or_wav=None):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    # 제로 분모 방지
    if np.mean(wav ** 2) > 0:
        dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    else:
         # 유효하지 않은 배열 경고 및 경로 출력
        print("Warning: Invalid audio data encountered. Array mean is zero.")
        print(f"Invalid audio data path: {fpath_or_wav}")  # fpath_or_wav를 받아오는 방법은 아래에서 설명
        return wav  # 원래 wav를 그대로 반환하거나 다른 처리 방안을 선택할 수 있음
        
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def mel_spectrogram_to_wav(mel_spectrogram, sr, n_fft, hop_length):
    """
    Converts a mel spectrogram back to a waveform using Griffin-Lim algorithm.
    
    :param mel_spectrogram: Mel spectrogram to be converted.
    :param sr: Sampling rate.
    :param n_fft: Length of the FFT window.
    :param hop_length: Number of samples between successive frames.
    :param n_mels: Number of Mel bands.
    :return: Reconstructed waveform.
    """
    # Inverse the mel filter bank
    mel_to_linear = librosa.feature.inverse.mel_to_stft(
        mel_spectrogram.T, sr=sr, n_fft=n_fft
    )

    # Use Griffin-Lim to approximate the original waveform from the linear spectrogram
    wav = librosa.griffinlim(
        mel_to_linear, n_iter=32, hop_length=hop_length, win_length=n_fft
    )
    
    return wav

