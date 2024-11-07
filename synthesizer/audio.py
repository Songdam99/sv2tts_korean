import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav.astype(np.float32), sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break
    
    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold
    
    return start, end

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size

def linearspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
    S = _amp_to_db(np.abs(D), hparams) - hparams.ref_level_db
    
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    print("Step 1: Preemphasis 처리 시작")
    wav_preemphasized = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
    print("Step 1: Preemphasis 처리 완료 - wav_preemphasized의 shape:", wav_preemphasized.shape)

    print("Step 2: Short-Time Fourier Transform (STFT) 계산 시작")
    D = _stft(wav_preemphasized, hparams)
    print("Step 2: STFT 계산 완료 - D의 shape:", D.shape)

    print("Step 3: Amplitude를 dB로 변환 및 Mel 변환 시작")
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    print("Step 3: Mel spectrogram 계산 완료 - S의 shape:", S.shape)

    if hparams.signal_normalization:
        print("Step 4: 신호 정규화 진행")
        normalized_S = _normalize(S, hparams)
        print("Step 4: 정규화 완료 - normalized_S의 shape:", normalized_S.shape)
        return normalized_S
    
    return S


def inv_linear_spectrogram(linear_spectrogram, hparams):
    """Converts linear spectrogram to waveform using librosa"""
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram
    
    S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def inv_mel_spectrogram(mel_spectrogram, hparams):
    """Converts mel spectrogram to waveform using librosa"""
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram
    
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear
    
    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectrogram, hparams):
    global _mel_basis

    print("Step 1: _mel_basis 확인 시작")
    if _mel_basis is None:
        print("Step 1: _mel_basis가 None이므로 생성 중...")
        _mel_basis = _build_mel_basis(hparams)
        print("Step 1: _mel_basis 생성 완료 - _mel_basis의 shape:", _mel_basis.shape)
    else:
        print("Step 1: _mel_basis가 이미 존재합니다 - _mel_basis의 shape:", _mel_basis.shape)

    print("Step 2: Mel 변환 수행 중 - 입력 spectrogram의 shape:", spectrogram.shape)
    mel_spectrogram = np.dot(_mel_basis, spectrogram)
    print("Step 2: Mel 변환 완료 - mel_spectrogram의 shape:", mel_spectrogram.shape)

    return mel_spectrogram


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(sr=hparams.sample_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
    print("Step 1: 입력 x의 shape:", x.shape)
    
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    print("Step 2: min_level 계산 완료 - min_level:", min_level)

    db_output = 20 * np.log10(np.maximum(min_level, x))
    print("Step 3: dB 변환 완료 - db_output의 shape:", db_output.shape)

    return db_output

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
    
    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)
