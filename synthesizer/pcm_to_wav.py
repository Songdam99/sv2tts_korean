import struct
from pathlib import Path

def make_wav_format(pcm_data:bytes, ch:int) -> bytes:
        """ 
        pcm_data를 통해서 wav 헤더를 만들고 wav 형식으로 저장한다.
        :param pcm_data: pcm bytes
        :param ch: 채널 수
        :return wav: wave bytes
        """
        waves = []
        waves.append(struct.pack('<4s', b'RIFF'))
        waves.append(struct.pack('I', 1))  
        waves.append(struct.pack('4s', b'WAVE'))
        waves.append(struct.pack('4s', b'fmt '))
        waves.append(struct.pack('I', 16))
        # audio_format, channel_cnt, sample_rate, bytes_rate(sr*blockalign:초당 바이츠수), block_align, bps
        if ch == 2:
            waves.append(struct.pack('HHIIHH', 1, 2, 16000, 64000, 4, 16))  
        else:
            waves.append(struct.pack('HHIIHH', 1, 1, 16000, 32000, 2, 16))
        waves.append(struct.pack('<4s', b'data'))
        waves.append(struct.pack('I', len(pcm_data)))
        waves.append(pcm_data)
        waves[1] = struct.pack('I', sum(len(w) for w in waves[2:]))
        return b''.join(waves)

# change the data_path!!
data_path = Path("C:/Users/otulp/Desktop/GraduateProject/AI 음성 데이터/KSponSpeech/KsponSpeech_01")
speaker_dirs = list(data_path.glob("*"))
target_dirs = [speaker_dir.parent / f"{speaker_dir.name}_wav" for speaker_dir in speaker_dirs]

for target_dir in target_dirs:
    target_dir.mkdir(parents=True, exist_ok=True)
    
for i, speaker_dir in enumerate(speaker_dirs):
    source_pcms = speaker_dir.glob('*.pcm')
    for source_pcm in source_pcms:
        pcm_bytes = Path(source_pcm).read_bytes()
        wav_bytes = make_wav_format(pcm_bytes, 1)
        wav_path = target_dirs[i] / f"{source_pcm.stem}.wav"  # .pcm 확장자를 .wav로 변경
        with open(wav_path, 'wb') as file:
            file.write(wav_bytes)
        print(f"Converted {source_pcm} to {wav_path}")