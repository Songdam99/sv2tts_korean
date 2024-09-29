import wave

# # PCM 파일 읽기
# pcm_file = "C:/Users/otulp/Desktop/GraduateProject/AI 음성 데이터/한국어 음성/한국어_음성_분야/KsponSpeech_01/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_001000.pcm"
# wav_file = "C:/Users/otulp/Desktop/KsponSpeech_001000.wav"

# # PCM 파일 메타 정보 설정
# sample_rate = 16000  # 샘플링 레이트 (예: 44100 Hz)
# num_channels = 1     # 채널 수 (모노: 1, 스테레오: 2)
# sample_width = 2     # 샘플 폭 (예: 16비트 -> 2 바이트)

# # PCM 파일 읽기 및 WAV 파일 저장
# with open(pcm_file, 'rb') as pcm:
#     pcm_data = pcm.read()

# with wave.open(wav_file, 'wb') as wav:
#     wav.setnchannels(num_channels)
#     wav.setsampwidth(sample_width)
#     wav.setframerate(sample_rate)
#     wav.writeframes(pcm_data)
li = [16,7,15,2,19,20,17,22,14,13,11,4,5,6,18,12,3,23,10,8,9,35,24,27,26,28,25,29,31,38,40,39,30,36,37,32,21,33,34,41]
print(sorted(li))