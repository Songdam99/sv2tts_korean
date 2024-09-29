import os
import numpy as np
import librosa

def check_wav_files(folder_path):
    """
    Checks all WAV files in the given folder and its subfolders, printing invalid files.

    :param folder_path: Path to the folder containing WAV files
    """
    # 전체 WAV 파일 수를 계산
    all_wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".wav"):
                all_wav_files.append(os.path.join(root, filename))
    
    total_files = len(all_wav_files)
    invalid_files = []  # 유효하지 않은 파일을 저장할 리스트
    
    # WAV 파일 검증 시작
    for idx, file_path in enumerate(all_wav_files):
        filename = os.path.basename(file_path)  # 파일 이름만 가져오기
        print(f"Checking {idx + 1}/{total_files}: {filename}")
        try:
            # WAV 파일 로드
            wav, sr = librosa.load(file_path, sr=None)
            # 에너지 계산
            energy = np.sum(wav ** 2)
            
            if energy <= 0:  # 에너지가 0 이하인 경우
                invalid_files.append(filename)  # 유효하지 않은 파일 추가
        except Exception as e:
            invalid_files.append(filename)  # 예외가 발생한 파일 추가

    # 검사 결과 출력
    if invalid_files:
        print("\nInvalid files:")
        for invalid_file in invalid_files:
            print(invalid_file)
    else:
        print("\nAll files are valid.")

# 파일 경로 (여기서는 폴더 경로)
folder_path = r"C:\Users\s_jinwoo0302\Desktop\GraduateProject\AI 음성 데이터\014.다화자 음성합성 데이터\01.데이터\1.Training\원천데이터\TS21\TS21\1.남성\12000문장"

# 유효성 검사 수행
check_wav_files(folder_path)
