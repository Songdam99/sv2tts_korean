import zipfile
import glob
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# 압축 파일들이 있는 디렉토리 경로
root = Path(r'C:/Users/otulp/Desktop/GraduateProject/AI 음성 데이터/131.다국어 통·번역 낭독체 데이터/01-1.정식개방데이터/Training/02.라벨링데이터')

# 압축 파일들을 순차적으로 압축 해제
for zip_path in root.glob('*.zip'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 압축 해제할 경로를 지정 (현재 zip 파일과 동일한 디렉토리)
        extract_path = zip_path.parent / zip_path.stem  # .zip 확장자 제외한 폴더명으로 압축 해제
        zip_ref.extractall(extract_path)
        print(f"Extracted {zip_path} to {extract_path}")