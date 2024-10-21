import glob
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing.pool import Pool

datasets_root = Path("C:/Users/admin/Desktop/Narrify_data/한국어 음성")
valid_transcript_path = "C:/Users/admin/Desktop/Narrify_data/한국어 음성/전시문_통합_스크립트/KsponSpeech_scripts/eval_clean.trn"
skip_existing = False

def make_alignment(speaker_dir, skip_existing):
    alignment_path = speaker_dir / "alignment.txt"
    alignment_path.touch(exist_ok=True)
    
    txt_paths = list(speaker_dir.glob("*.txt"))

    existing_basenames = set()
    if skip_existing:
        with alignment_path.open(mode='r', encoding='utf-8') as alignment_file:
            for line in alignment_file:
                existing_basename = line.split(" ")[0]  # 줄의 첫 번째 부분은 basename
                existing_basenames.add(existing_basename)

    with alignment_path.open(mode='w', encoding='utf-8') as alignment_file:
        for txt_path in txt_paths:
            basename = txt_path.stem
            if basename == "alignment":
                continue

            if skip_existing and basename in existing_basenames:
                continue

            # txt 파일 열어서 내용 읽기
            with txt_path.open(mode='r', encoding='cp949') as file:
                content = file.read().strip()  # 내용의 앞뒤 공백 제거

            # # 공백을 쉼표로 변경
            # content = content.replace(" ", ",")

            # txt 파일의 basename과 내용 적기
            alignment_file.write(f'{basename} {content}\n')

def make_alignment_val(speaker_dir, skip_existing, content):
    alignment_path = speaker_dir / "alignment.txt"
    alignment_path.touch(exist_ok=True)
    
    existing_basenames = set()
    if skip_existing:
        with alignment_path.open(mode='r', encoding='utf-8') as alignment_file:
            for line in alignment_file:
                existing_basename = line.split(" ")[0]  # 줄의 첫 번째 부분은 basename
                existing_basenames.add(existing_basename)

    with alignment_path.open(mode='w', encoding='utf-8') as alignment_file:
        for line in content:
            splitted = line.split(" :: ")
            basename = splitted[0].split('/')[-1][:-4]
            text = splitted[1].strip().replace(" ", ",")
            alignment_file.write(f'{basename} {text}\n')
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "valid"], help=\
        "Mode of data to create alignment.txt")
    args = parser.parse_args()

    if args.mode=='train':
        dataset_root = datasets_root.joinpath("KSponSpeech")
        input_dirs = [dataset_root.joinpath("KsponSpeech_01"),
        dataset_root.joinpath("KsponSpeech_02"),
        dataset_root.joinpath("KsponSpeech_03"),
        dataset_root.joinpath("KsponSpeech_04"),
        dataset_root.joinpath("KsponSpeech_05")]
    if args.mode=='valid':
        datasets_root = datasets_root.joinpath("평가용_데이터/KsponSpeech_eval")

    if args.mode == 'train':
        speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
        for i, speaker_dir in enumerate(tqdm(speaker_dirs, desc="Generating alignment files", total=len(input_dirs), unit="th speakers")):
            make_alignment(speaker_dir, skip_existing)
    elif args.mode == 'valid':
        speaker_dirs = [datasets_root.joinpath("eval_clean")]
        with open(valid_transcript_path, 'r', encoding='utf-8') as f:
            for i, speaker_dir in enumerate(tqdm(speaker_dirs, desc="Generating alignment files", total=len(speaker_dirs), unit="th speakers")):
                content = f.readlines()
                make_alignment_val(speaker_dir, skip_existing, content)