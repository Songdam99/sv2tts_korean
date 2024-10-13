import os
import shutil

def copy_npy_files_with_minimum_multiple_dirs(source_dirs: list, target_dir: str, min_files: int = 30, max_files: int = 150):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    total_speakers = 0  # 총 화자 수
    eligible_speakers = 0  # npy 파일이 30개 이상인 화자 수

    # 전체 화자 수 계산
    for root_dir in source_dirs:
        if not os.path.isdir(root_dir):
            print(f"Directory does not exist: {root_dir}")
            continue
        total_speakers += sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))

    checked_speakers = 0  # 체크한 화자 수

    for root_dir in source_dirs:
        if not os.path.isdir(root_dir):
            print(f"Directory does not exist: {root_dir}")
            continue

        for speaker_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, speaker_folder)
            if os.path.isdir(folder_path):
                npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

                checked_speakers += 1

                # Check if the speaker has at least `min_files` .npy files
                if len(npy_files) >= min_files:
                    eligible_speakers += 1
                    print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers} checked)")

                    # Create a new folder for the speaker in the target directory
                    speaker_target_folder = os.path.join(target_dir, speaker_folder)
                    os.makedirs(speaker_target_folder, exist_ok=True)

                    # Select up to `max_files` npy files
                    selected_files = npy_files[:max_files]  # 최대 150개 파일 선택
                    total_copied_files = 0  # Counter for copied files for the current speaker
                    for npy_file in selected_files:
                        file_path = os.path.join(folder_path, npy_file)
                        # Copy the file to the speaker's target directory
                        shutil.copy(file_path, os.path.join(speaker_target_folder, npy_file))
                        total_copied_files += 1

                    print(f"Total .npy files copied for {speaker_folder}: {total_copied_files}")

                # 진행 상황을 전체 화자 대비 백분율로 출력
                progress_percentage = (checked_speakers / total_speakers) * 100
                print(f"Progress: {progress_percentage:.2f}% ({checked_speakers}/{total_speakers})")

    print("File copying completed.")

# Usage
source_dirs = [
    r"D:\encoder",
    r"D:\npy_under150"
]
target_dir = r"C:\Users\admin\Desktop\Korean_dataset"

copy_npy_files_with_minimum_multiple_dirs(source_dirs, target_dir)
