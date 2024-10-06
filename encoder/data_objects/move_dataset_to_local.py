import os
import numpy as np
import shutil

def copy_npy_files(root_dir: str, target_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    total_speakers = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))  # 총 화자 수
    checked_speakers = 0  # 체크한 화자 수

    for speaker_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, speaker_folder)
        if os.path.isdir(folder_path):
            checked_speakers += 1
            print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers})")
            npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

            # Create a new folder for the speaker in the target directory
            speaker_target_folder = os.path.join(target_dir, speaker_folder)
            os.makedirs(speaker_target_folder, exist_ok=True)

            # Limit to maximum 150 files
            total_copied_files = 0  # Counter for copied files for the current speaker
            for npy_file in npy_files[:150]:
                file_path = os.path.join(folder_path, npy_file)
                # Copy the file to the speaker's target directory
                shutil.copy(file_path, os.path.join(speaker_target_folder, npy_file))
                total_copied_files += 1

            print(f"Total .npy files copied for {speaker_folder}: {total_copied_files}")

            # 현재 화자 복사 완료 후 진행 상황 출력 (npy_file 이름 생략)
            print(f"Finished copying files for speaker {checked_speakers}/{total_speakers}.")

    print("File copying completed.")

# Usage
copy_npy_files("D:\\encoder", "C:\\Users\\s_jinwoo0302\\Desktop\\npy_over150")
