import os
import matplotlib.pyplot as plt

def check_npy_file_counts_and_plot_histogram(root_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return

    # Print initial number of speaker folders
    initial_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Initial number of speaker folders: {initial_speaker_count}")

    # 수정된 범주 설정: 0-99부터 2900-2999까지, 그리고 3000 이상의 범주 추가
    speaker_counts = {f"{i}-{i+99}": 0 for i in range(0, 3000, 100)}  # 0-99, 100-199, ..., 2900-2999
    speaker_counts["3000+"] = 0  # 3000 이상의 경우를 따로 저장

    total_speakers = initial_speaker_count
    checked_speakers = 0

    # 최소값과 최대값을 추적할 변수들 초기화
    min_npy_files = float('inf')  # 초기값을 무한대로 설정
    max_npy_files = float('-inf')  # 초기값을 음의 무한대로 설정

    for speaker_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, speaker_folder)
        if os.path.isdir(folder_path):
            checked_speakers += 1
            print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers})")
            npy_file_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.npy'))

            # 3000까지 100개 단위로 분류
            for i in range(0, 3000, 100):
                if i <= npy_file_count < i + 100:
                    speaker_counts[f"{i}-{i+99}"] += 1
                    break
            else:
                # 3000 이상의 .npy 파일이 있는 경우
                if npy_file_count >= 3000:
                    speaker_counts["3000+"] += 1

            if npy_file_count == 0:
                print(f"No .npy files found in {speaker_folder}.")
            else:
                print(f"Number of .npy files in {speaker_folder}: {npy_file_count}")

            # 최소값과 최대값 업데이트
            min_npy_files = min(min_npy_files, npy_file_count)
            max_npy_files = max(max_npy_files, npy_file_count)

    # Output the number of speakers for each range
    for range_key, count in speaker_counts.items():
        print(f"Number of speakers with {range_key} .npy files: {count}")

    # Print final number of speaker folders
    final_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Final number of speaker folders: {final_speaker_count}")

    # Print the minimum and maximum .npy file counts
    print(f"Minimum .npy files in a folder: {min_npy_files}")
    print(f"Maximum .npy files in a folder: {max_npy_files}")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(speaker_counts.keys(), speaker_counts.values(), color='skyblue')
    plt.xlabel('Range of .npy Files')
    plt.ylabel('Number of Speakers')
    plt.title('Distribution of Speakers by Number of .npy Files (in 100s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Usage
check_npy_file_counts_and_plot_histogram(r"C:\Users\admin\Desktop\Korean_dataset_npy30_150")
