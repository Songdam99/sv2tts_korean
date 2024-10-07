import os
import matplotlib.pyplot as plt

def check_npy_file_counts_and_plot_histogram(root_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return

    # Print initial number of speaker folders
    initial_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Initial number of speaker folders: {initial_speaker_count}")

    speaker_counts = {}
    total_speakers = initial_speaker_count
    checked_speakers = 0

    # Counters for different thresholds
    speakers_with_150_or_more = 0
    speakers_with_200_or_more = 0
    speakers_with_250_or_more = 0
    speakers_with_300_or_more = 0
    speakers_with_350_or_more = 0
    speakers_with_400_or_more = 0
    speakers_with_450_or_more = 0

    for speaker_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, speaker_folder)
        if os.path.isdir(folder_path):
            checked_speakers += 1
            print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers})")
            npy_file_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.npy'))
            speaker_counts[npy_file_count] = speaker_counts.get(npy_file_count, 0) + 1
            
            # Count speakers for each threshold
            if npy_file_count >= 150:
                speakers_with_150_or_more += 1
            if npy_file_count >= 200:
                speakers_with_200_or_more += 1
            if npy_file_count >= 250:
                speakers_with_250_or_more += 1
            if npy_file_count >= 300:
                speakers_with_300_or_more += 1
            if npy_file_count >= 350:
                speakers_with_350_or_more += 1
            if npy_file_count >= 400:
                speakers_with_400_or_more += 1
            if npy_file_count >= 450:
                speakers_with_450_or_more += 1

            if npy_file_count == 0:
                print(f"No .npy files found in {speaker_folder}.")
            else:
                print(f"Number of .npy files in {speaker_folder}: {npy_file_count}")

    # Output the number of speakers for each threshold
    print(f"Number of speakers with 150 or more .npy files: {speakers_with_150_or_more}")
    print(f"Number of speakers with 200 or more .npy files: {speakers_with_200_or_more}")
    print(f"Number of speakers with 250 or more .npy files: {speakers_with_250_or_more}")
    print(f"Number of speakers with 300 or more .npy files: {speakers_with_300_or_more}")
    print(f"Number of speakers with 350 or more .npy files: {speakers_with_350_or_more}")
    print(f"Number of speakers with 400 or more .npy files: {speakers_with_400_or_more}")
    print(f"Number of speakers with 450 or more .npy files: {speakers_with_450_or_more}")

    # Print final number of speaker folders
    final_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Final number of speaker folders: {final_speaker_count}")

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(speaker_counts.keys(), speaker_counts.values(), color='skyblue')
    plt.xlabel('Number of .npy Files')
    plt.ylabel('Number of Speakers')
    plt.title('Number of Speakers for Each Count of .npy Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Usage
check_npy_file_counts_and_plot_histogram("D:\\encoder")
