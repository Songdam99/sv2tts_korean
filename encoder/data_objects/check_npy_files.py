import os
import matplotlib.pyplot as plt
import shutil

def check_npy_file_counts_and_plot_histogram(root_dir: str, target_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Print initial number of speaker folders
    initial_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Initial number of speaker folders: {initial_speaker_count}")

    speaker_counts = {}
    total_speakers = initial_speaker_count
    checked_speakers = 0
    speakers_with_150_or_more = 0  # Counter for speakers with 150 or more .npy files

    for speaker_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, speaker_folder)
        if os.path.isdir(folder_path):
            checked_speakers += 1
            print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers})")
            npy_file_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.npy'))
            speaker_counts[npy_file_count] = speaker_counts.get(npy_file_count, 0) + 1
            
            if npy_file_count < 150:  # Check if .npy count is less than 150
                # Move the folder to the target directory
                shutil.move(folder_path, os.path.join(target_dir, speaker_folder))
                print(f"Moved {speaker_folder} with {npy_file_count} .npy files to {target_dir}.")
            else:
                speakers_with_150_or_more += 1

            if npy_file_count == 0:
                print(f"No .npy files found in {speaker_folder}.")
            else:
                print(f"Number of .npy files in {speaker_folder}: {npy_file_count}")

    # Output the number of speakers with 150 or more .npy files
    print(f"Number of speakers with 150 or more .npy files: {speakers_with_150_or_more}")

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
check_npy_file_counts_and_plot_histogram("D:\\encoder", "D:\\npy_under150")
