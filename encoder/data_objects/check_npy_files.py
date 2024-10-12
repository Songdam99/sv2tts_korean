import os
import matplotlib.pyplot as plt

def check_npy_file_counts_and_plot_histogram(root_dir: str):
    if not os.path.isdir(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return

    # Print initial number of speaker folders
    initial_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Initial number of speaker folders: {initial_speaker_count}")

    speaker_counts = {f"{i}-{i+99}": 0 for i in range(0, 1000, 100)}  # Dictionary to count ranges 0-99, 100-199, ..., 900-999
    total_speakers = initial_speaker_count
    checked_speakers = 0

    for speaker_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, speaker_folder)
        if os.path.isdir(folder_path):
            checked_speakers += 1
            print(f"Checking in speaker folder: {speaker_folder} ({checked_speakers}/{total_speakers})")
            npy_file_count = sum(1 for file in os.listdir(folder_path) if file.endswith('.npy'))

            # Categorize npy file counts into 100-file bins
            for i in range(0, 1000, 100):
                if i <= npy_file_count < i + 100:
                    speaker_counts[f"{i}-{i+99}"] += 1
                    break
            else:
                # Count speakers with 1000 or more .npy files
                if npy_file_count >= 1000:
                    speaker_counts["1000+"] = speaker_counts.get("1000+", 0) + 1

            if npy_file_count == 0:
                print(f"No .npy files found in {speaker_folder}.")
            else:
                print(f"Number of .npy files in {speaker_folder}: {npy_file_count}")

    # Output the number of speakers for each range
    for range_key, count in speaker_counts.items():
        print(f"Number of speakers with {range_key} .npy files: {count}")

    # Print final number of speaker folders
    final_speaker_count = sum(1 for _ in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _)))
    print(f"Final number of speaker folders: {final_speaker_count}")

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
check_npy_file_counts_and_plot_histogram("D:\\encoder")
