from pathlib import Path

def create_sources_file(speaker_folder: Path):
    sources_file_path = speaker_folder.joinpath("_sources.txt")
    npy_files = list(speaker_folder.glob("*.npy"))

    # Create the sources file
    with sources_file_path.open("w", encoding='utf-8') as sources_file:
        for npy_file in npy_files:
            wav_file_path = str(npy_file).replace('.npy', '.wav')
            sources_file.write(f"{npy_file.name},{wav_file_path}\n")
            
    total_files = len(npy_files)
    print(f"Created _sources.txt in '{speaker_folder.name}' with {total_files} entries.")

def main(root_directory: str):
    root_path = Path(root_directory)
    speaker_folders = [folder for folder in root_path.iterdir() if folder.is_dir()]
    
    total_speakers = len(speaker_folders)
    print(f"Total speakers: {total_speakers}")
    
    for idx, speaker_folder in enumerate(speaker_folders, start=1):
        create_sources_file(speaker_folder)
        print(f"Progress: {idx}/{total_speakers} ({(idx / total_speakers) * 100:.2f}%)")

if __name__ == "__main__":
    main("C:\\Users\\s_jinwoo0302\\Desktop\\validation2_npy150")  # Update with your specific path
