import os
import shutil

def move_npy_files(train_folder, validation_folder, npy_count_to_move=10):
    # Ensure validation folder exists
    os.makedirs(validation_folder, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(train_folder):
        # Iterate over each speaker subfolder
        for dirname in dirnames:
            train_subfolder_path = os.path.join(dirpath, dirname)
            validation_subfolder_path = os.path.join(validation_folder, dirname)
            
            # Create the corresponding validation subfolder if it doesn't exist
            os.makedirs(validation_subfolder_path, exist_ok=True)

            # List all .npy files in the current subfolder
            npy_files = [f for f in os.listdir(train_subfolder_path) if f.endswith('.npy')]

            # If there are enough .npy files, move the specified number
            if len(npy_files) >= npy_count_to_move:
                files_to_move = npy_files[:npy_count_to_move]  # Get the first 10 .npy files

                # Move each selected .npy file
                for file in files_to_move:
                    src_file = os.path.join(train_subfolder_path, file)
                    dst_file = os.path.join(validation_subfolder_path, file)
                    shutil.move(src_file, dst_file)
                    print(f"Moved {src_file} to {dst_file}")
            else:
                print(f"Not enough .npy files in {train_subfolder_path}. Skipping this folder.")

# 경로를 여기에 입력하세요.
train_folder_path = r"C:\Users\admin\Desktop\Korean_dataset_train"
validation_folder_path = r"C:\Users\admin\Desktop\Korean_dataset_validation"

# npy 파일 10개씩 옮기기
move_npy_files(train_folder_path, validation_folder_path, npy_count_to_move=10)
