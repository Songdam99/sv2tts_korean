import os
import shutil

def find_empty_npy_folders(root_folder):
    empty_folders = []
    
    # Walk through the immediate subdirectories of the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Only consider the immediate subdirectories
        if dirpath == root_folder:
            for dirname in dirnames:
                subfolder_path = os.path.join(dirpath, dirname)
                # Check if .npy files are present
                if not any(filename.endswith('.npy') for filename in os.listdir(subfolder_path)):
                    empty_folders.append(subfolder_path)
    
    return empty_folders

def delete_folders(folders):
    for folder in folders:
        # Delete the empty folder
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")

# 경로를 여기에 입력하세요.
root_folder_path = r"D:\encoder"  # 예: D:\encoder
empty_npy_folders = find_empty_npy_folders(root_folder_path)

# 삭제할 폴더 목록 출력
if empty_npy_folders:
    print("Empty folders found:")
    for folder in empty_npy_folders:
        print(folder)

    # 삭제 수행
    delete_folders(empty_npy_folders)
else:
    print("No empty folders found.")
