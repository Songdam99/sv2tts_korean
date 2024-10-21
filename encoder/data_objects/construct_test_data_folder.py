import os
import shutil

# 폴더 A와 폴더 B의 경로 설정
folder_A = 'C:/Users/admin/Downloads/for_test_encoder'
folder_B = 'C:/Users/admin/Desktop/Narrify_data/한국어 음성/multispeaker_test'

# 파일을 1000개씩 옮기는 함수
def move_npy_files(source_dir, target_dir, num_files=1000):
    # # 폴더에 있는 모든 npy 파일을 가져옴
    # npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    
    # # 이동할 파일 1000개씩 가져오기
    # files_to_move = npy_files[:num_files]
    
    # # 폴더 B 안에 동일한 이름의 디렉토리 만들기
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # # 파일 옮기기
    # for file in files_to_move:
    #     src_file = os.path.join(source_dir, file)
    #     dest_file = os.path.join(target_dir, file)
    #     shutil.move(src_file, dest_file)

    # _sources.txt 파일 복사하기
    source_txt_file = os.path.join(source_dir, '_sources.txt')
    target_txt_file = os.path.join(target_dir, '_sources.txt')
    if os.path.exists(source_txt_file):
        shutil.copy(source_txt_file, target_txt_file)

# # 폴더 A의 하위 디렉토리를 순회
# for sub_dir in os.listdir(folder_A):
#     sub_dir_A = os.path.join(folder_A, sub_dir)
#     sub_dir_B = os.path.join(folder_B, sub_dir)
    
#     if os.path.isdir(sub_dir_A):  # 하위 디렉토리인지 확인
#         move_npy_files(sub_dir_A, sub_dir_B, num_files=1000)


# _sources.txt 파일을 실제 npy 파일에 맞춰 수정하는 함수
def update_sources_txt(speaker_dir):
    # B 폴더 안에 있는 실제 npy 파일 목록
    existing_npy_files = set([f for f in os.listdir(speaker_dir) if f.endswith('.npy')])

    # _sources.txt 파일 경로
    sources_txt_file = os.path.join(speaker_dir, '_sources.txt')

    if os.path.exists(sources_txt_file):
        # _sources.txt 파일 수정
        updated_lines = []
        with open(sources_txt_file, 'r') as file:
            for line in file:
                npy_filename = line.split(',')[0]  # 각 줄에서 npy 파일명 추출
                if npy_filename in existing_npy_files:
                    updated_lines.append(line)  # 실제 npy 파일이 있으면 해당 줄을 기록

        # 수정된 내용을 다시 _sources.txt에 기록
        with open(sources_txt_file, 'w') as file:
            file.writelines(updated_lines)

# 폴더 B의 하위 디렉토리를 순회하며 _sources.txt 파일을 수정
for sub_dir in os.listdir(folder_B):
    sub_dir_path = os.path.join(folder_B, sub_dir)
    
    if os.path.isdir(sub_dir_path):  # 하위 디렉토리인지 확인
        update_sources_txt(sub_dir_path)