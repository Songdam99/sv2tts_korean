import os

# 입력 및 출력 폴더 경로 설정
input_folder = r'C:\Users\admin\Desktop\dirty\output'
output_file = r'C:\Users\admin\Desktop\dirty\output\alignment.txt'

# 출력 파일 초기화
with open(output_file, 'w', encoding='utf-8') as out_file:
    # 입력 폴더의 모든 파일 반복
    for filename in os.listdir(input_folder):
        if filename.endswith('.TextGrid'):
            # 파일 경로 설정
            file_path = os.path.join(input_folder, filename)

            # TextGrid 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # 파일 이름 기록 (확장자 제거)
                filename_without_extension = os.path.splitext(filename)[0]
                out_file.write(f"{filename_without_extension} ")

                # 단어 및 타임스탬프를 저장할 리스트 초기화
                words = []
                timestamps = []

                # 'words' 타이어를 찾아서 단어와 타임스탬프 읽기
                inside_words_tier = False
                for i in range(len(lines)):
                    line = lines[i].strip()
                    
                    if line.startswith('name = "words"'):
                        inside_words_tier = True
                    elif line.startswith('name = "phones"'):
                        inside_words_tier = False

                    if inside_words_tier:
                        if line.startswith('intervals: size ='):
                            intervals_count = int(line.split('=')[1].strip())
                        elif line.startswith('intervals ['):
                            # interval 정보 가져오기
                            xmin_line = lines[i + 1].strip()
                            xmax_line = lines[i + 2].strip()
                            text_line = lines[i + 3].strip()

                            xmin = float(xmin_line.split('=')[1].strip())
                            xmax = float(xmax_line.split('=')[1].strip())
                            text = text_line.split('=')[1].strip().strip('"')

                            words.append(text)
                            timestamps.append(f"{xmin},{xmax}")

                # 단어와 타임스탬프를 포맷에 맞춰 파일에 쓰기
                words_line = ",".join(words)
                timestamps_line = ",".join(timestamps[1:])  # 첫 번째 타임스탬프를 건너뛰기

                out_file.write(f'"{words_line}" ')
                out_file.write(f'"{timestamps_line}"\n')
