import os

# VCTK 데이터셋 경로
vctk_dataset_path = "/userHome/userhome2/dahyun/MultiSpeech/VCTK_psv/test/VCTK_test"

# PSV 파일 생성
psv_file = open("vctk_data.psv", "w")
psv_file.write("speaker_id|audio_path|text|duration\n")

# 각 화자(스피커) 폴더 반복
speaker_id = 0  # 초기화
for speaker_folder in os.listdir(vctk_dataset_path):
    speaker_path = os.path.join(vctk_dataset_path, speaker_folder)
    
    # 각 화자 폴더 내의 음성 및 텍스트 파일 처리
    for i in range(1, 6):  # 5개의 문장을 가정
        audio_file = f"{speaker_folder}_{i:03d}_mic1.wav"  # 예: p1.wav, p2.wav, ...
        audio_path = os.path.join(speaker_path, audio_file)
    
        text_file = f"{speaker_folder}_{i:03d}.txt"   # 예: p1.txt, p2.txt, ...
        text_path = os.path.join(speaker_path, text_file)
        
        # 각 텍스트 파일에서 텍스트 데이터를 읽어옴
        with open(text_path, "r") as text_file:
            transcription = text_file.read().strip()
        
        # 음성 파일의 duration을 계산 (예: 파일 크기를 사용)
        audio_duration = os.path.getsize(audio_path) / 16000  # 예: 파일 크기를 샘플링 속도로 나눔
        
        # PSV 파일에 데이터 추가
        psv_file.write(f"{speaker_id}|{audio_path}|{transcription}|{audio_duration:.1f}\n")
    
    # 다음 화자를 위해 speaker_id 증가
    speaker_id += 1

# PSV 파일 닫기
psv_file.close()