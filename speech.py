from pydub import AudioSegment

def convert_flac_to_wav(input_flac_file, output_wav_file):
    audio = AudioSegment.from_file(input_flac_file, format="flac")
    audio.export(output_wav_file, format="wav")

if __name__ == "__main__":
    input_flac_file = "/userHome/userhome2/dahyun/MultiSpeech/VCTK_psv/p225/p225_346_mic1.flac"  # 입력 FLAC 파일 경로
    output_wav_file = "/userHome/userhome2/dahyun/MultiSpeech/VCTK_psv/p225/p225_346_mic1.wav"  # 출력 WAV 파일 경로
    convert_flac_to_wav(input_flac_file, output_wav_file)

