import librosa
import os
import numpy as np
import matplotlib.pyplot as plt


wav = '/userHome/userhome2/dahyun/MultiSpeech/p225_366_mic1.wav'
(file_dir, file_id) = os.path.split(wav)
print("file_dir:", file_dir)
print("file_id:", file_id)

# original
y, sr = librosa.load(wav, sr=16000)
time = np.linspace(0, len(y)/sr, len(y)) # time axis
fig, ax1 = plt.subplots() # plot
ax1.plot(time, y, color = 'b', label='speech waveform')
ax1.set_ylabel("Amplitude") # y 축
ax1.set_xlabel("Time [s]") # x 축
plt.title(file_id) # 제목
plt.savefig(file_id+'.png')
plt.show()
librosa.output.write_wav('original_file.mp3', y, sr) # original wav to save mp3 file

# [64, 2, 72, 100]

# # cut half and save
# half = len(y)/2
# y2 = y[:round(half)]
# time2 = np.linspace(0, len(y2)/sr, len(y2))
# fig2, ax2 = plt.subplots()
# ax2.plot(time2, y2, color = 'b', label='speech waveform')
# ax1.set_ylabel("Amplitude") # y 축
# ax1.set_xlabel("Time [s]") # x 축
# plt.title('cut '+file_id)
# plt.savefig('cut_half '+file_id+'.png')
# plt.show()
# librosa.output.write_wav('cut_file.mp3', y2, sr) # save half-cut file