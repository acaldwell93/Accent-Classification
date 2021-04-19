import os
import librosa
import soundfile as sf

current_folder = "../data/recordings/wav_recordings"
new_folder = "../data/recordings/wav_16khz"

files = os.listdir(current_folder)

for file in files:
    current_path = os.path.join(current_folder, file)
    new_path = os.path.join(new_folder, file)
    sound_16k, sampling_rate = librosa.load(current_path, sr=16000)
    sf.write(new_path, sound_16k, sampling_rate)