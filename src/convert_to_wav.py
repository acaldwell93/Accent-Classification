import os
from pydub import AudioSegment

home = os.path.expanduser("~")
current_folder = home + "/Desktop/archive/recordings/recordings"
new_folder = home + "/Desktop/archive/recordings/wav_recordings"

files = os.listdir(current_folder)

for file in files:
    current_path = os.path.join(current_folder, file)
    sound = AudioSegment.from_mp3(current_path)
    new_path = os.path.join(new_folder, file[:-4] + ".wav")
    sound.export(new_path, format='wav')