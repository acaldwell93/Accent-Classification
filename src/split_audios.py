import os
from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf

def get_paths_and_labels(audio_directory_path):
    class_names = os.listdir(audio_directory_path)
    if '.DS_Store' in class_names:
        class_names.remove('.DS_Store')
    label_class_dict = {num:class_ for num, class_ in enumerate(class_names)}

    audio_paths = []
    labels = []

    for label, name in enumerate(class_names):
        dir_path = os.path.join(audio_directory_path, name)
        full_paths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename != ".DS_Store"]
        audio_paths += full_paths
        labels += [label] * len(full_paths)

    return audio_paths, labels, label_class_dict

training_filepath = '../data/recordings/cleaned_set_3/cleaned_train_set'
testing_filepath = '../data/recordings/cleaned_set_3/cleaned_test_set'

audio_paths, labels, label_class_dict = get_paths_and_labels(training_filepath)
for file_ in audio_paths:
    if file_.endswith('DS_Store'):
        continue
    # audio = AudioSegment.from_wav(file_)
    # unit = len(audio)/3
    
    # clip1 = audio[:unit]
    # clip2 = audio[unit:2*unit]
    # clip3 = audio[2*unit:]
    
    # clip1.export(file_[:-4] + 'clip1.wav', format='wav')
    # clip2.export(file_[:-4] + 'clip2.wav', format='wav')
    # clip3.export(file_[:-4] + 'clip3.wav', format='wav')

    wav, sr = librosa.load(file_, 16000)
    for i in range(2):
        new_wav = np.roll(wav, np.random.randint(0, len(wav)))
        new_filename = file_[:-4] + f'clip{i}.wav'
        sf.write(new_filename, new_wav, sr)