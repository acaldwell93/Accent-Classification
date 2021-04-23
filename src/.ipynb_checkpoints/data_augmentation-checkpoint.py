import numpy as np
import librosa
import soundfile as sf
import os

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

def add_random_transformations(audio_path):
    wav, sr = librosa.load(audio_path, 16000)
    new_wav_high, new_wav_low = librosa.effects.pitch_shift(wav, sr, n_steps=1), librosa.effects.pitch_shift(wav, sr, n_steps=-1)
    new_wav_high, new_wav_low = np.roll(new_wav_high, np.random.randint(0, len(new_wav_high)//2)), np.roll(new_wav_low, np.random.randint(0, len(new_wav_low)//2))
    return new_wav_high, new_wav_low, sr



training_filepath = '../../data/data/recordings/cleaned_set/cleaned_train_set'
# testing_filepath = '../../data/data/recordings/cleaned_set/cleaned_test_set'

audio_paths, labels, label_class_dict = get_paths_and_labels(training_filepath)
for file_ in audio_paths:
    if file_.endswith('DS_Store') or file_.endswith('checkpoints'):
        continue
    new_wav_high, new_wav_low, sr = add_random_transformations(file_)
    high_filename = file_[:-4] + "high.wav"
    low_filename = file_[:-4] + "low.wav"
    sf.write(high_filename, new_wav_high, sr)
    sf.write(low_filename, new_wav_low, sr)


