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
        full_paths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]
        audio_paths += full_paths
        labels += [label] * len(full_paths)

    return audio_paths, labels, label_class_dict

def add_random_transformations(audio_path):
    wav, sr = librosa.load(audio_path, 16000)
    new_wav = wav + 0.002*np.random.normal(0,1,len(wav))
    new_wav = np.roll(new_wav, np.random.randint(0, len(new_wav)//2))
#     new_wav = librosa.effects.time_stretch(new_wav,np.random.randint(97,104)*.01)
    new_wav = librosa.effects.pitch_shift(new_wav, sr, n_steps=np.random.randint(-20, 21)*0.1)
    final_wav = new_wav.copy()
    final_wav.resize((432000), refcheck=False)
    return final_wav, sr


audio_paths, labels, label_class_dict = get_paths_and_labels("../data/recordings/train_set")
for file_ in audio_paths:
    if file_.endswith('DS_Store'):
        continue
    for i in range(3):
        new_wav, sr = add_random_transformations(file_)
        new_filename = file_[:-4] + f"_{i}.wav"
        sf.write(new_filename, new_wav, sr)