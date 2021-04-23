import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os
import matplotlib.pyplot as plt

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

test_directory = '../data/recordings/cleaned_set/cleaned_test_set'

audio_paths, labels, label_class_dict = get_paths_and_labels(test_directory)

for file_ in audio_paths:
    audio = tfio.audio.AudioIOTensor(file_)
    audio_slice = audio[0:]
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    tensor = tf.cast(audio_tensor, tf.float32)
    spectrogram = tfio.experimental.audio.spectrogram(tensor, nfft=1024, window=1024, stride=256)
    plt.imsave(file_[:-4] + 'spectrogram.png', tf.math.log(spectrogram).numpy())

