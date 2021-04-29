import numpy as np
import os
import math
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
import librosa
import sklearn.preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv1D, MaxPooling1D


def get_paths_and_labels(audio_directory_path):
#     class_names = os.listdir(audio_directory_path)
#     class_names.remove('.DS_Store')
    class_names = ['english', 'spanish', 'mandarin']
    label_class_dict = {num:class_ for num, class_ in enumerate(class_names)}
    
    audio_paths = []
    labels = []

    for label, name in enumerate(class_names):
        dir_path = os.path.join(audio_directory_path, name)
        full_paths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if 'ipynb' not in filename and '.DS_Store' not in filename]
        audio_paths += full_paths
        labels += [label] * len(full_paths)
    
    return audio_paths, labels, label_class_dict

def path_to_mfcc(path):
    audio, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(audio, sr, n_fft=1024,
                                    n_mfcc=40, n_mels=40,
                                    hop_length=256)
#     mfcc = librosa.feature.delta(mfcc, order=1)
    mfccs = mfcc.copy()
    mfccs.T
    mfccs.resize((6000, 40), refcheck=False)
#     mfccs = sklearn.preprocessing.scale(mfccs)
    return mfccs

def create_dataset(audio_paths, labels):
    X = np.zeros((len(audio_paths), 6000, 40))
    for idx, file in enumerate(audio_paths):
        X[idx] = path_to_mfcc(file)
    y = np.array(labels)
    return tf.data.Dataset.from_tensor_slices((X, y))

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)



train_path = '../../data/cleaned_set_44/cleaned_set_train'
test_path = '../../data/cleaned_set_44/cleaned_set_test'

batch_size = 16

train_audio_paths, train_labels, train_label_class_dict = get_paths_and_labels(train_path)
train_ds = create_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=len(train_ds)).batch(batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_audio_paths, test_labels, test_label_class_dict = get_paths_and_labels(test_path)
test_ds = create_dataset(test_audio_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=len(test_ds)).batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


model=tf.keras.models.Sequential()

model.add(Conv1D(16, 3, strides=2, padding='same', input_shape=(6000, 40), activation='relu'))
model.add(Conv1D(16, 3, strides=2, padding='same', activation='relu'))
model.add(Dropout(.2))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(Dropout(.2))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Dropout(.2))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=2))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=.0001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
