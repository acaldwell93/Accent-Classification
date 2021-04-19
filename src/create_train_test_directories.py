import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

df = pd.read_csv('../data/speakers_all.csv')
df.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], inplace=True)
df_full = df[df['file_missing?'] == False]
df_full['sex'] = df_full['sex'].apply(lambda x: 'female' if x == 'famale' else x)


top10languages = df_full['native_language'].value_counts()[:10].index.values


train_set_filenames = []
test_set_filenames = []

for language in top10languages:
    X = df_full[df_full['native_language'] == language]['filename'].values
    y = [language] * len(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_set_filenames.append(X_train)
    test_set_filenames.append(X_test)
    

train_names = []
for lst in train_set_filenames:
    for file in lst:
        train_names.append(file)

test_names = []
for lst in test_set_filenames:
    for file in lst:
        test_names.append(file)


df_full['train_test_none'] = df_full['filename'].apply(lambda x: "train" if x in train_names else "test" if x in test_names else "none")


df_full[df_full['train_test_none'] != 'none'].to_csv("../data/train_test.csv")


current_dir = "../data/recordings/wav_16khz"

for idx, row in df_full.iterrows():
    if row['train_test_none'] == 'none':
        continue
    elif row['train_test_none'] == 'train':
        new_path = os.path.join("../data/recordings/train_set", row['native_language'], row['filename'] + ".wav")
    elif row['train_test_none'] == 'test':
        new_path = os.path.join("../data/recordings/test_set", row['native_language'], row['filename'] + ".wav")
    current_path = os.path.join(current_dir, row['filename'] + ".wav")
    shutil.copyfile(current_path, new_path)