import os

directory = "../data/recordings/train_set/spanish_copy"

for rec in os.listdir(directory):
    if "_" in rec:
        os.remove(os.path.join(directory, rec))
