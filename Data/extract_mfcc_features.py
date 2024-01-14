import librosa
import numpy as np
import os
import csv

def extract_mfcc(filepath, n_mfcc=13, hop_length=512, n_fft=2048):
    audio, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    features = np.mean(mfcc.T, axis=0)  # Transpose and take the mean along the axis
    return features

train_folder = '../../augmented_train/'
wav_files = [f for f in os.listdir(train_folder) if f.endswith('.wav')]

header = ['ID', 'FilePath'] + [f'MFCC_{i+1}' for i in range(13)] + ['ClassID']
output_csv_path = 'mfcc_features_augmented_train.csv'
train_csv_path = 'train.csv'

labels_dict = {}
with open(train_csv_path, 'r') as train_csv:
    csv_reader = csv.reader(train_csv)
    next(csv_reader)
    for row in csv_reader:
        filepath, label = row[1], row[2]
        filepath = filepath.replace('\\', '/')
        labels_dict[filepath] = label

print(labels_dict)

with open(output_csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    for index, wav_file in enumerate(wav_files):
        wav_file_path = os.path.join(train_folder, wav_file)
        mfcc_features = extract_mfcc(wav_file_path)

        matching_label = ''
        for key in labels_dict.keys():
            if key in wav_file_path:
                matching_label = labels_dict[key]
                break

        row_data = [index, wav_file_path] + mfcc_features.tolist() + [matching_label]
        csv_writer.writerow(row_data)

print(f"MFCC features with Labels saved to {output_csv_path}")
