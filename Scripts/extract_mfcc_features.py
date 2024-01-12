import librosa
import numpy as np
import os
import csv

def extract_mfcc(filepath, n_mfcc=13, hop_length=512, n_fft=2048):
    audio, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    features = np.mean(mfcc.T, axis=0)  # Transpose and take the mean along the axis
    return features

# Set the path to your 'train' folder
#train_folder = 'train/'
train_folder = 'test/'

# Get a list of all WAV files in the 'train' folder
wav_files = [f for f in os.listdir(train_folder) if f.endswith('.wav')]

def custom_sort_key(file_name):
    # Extract everything but the last four characters (including the dot and extension)
    key_part = file_name[:-4]
    return int(key_part)

# Sort the wav_files list using the custom sort key
sorted_wav_files = sorted(wav_files, key=custom_sort_key)

# print(sorted_wav_files)

# CSV header
header = ['FilePath'] + [f'MFCC_{i+1}' for i in range(13)]  # Assuming n_mfcc is always 13

# CSV output file
#output_csv_path = 'mfcc_features_sorted.csv'
output_csv_path = 'mfcc_features_sorted_test.csv'

# Writing the header to the CSV file
with open(output_csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    # Iterate through each WAV file, extract MFCC, and write to the CSV file
    for wav_file in sorted_wav_files:
        wav_file_path = os.path.join(train_folder, wav_file)
        mfcc_features = extract_mfcc(wav_file_path)
        row_data = [wav_file_path] + mfcc_features.tolist()
        csv_writer.writerow(row_data)

print(f"Sorted MFCC features saved to {output_csv_path}")
