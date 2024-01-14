import os
import librosa
import soundfile as sf

# Define the directory paths
data_dir = '../../'
output_dir = os.path.join(data_dir, 'augmented_train')
input_dir = output_dir

# Function to load a sound file
def load_sound_file(file_path):
    audio_data, sr = librosa.load(file_path, sr=None)
    return audio_data, sr

# Function to perform pitch shifting
def pitch_shift(audio_data, sr, pitch_shift_factor=2):
    return librosa.effects.pitch_shift(audio_data, n_steps=pitch_shift_factor, sr=sr)

# Function to perform time stretching
def time_stretch(audio_data, time_stretch_factor=1.5):
    return librosa.effects.time_stretch(audio_data, rate=time_stretch_factor)

# Function to perform amplitude scaling
def amplitude_scale(audio_data, scale_factor=0.5):
    return audio_data * scale_factor

# Function to save augmented audio data
def save_augmented_audio(audio_data, sr, file_name, suffix):
    output_path = os.path.join(output_dir, f"{file_name}_{suffix}.wav")
    sf.write(output_path, audio_data, sr)

# Iterate through sound files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        file_path = os.path.join(input_dir, filename)

        # Load the original sound file
        audio_data, sr = load_sound_file(file_path)

        # Apply data augmentation
        augmented_audio_pitch = pitch_shift(audio_data, sr, pitch_shift_factor=2)
        augmented_audio_time = time_stretch(audio_data, time_stretch_factor=1.5)
        augmented_audio_amplitude = amplitude_scale(audio_data, scale_factor=0.5)

        # Save the augmented audio files with suffixes
        save_augmented_audio(augmented_audio_pitch, sr, filename, 'pitch')
        save_augmented_audio(augmented_audio_time, sr, filename, 'time')
        save_augmented_audio(augmented_audio_amplitude, sr, filename, 'amplitude')
