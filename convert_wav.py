import glob
import librosa
import os
from pandas import DataFrame


def wav_to_mfcc(audio_path, n_mfcc=20, duration=4):
  # Load audio data
  y, sr = librosa.load(audio_path)

  # Get total audio length
  total_duration = len(y) / sr

  # Check if requested duration is within file limits
  if duration > total_duration:
    print(f"Warning: Requested duration ({duration}s) exceeds file length ({total_duration:.2f}s). Using full audio.")
    duration = total_duration

  # Extract desired segment (4 seconds)
  start_sample = 0
  end_sample = int(sr * duration)
  y_segment = y[start_sample:end_sample]

  # Extract MFCC features from the segment
  mfcc_feat = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)

  return mfcc_feat


def read_wavs_to_dataframe(parent_dir, subdir_name="sounds", n_mfcc=20, duration=4):
  # Construct subdirectory path
  subdirectory = os.path.join(parent_dir, subdir_name)

  # Search for all .wav files using glob
  wav_files = glob.glob(os.path.join(subdirectory, "*.wav"), recursive=False)

  # Create empty lists to store data
  filenames = []
  mfcc_features = []

  # Loop through each wav file
  for wav_file in wav_files:
    filename = os.path.basename(wav_file)  # Extract filename
    mfcc_feat = wav_to_mfcc(wav_file, n_mfcc, duration)
    filenames.append(filename)
    #mfcc_features.append(mfcc_feat.T)  # Transpose for row-wise storage in DataFrame
    mfcc_features.append(mfcc_feat)

  # Create DataFrame (assuming all MFCC features have the same shape)
  df = DataFrame({"filename": filenames, "mfcc_features": mfcc_features})

  return df


# Example usage
parent_dir = os.getcwd()
df = read_wavs_to_dataframe(parent_dir)

print(df)

df.to_pickle('mfcc_data.pkl')

print("DataFrame saved in mfcc_data.pkl successfully!")
