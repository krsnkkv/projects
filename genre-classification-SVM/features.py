# features.py

import os
import numpy as np
import librosa


def extract_features(file_path):
    """
    extract mfcc-based features from a single audio file.

    returns a 26-dimensional feature vector:
    - 13 mfcc means
    - 13 mfcc standard deviations
    """

    # load the audio file at its original sampling rate
    # sr=None ensures we don't resample unnecessarily
    y, sr = librosa.load(file_path, sr=None)

    # compute 13 mfcc coefficients
    # mfcc shape is (n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # take mean across time to get global timbral summary
    mfcc_mean = np.mean(mfcc, axis=1)

    # take standard deviation to capture variation over time
    # this helps preserve some dynamics lost in simple averaging
    mfcc_std = np.std(mfcc, axis=1)

    # concatenate mean and std to form final 26-d feature vector
    return np.concatenate([mfcc_mean, mfcc_std])


def load_dataset(data_dir):
    """
    iterate through genre folders and build the full dataset.

    returns:
    - feature matrix (num_samples x 26)
    - label array (num_samples,)
    """

    features = []
    labels = []
    skipped = 0

    # iterate over genre folders (sorted for consistency)
    for genre in sorted(os.listdir(data_dir)):
        genre_path = os.path.join(data_dir, genre)

        # skip anything that is not a directory
        if not os.path.isdir(genre_path):
            continue

        # iterate through audio files inside each genre
        for i, filename in enumerate(sorted(os.listdir(genre_path))):

            # only process .wav files
            if not filename.lower().endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, filename)

            try:
                # extract features for current file
                feat = extract_features(file_path)

                features.append(feat)
                labels.append(genre)

            except Exception as e:
                # sometimes gtzan contains corrupted or unreadable files
                # instead of crashing the pipeline, we skip and log them
                skipped += 1
                print(f"[skip] {file_path} ({type(e).__name__}: {e})")
                continue

            # simple progress logging so we know it's working
            if i % 10 == 0:
                print(f"processed {i} files in {genre}")

    print(f"done. loaded {len(features)} files. skipped {skipped} files.")

    return np.array(features), np.array(labels)