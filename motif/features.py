from typing import Tuple

import librosa
import numpy as np
import scipy.linalg as la
import scipy.stats as stats

# coefficients from: http://rnhart.net/articles/key-finding/
major_coeffs = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
minor_coeffs = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
major = la.circulant(stats.zscore(major_coeffs)).T
minor = la.circulant(stats.zscore(minor_coeffs)).T

# map an index 0-11 to a key
keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def map_key(is_major: bool, index: int) -> str:
    return keys[index] + " " + ("major" if is_major else "minor")


def find_key(y: np.ndarray, sr: int) -> Tuple[bool, int]:
    """
    Estimate the major or minor key of the input audio sample
    :param y: np.ndarray [shape=(n,)]
    Audio time series
    :param sr: number > 0
    Sampling rate of y
    :return: (bool, int)
    Whether the sample is in a major key (as opposed to a minor key)
    Key of the audio sample
    """
    # compute the chromagram of the audio sample
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    # find the average of each pitch over the entire audio sample
    average_pitch = chroma_cq.mean(axis=1)

    # Krumhansl-Schmuckler algorithm (key estimation)
    x = stats.zscore(average_pitch)
    major_corr, minor_corr = major.dot(x), minor.dot(x)
    major_key, minor_key = major_corr.argmax(), minor_corr.argmax()

    # determine if the key is major or minor
    print(major_key, minor_key)
    is_major = major_corr[major_key] > minor_corr[minor_key]

    return is_major, major_key if is_major else minor_key


def get_spectral_rolloff(y: np.ndarray, sr: int, roll_percent, get_stats=False):
    """
    Compute the spectral rolloff for each frame
    :param y: np.ndarray [shape=(n,)]
    Audio time series
    :param sr: number > 0
    Sampling rate of y
    :param roll_percent: float [0 < roll_percent < 1]
    Roll-off percentage
    :param get_stats: bool
    Whether to instead return the mean and variance of the spectral rolloff
    :return: np.ndarray [shape=(1,t)] or (float, float)
    Roll-off frequency for each frame, or the mean and variance
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)
    if get_stats:
        return rolloff.mean(), rolloff.var()
    else:
        return rolloff
