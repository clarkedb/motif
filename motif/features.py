from typing import Tuple

import librosa
from librosa.feature.spectral import zero_crossing_rate
import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from tonnetz import get_ton
from rhythm import ac_peaks

import json
from data import genre_dataframe, get_wav_filepath
import pandas as pd

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


def get_zero_crossing_rate(y, get_mean=True):
    """
    Compute the Zero Crossing Rate (ZCR)
    :param y: np.ndarray [shape=(n,)]
    Sampling rate of y
    :param get_mean: bool
    Whether to instead return the mean of ZCR over all frames
    :return: np.ndarray [shape=(1,t)] or float
    ZCR for each frame, or the mean ZCR
    """
    zcrs = librosa.feature.zero_crossing_rate(y=y)
    if get_mean:
        return zcrs.mean()
    else:
        return zcrs


def get_frequency_range(y, sr, eps=0.01):
    """
    Compute the range of frequencies in majority interval
    :param y: np.ndarray [shape=(n,)]
    Audio time series
    :param sr: number > 0
    Sampling rate of y
    :param eps: float in (0.0, 0.5)
    The percentage of frequencies to ignore in each tail
    :return: (float, float)
    The min and max frequency in the interval perscribed by eps
    """
    if eps >= 0.5 or eps <= 0:
        raise ValueError('The interval must be in the interval (0.0,0.5)')

    max_freq = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=(1-eps)).max()
    min_freq = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=(eps)).min()

    return min_freq, max_freq


def get_mfcc(y, sr, n_mfcc=20, get_mean=True):
    """
    Calculate the Mel Frequency Cepstral Coefficients (MFCCs)
    :param y: np.ndarray [shape=(m,)]
    Audio time series
    :param sr: number > 0
    Sampling rate of y
    :param n_mfcc: int > 0
    Number of coefficients to calcluat per frame
    :param get_mean: bool
    If true, returns the average for each coefficient
    :return: np.ndarray [shape=(n,t)] or np.ndarray [shape=(n,)]
    The MFCC of each time frame or average of each MFCC across frames
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if get_mean:
        return mfcc.mean(axis=1)
    else:
        return mfcc


class FeatureProcessor:
    def __init__(self):
        with open("../config/feature_config.json", "r") as read_file:
            self.config = json.load(read_file)

    def process_file(self, filepath):
        """Processes the given file creating features"""
        try:
            y, sr = librosa.load(filepath, duration=self.config['file-duration'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file or directory: '{filepath}'")
        except:
            raise ValueError(f"File must be wav file or mp3 file")

        features = np.array([])

        # zero crossing rate
        zcr = get_zero_crossing_rate(y, get_mean=self.config['zero-crossing-rate']['use-mean'])
        features = np.append(features, zcr)

        # frequency range
        freq_range = np.array(get_frequency_range(y, sr, eps=self.config['frequency-range']['eps']))
        features = np.append(features, freq_range)

        # mfcc
        mfcc = get_mfcc(y, sr, n_mfcc=self.config['mfcc']['n'], get_mean=self.config['mfcc']['use-mean'])
        features = np.append(features, mfcc)

        # Tempo autocorrelation peaks (top three)
        tempo = ac_peaks(y, sr)
        features = np.append(features, tempo)

        # Tonnetz
        ton = get_ton(y, sr)
        features = np.append(features, ton)

        return features

    def feature_list(self):
        """Returns a list of the feature labels"""
        fl = ['zcr', 'min_freq', 'max_freq']

        for i in range(self.config['mfcc']['n']):
            fl.append(f'mfcc{i+1}')

        for i in range(self.config['tempo']['n']):
            fl.append(f'tempo{i+1}')

        for i in range(self.config['tonnetz']['n']):
            fl.append(f'tonnetz{i+1}')

        return fl

    def process_df(self, df):
        """Takes a dataframe of track_ids and genre labels and creates the
        features for each track appending them as columns to the dataframe.
        """
        columns = self.feature_list()

        # placeholder
        features = np.empty_like(columns)

        for track_id in df.track_id:
            feature_array = self.process_file(get_wav_filepath(track_id))
            features = np.vstack((features, feature_array))

        # remove dummy row and make df
        features = features[1:]
        fdf = pd.DataFrame(features, columns=columns)

        fdf['track_id'] = df['track_id']
        fdf['genre_code'] = df['genre'].apply(lambda gen : self.config['genre-codes'][gen])

        return fdf
