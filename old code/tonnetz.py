# Import packages
import librosa
import numpy as np
from librosa.feature import tonnetz
from librosa import display
from matplotlib import pyplot as plt
from matplotlib import style, rcParams
style.use('seaborn')
rcParams['figure.figsize'] = (16, 8)


def get_ton(data, rate, plot=False):
    """
        Return the tonal centroids at each time step/frame as a list of 1xt arrays.
        The rows of the array correspond to the following:
            0: Fifth x-axis
            1: Fifth y-axis
            2: Minor x-axis
            3: Minor y-axis
            4: Major x-axis
            5: Major y-axis
    """

    # Get the tonnetz
    ton = tonnetz(data, rate)[:, :1000]

    # If requested, plot
    if plot:
        display.specshow(ton, x_axis='time', y_axis='tonnetz', cmap='coolwarm')
        plt.colorbar()
        plt.title('Tonal Centroids (Tonnetz)')
        plt.show()

    # Separate into a list of arrays truncated to time index 1000 and return the list
    ton_list = []
    for k in range(ton.shape[0]):
        ton_list.append(np.mean(ton[k, :]))
        ton_list.append(np.var(ton[k, :]))

    return ton_list


if __name__ == '__main__':

    filepath = './../data/wavs/010527.wav'
    data, rate = librosa.load(filepath)
    test = get_ton(data, rate, plot=True)
    test = get_ton(data, rate)
    for k in range(len(test)):
        print(test[k])
