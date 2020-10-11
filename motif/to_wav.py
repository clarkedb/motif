# Import packages
import subprocess
import glob
import os


def mp3_to_wav(show_progress=True):
    """
        Find all the mp3 files in any subdirectory of data/fma_small and convert them to wave
        files. For this function to work, the directory structure must fit the following:

        motif
            data
                fma_metadata
                fma_small
                wavs
            motif

        Under this structure, this script should be in the motif/motif directory, and there must
        be a motif/data/wavs directory that is empty when the script is initially run.
    """

    # Define a devnull var to supress subprocess output
    devnull = open(os.devnull, 'w')

    # Get a list of the filepath for each of the mp3 files in each subdirectory of data/fma_small
    file_list = glob.glob('./../data/fma_small/*/*.mp3')

    # Get the number of files N and initialize a counter
    N = len(file_list)
    counter = 0

    # For each file/filepath, convert that file to wav format and save it to data/wavs/*/*.wav (so as a wave file)
    for filepath in file_list:

        # Every 100 file conversions, print a progress update
        if counter % 50 == 49 and show_progress:
            progress = str(round(100 * counter / N, 2))
            print('File conversion ' + progress + '% complete.')

        # Get the file name from the path and define a new path for the wav file
        file_name = filepath[24:-4]
        new_path = './../data/wavs/' + file_name + '.wav'

        # Call the subprocess using ffmpeg to convert the file to wav format (and supress all the output)
        subprocess.call(['ffmpeg', '-i', filepath, new_path], stdout=devnull, stderr=devnull)

        # Increment the counter
        counter += 1


##########
##########


if __name__ == '__main__':
    mp3_to_wav()
