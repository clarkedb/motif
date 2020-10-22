import pandas as pd
import os


def get_mp3_filenames(directory="../data/fma_small"):
    """
    Get the path of each mp3 file under the given root directory
    :param directory: root directory to search
    :return: List[String]
    """
    filenames = [
        os.path.join(root, file) for root, _, f in os.walk(directory) for file in f
    ]
    return list(filter(lambda s: s.endswith(".mp3"), filenames))


def get_genre_metadata(tracks="../data/fma_metadata/tracks.csv"):
    """
    Return genre metadata (indexed by track ID) for the small FMA dataset
    :param tracks: location of tracks.csv metadata
    :return: pandas.Series
    """
    tracks = pd.read_csv(tracks, header=[0, 1, 2]).set_index(
        ("Unnamed: 0_level_0", "Unnamed: 0_level_1", "track_id")
    )
    small = tracks[tracks["set", "subset", "Unnamed: 32_level_2"] == "small"]
    return small["track", "genre_top", "Unnamed: 40_level_2"]


def generate_genre_dataframe(tracks="../data/fma_metadata/tracks.csv", outfile="../data/genres.csv"):
    """
    Creates a genre dataframe and stores it as a csv.
    """
    genre_series = get_genre_metadata(tracks)
    df = pd.DataFrame({'track_id': genre_series.index.values, 'genre': genre_series.values})
    df.to_csv(outfile)
    return


def genre_dataframe(filename="../data/genres.csv"):
    """
    Loads the genre csv as a pandas dataframe.
    """
    df = pd.read_csv(filename, header=0, index_col=0)
    return df


def get_wav_filepath(track_id):
    track_file = "%06.i" % track_id + '.wav'

    return "../data/wavs/" + track_file
