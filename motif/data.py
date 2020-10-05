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
